import numpy as np
import torch
import deepxde.nn.pytorch as nn
import sys
import json

sys.path.append("..")
from .submodel import SubModel
from .subdata import PDE, Function
from .pointset2d import PointSets2D
from deepxde import utils

from deepxde.icbc import PointSetBC
import time
from shapely.geometry import Polygon, Point, LineString
from shapely.geometry.multilinestring import MultiLineString
import os
import networkx as nx
from . import config_pinnadd
from profilehooks import profile

class PINNADD:
    def __init__(self, net, data, save_dir=None):
        config_pinnadd.set_random_seed(config_pinnadd.random_seed)
        self.domain_num = len(net)
        self.net_layers = net
        self.data = data
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if "pde" in data.keys():
            self.data_type = "pde"
            self.init_pde(data)
        elif "function" in data.keys():
            self.data_type = "function"
            self.init_function(data)
        else:
            raise ValueError("No PDE or function is provided.")


    def init_function(self, data):
        self.filtered_col_points_for_each_polygon = None
        self.filtered_boundary_points_for_each_polygon = None
        self.polygon_area = None
        self.polygon_network = None
        self.if_adjust_subdomain = data["adjust_subdomain"] if "adjust_subdomain" in data.keys() else False
        self.initial_subdomain_balance = data[
            "initial_subdomain_balance"] if "initial_subdomain_balance" in data.keys() else False
        self.analyze_gradient = False
        init_divide_type = "balance" if self.initial_subdomain_balance else "distance"

        self.function = data["function"]
        self.num_domain = data["num_domain"] if "num_domain" in data.keys() else None
        self.num_boundary = None
        self.n_clusters = data["n_clusters"] if "n_clusters" in data.keys() else 100
        self.interface_type = data["interface_type"] if "interface_type" in data.keys() else "line"

        self.mannual_centers = data["mannual_centers"] if "mannual_centers" in data.keys() else None
        self.mannual_scales = data["mannual_scales"] if "mannual_scales" in data.keys() else None

        self.pointset = data["pointset"]  # PointSets2D class

        self.icbc = []
        self.t_span = None
        self.t_steps = None
        self.loss_weights = None
        self.ref_data = None
        self.ref_sol = None
        self.global_pretrain_model = None

        # initial nets
        activation = "sin"
        initializer = "Glorot normal"
        self.nets = []
        for n in self.net_layers:
            self.nets.append(nn.FNN(n, activation, initializer))

        self.data_col, self.data_bc, self.data_inter, self.connection, self.subdomain_polygons = (
            self.init_subdomain(subdomain_divide_type=init_divide_type, filter_points_type=self.interface_type))


        # initial pde for each domain
        self.data = []
        for i in range(self.domain_num):
            col_pts = self.data_col[i]
            inter_pts = []
            for j in range(len(self.connection)):
                con = self.connection[j]
                if i in con:
                    inter_pts.append(self.data_inter[j])
                else:
                    inter_pts.append(None)

            function_domain = Function(col_pts,
                             self.function)

            self.data.append(function_domain)

        self.models = []
        for i in range(self.domain_num):
            self.models.append(SubModel(self.data[i], self.nets[i]))
            if self.global_pretrain_model is not None:
                self.models[i].net.load_state_dict(self.global_pretrain_model.net.state_dict())

        print("PINNADD model is initialized.")

    def init_pde(self, data):
        self.filtered_col_points_for_each_polygon = None
        self.filtered_boundary_points_for_each_polygon = None
        self.polygon_area = None
        self.polygon_network = None
        self.if_adjust_subdomain = data["adjust_subdomain"] if "adjust_subdomain" in data.keys() else False
        self.initial_subdomain_balance = data[
            "initial_subdomain_balance"] if "initial_subdomain_balance" in data.keys() else False
        self.analyze_gradient = data["analyze_gradient"] if "analyze_gradient" in data.keys() else False
        self.ignore_inter_pde_loss = data["ignore_inter_pde_loss"] if "ignore_inter_pde_loss" in data.keys() else False
        init_divide_type = "balance" if self.initial_subdomain_balance else "distance"

        self.pde = data["pde"]
        self.num_boundary = data["num_boundary"] if "num_boundary" in data.keys() else None
        self.num_domain = data["num_domain"] if "num_domain" in data.keys() else None
        self.n_clusters = data["n_clusters"] if "n_clusters" in data.keys() else 100
        self.interface_type = data["interface_type"] if "interface_type" in data.keys() else "line"
        self.mannual_centers = data["mannual_centers"] if "mannual_centers" in data.keys() else None
        self.mannual_scales = data["mannual_scales"] if "mannual_scales" in data.keys() else None

        self.pointset = data["pointset"] # PointSets2D class
        self.pointset_bc_points = None
        self.pointset_bc_vals = None
        self.icbc = data["icbc"]
        for constraint in self.icbc: # only 1 pointset bc is supported
            if isinstance(constraint, PointSetBC):
                self.pointset_bc_points = constraint.points
                self.pointset_bc_vals = constraint.values.cpu().detach().numpy()

        self.t_span = data["t_span"] if "t_span" in data.keys() else None
        self.t_steps = data["t_steps"] if "t_steps" in data.keys() else None

        self.loss_weights = data["loss_weights"] if "loss_weights" in data.keys() else None
        self.ref_data = data["ref_data"] if "ref_data" in data.keys() else None
        self.ref_sol = data["ref_sol"] if "ref_sol" in data.keys() else None
        self.global_pretrain_model = None
        activation = "sin"
        initializer = "Glorot normal"
        self.nets = []
        for n in self.net_layers:
            self.nets.append(nn.FNN(n, activation, initializer))

        self.data_col, self.data_bc, self.data_inter, self.connection, self.subdomain_polygons = (
            self.init_subdomain(subdomain_divide_type=init_divide_type, filter_points_type=self.interface_type))
        # move pointset properties to self

        # initial nets


        # initial pde for each domain
        self.data = []
        for i in range(self.domain_num):
            col_pts = self.data_col[i]
            bc_pts = None
            if self.data_bc is not None:
                bc_pts = self.data_bc[i]

            inter_pts = []
            for j in range(len(self.connection)):
                con = self.connection[j]
                if i in con:
                    inter_pts.append(self.data_inter[j])
                else:
                    inter_pts.append(None)

            pointset_bc = None
            if self.pointset_bc_points is not None:
                pointset_bc_idx = [idx for idx in range(self.pointset_bc_points.shape[0]) if (self.subdomain_polygons[i].contains(Point(self.pointset_bc_points[idx]))
                                   or self.subdomain_polygons[i].boundary.contains(Point(self.pointset_bc_points[idx]))) ]
                if len(pointset_bc_idx) > 0:
                    pts = self.pointset_bc_points[pointset_bc_idx]
                    vals = self.pointset_bc_vals[pointset_bc_idx]
                    pointset_bc = [pts, vals]

            loss_w = None
            if self.loss_weights is not None:
                loss_w = self.loss_weights[i]

            pde_domain = PDE(col_pts,
                             self.pde,
                             bc_pts=bc_pts,
                             inter_pts=inter_pts,
                             icbc=self.icbc,
                             pointset_bc=pointset_bc,
                             loss_weights=loss_w,
                             ref_data=self.ref_data,
                             ref_sol=self.ref_sol,
                             t_span=self.t_span,
                             t_steps=self.t_steps,
                             ignore_inter_pde_loss=self.ignore_inter_pde_loss,
                             analyze_gradient=self.analyze_gradient)

            self.data.append(pde_domain)

        self.models = []
        for i in range(self.domain_num):
            self.models.append(SubModel(self.data[i], self.nets[i]))
            if self.global_pretrain_model is not None:
                self.models[i].net.load_state_dict(self.global_pretrain_model.net.state_dict())

        print("PINNADD model is initialized.")

    def init_subdomain(self, subdomain_divide_type="distance", filter_points_type="line"):
        plot_config = {
            "voronoi": [True, self.save_dir],
            "polygons": [True, self.save_dir],
            "points_cluster": [True, self.save_dir],
            "subdomain": [True, self.save_dir],
            "filtered_points": [True, self.save_dir]
        }
        self.pointset.plot_config = plot_config
        self.pointset.adjust_count = 0
        print("Initializing subdomains...")
        self.pointset.set_random_seed(config_pinnadd.random_seed)
        np.random.seed(config_pinnadd.random_seed)
        distribution = "pseudo"
        self.col_points_all = self.pointset.geom.random_points(self.num_domain, random=distribution)
        if self.num_boundary is not None:
            self.boundary_points_all = self.pointset.geom.random_boundary_points(self.num_boundary, random=distribution)
            self.train_points_all = np.vstack([self.col_points_all, self.boundary_points_all])
        else:
            self.boundary_points_all = None
            self.train_points_all = self.col_points_all

        if self.data_type == "function":
            self.global_function = Function(self.col_points_all, self.function)
        elif self.data_type == "pde":
            self.global_pde = PDE(self.col_points_all,
                                  self.pde,
                                  bc_pts=self.boundary_points_all,
                                  inter_pts=[],
                                  icbc=self.icbc,
                                  pointset_bc=[self.pointset_bc_points,
                                               self.pointset_bc_vals] if self.pointset_bc_points is not None else None, # todo
                                  loss_weights=self.loss_weights[0] if self.loss_weights is not None else None,
                                  ref_data=self.ref_data,
                                  ref_sol=self.ref_sol,
                                  t_span=self.t_span,
                                  t_steps=self.t_steps,
                                  ignore_inter_pde_loss=self.ignore_inter_pde_loss,
                                  analyze_gradient=False)

        self.pointset.vor = self.pointset.voronoi(self.train_points_all, n_clusters=self.n_clusters, if_plot=True)
        self.pointset.polygons = self.pointset.get_polygons(self.pointset.vor, if_plot=True)
        self.pointset.patch_center_points = self.pointset.vor.points

        if subdomain_divide_type == "distance" or subdomain_divide_type == "balance":
            self.pointset.subdomain_polygons, self.pointset.polygon_labels \
                = self.pointset.patch_cluster_by_distance(self.domain_num,
                                                          self.pointset.patch_center_points,
                                                          mannual_centers=self.mannual_centers,
                                                          mannual_scales=self.mannual_scales,
                                                          if_plot=True)
        else:
            raise ValueError("Subdomain divide type is not supported yet.")

        if subdomain_divide_type == "balance":
            self.global_pretrain_model = self.pretrain()
            cost_for_each_polygon = self.calculate_cost_for_each_polygon(self.global_pretrain_model)
            self.pointset.subdomain_polygons, self.pointset.polygon_labels \
                = self.pointset.patch_cluster_weighted_balance(self.domain_num,
                                                          cost_for_each_polygon,
                                                          self.polygon_network,
                                                          self.pointset.polygon_labels,
                                                          # mannual_centers=self.mannual_centers,
                                                          if_plot=True)


        if filter_points_type == "line":
            self.pointset.filtered_col_points, self.pointset.filtered_boundary_points, self.pointset.interface_pts, self.pointset.connection = (
                self.pointset.filter_points_interface_line(self.pointset.subdomain_polygons, self.col_points_all, self.boundary_points_all, if_plot=True))
        elif filter_points_type == "band":
            self.pointset.filtered_col_points, self.pointset.filtered_boundary_points, self.pointset.interface_pts, self.pointset.connection = (
                self.pointset.filter_points_interface_band(self.pointset.subdomain_polygons, self.pointset.polygons, self.pointset.polygon_labels,
                                                           self.col_points_all, self.boundary_points_all, if_plot=True))

        print("Subdomains are initialized.")
        return self.pointset.filtered_col_points, self.pointset.filtered_boundary_points, self.pointset.interface_pts, self.pointset.connection, self.pointset.subdomain_polygons


    def pretrain(self, iterations=1000):
        iterations = config_pinnadd.pretrain_iterations
        if self.data_type == "pde":
            global_pretrain_model = SubModel(self.global_pde, self.nets[0])
        elif self.data_type == "function":
            global_pretrain_model = SubModel(self.global_function, self.nets[0])
        print("Start pretraining...")
        time_start = time.time()
        for i in range(iterations):
            global_pretrain_model.train_sgd()

        time_end = time.time()
        train_time = time_end - time_start
        print("Total time for pretraining: ", train_time)
        return global_pretrain_model

    def calculate_cost_for_each_polygon(self, models):
        print("Calculating cost for each polygon...")
        polygons = self.pointset.polygons

        if self.polygon_area is None:
            self.polygon_area = np.array([polygons[i].area for i in range(len(polygons))])

        if self.polygon_network is None:
            self.polygon_network = nx.Graph()
            for i in range(len(polygons)):
                self.polygon_network.add_node(i)
            for i in range(len(polygons)):
                for j in range(i+1, len(polygons)):
                    intersect = polygons[i].intersection(polygons[j])
                    if isinstance(intersect, LineString) or isinstance(intersect, MultiLineString):
                        self.polygon_network.add_edge(i, j)
            print("Polygon network is built.")

        cluster_all_points = self.pointset.cluster_all_points
        # filter过程比较费时
        if self.col_points_all is not None and self.filtered_col_points_for_each_polygon is None:
            self.filtered_col_points_for_each_polygon = self.pointset.cluster_all_points_inside

        if self.boundary_points_all is not None and self.filtered_boundary_points_for_each_polygon is None:
            self.filtered_boundary_points_for_each_polygon = self.pointset.cluster_all_points_boundary

        loss_for_polygons = []
        for i in range(len(polygons)):
            if self.data_type == "pde":
                if isinstance(models, list):
                    loss_for_each_polygon_list = []
                    for j in range(len(models)):
                        loss_for_each_polygon = (
                            self.global_pde.loss_all_for_inputs(
                                models[j], models[j].loss_fn,
                                self.filtered_col_points_for_each_polygon[i],
                                self.filtered_boundary_points_for_each_polygon[i]))
                        loss_for_each_polygon_list.append(utils.to_numpy(loss_for_each_polygon))
                    loss_for_each_polygon = np.hstack(loss_for_each_polygon_list)

                else:
                    loss_for_each_polygon = (
                            self.global_pde.loss_all_for_inputs(
                                models, models.loss_fn,
                                self.filtered_col_points_for_each_polygon[i],
                                self.filtered_boundary_points_for_each_polygon[i]))
                    loss_for_each_polygon = [utils.to_numpy(loss_for_each_polygon)]
                loss_for_polygons.append(loss_for_each_polygon)
            elif self.data_type == "function":
                if isinstance(models, list):
                    loss_for_each_polygon_list = []
                    for j in range(len(models)):
                        loss_for_each_polygon = (
                            self.global_function.loss_all_for_inputs(
                                models[j], models[j].loss_fn,
                                self.filtered_col_points_for_each_polygon[i]))
                        loss_for_each_polygon_list.append(utils.to_numpy(loss_for_each_polygon))
                    loss_for_each_polygon = np.hstack(loss_for_each_polygon_list)
                else:
                    loss_for_each_polygon = (
                            self.global_function.loss_all_for_inputs(
                                models, models.loss_fn,
                                self.filtered_col_points_for_each_polygon[i]))
                    loss_for_each_polygon = [utils.to_numpy(loss_for_each_polygon)]
                loss_for_polygons.append(loss_for_each_polygon)

            else:
                raise ValueError("No PDE or function is provided.")

        loss_for_polygons = np.array(loss_for_polygons)
        cost_for_each_polygon = loss_for_polygons * self.polygon_area[:, np.newaxis]

        # plot

        return cost_for_each_polygon

    def adjust_subdomain(self):
        cost_for_each_polygon = self.calculate_cost_for_each_polygon(self.models)
        self.pointset_subdomain_polygons, self.pointset.polygon_labels \
            = self.pointset.patch_cluster_weighted_min(self.domain_num,
                                                       cost_for_each_polygon,
                                                       self.polygon_network,
                                                       self.pointset.polygon_labels, # previous labels
                                                       if_plot=True)
        self.subdomain_polygons = self.pointset_subdomain_polygons
        if self.interface_type == "band":
            self.pointset.filtered_col_points, self.pointset.filtered_boundary_points, \
            self.pointset.interface_pts, self.pointset.connection = (
            self.pointset.filter_points_interface_band(self.pointset.subdomain_polygons,
                                                       self.pointset.polygons,
                                                       self.pointset.polygon_labels,
                                                       self.col_points_all,
                                                       self.boundary_points_all,
                                                       if_plot=True))

            self.data_col, self.data_bc, self.data_inter, self.connection = (
                self.pointset.filtered_col_points, self.pointset.filtered_boundary_points,
                self.pointset.interface_pts, self.pointset.connection)

        else:
            raise ValueError("Interface type is not supported yet.")

        # update data
        self.data = []
        for i in range(self.domain_num):
            col_pts = self.data_col[i]
            inter_pts = []
            for j in range(len(self.connection)):
                con = self.connection[j]
                if i in con:
                    inter_pts.append(self.data_inter[j])
                else:
                    inter_pts.append(None)

            loss_w = None
            if self.loss_weights is not None:
                loss_w = self.loss_weights[i]

            if self.data_type == "pde":
                bc_pts = None
                if self.data_bc is not None:
                    bc_pts = self.data_bc[i]

                pointset_bc = None
                if self.pointset_bc_points is not None:
                    pointset_bc_idx = [idx for idx in range(self.pointset_bc_points.shape[0]) if
                                       (self.subdomain_polygons[i].contains(Point(self.pointset_bc_points[idx]))
                                        or self.subdomain_polygons[i].boundary.contains(Point(self.pointset_bc_points[idx])))]
                    if len(pointset_bc_idx) > 0:
                        pts = self.pointset_bc_points[pointset_bc_idx]
                        vals = self.pointset_bc_vals[pointset_bc_idx]
                        pointset_bc = [pts, vals]

                pde_domain = PDE(col_pts,
                                 self.pde,
                                 bc_pts=bc_pts,
                                 inter_pts=inter_pts,
                                 icbc=self.icbc,
                                 pointset_bc=pointset_bc,
                                 loss_weights=loss_w,
                                 ref_data=self.ref_data,
                                 ref_sol=self.ref_sol,
                                 t_span=self.t_span,
                                 t_steps=self.t_steps,
                                 ignore_inter_pde_loss=self.ignore_inter_pde_loss,
                                 analyze_gradient=self.analyze_gradient)
                self.data.append(pde_domain)

            elif self.data_type == "function":
                function_domain = Function(col_pts,
                                           self.function)
                self.data.append(function_domain)


        for i in range(self.domain_num):
            self.models[i].data = self.data[i]

    # @profile
    def aggregate_interface(self):
        if not hasattr(self, "pde"):
            return 0
        outputs_inter = []
        for i in range(self.domain_num):
            outputs_inter.append(self.models[i].outputs_inter())

        for k in range(len(self.connection)):
            con = self.connection[k]
            inter_vals = []
            inter_pdes = []
            for n in range(len(con)):
                idx = con[n]
                inter_vals.append(outputs_inter[idx][k][0].detach())
                if isinstance(outputs_inter[idx][k][1], list):
                    inter_pdes.append([p.detach() for p in outputs_inter[idx][k][1]])
                else:
                    inter_pdes.append(outputs_inter[idx][k][1].detach())

            for n in range(len(con)):
                idx = con[n]
                inter_val_avg = torch.mean(torch.stack(inter_vals), dim=0)

                if isinstance(inter_pdes[0], list):
                    inter_pdes_all = [torch.zeros_like(inter_pdes[0][l]) for l in range(len(inter_pdes[0]))]
                    for m in range(len(con)):
                        if m != n:
                            for l in range(len(inter_pdes[0])):
                                inter_pdes_all[l] += inter_pdes[m][l]

                        inter_pde_avg = [inter_pdes_all[l] / (len(con) - 1) for l in range(len(inter_pdes[0]))]
                else:
                    inter_pdes_all = torch.zeros_like(inter_pdes[0])
                    for m in range(len(con)):
                        if m != n:
                            inter_pdes_all += inter_pdes[m]

                    inter_pde_avg = inter_pdes_all / (len(con) - 1)

                self.models[idx].aggregate_interface[k] = [inter_val_avg, inter_pde_avg]

    def train(self, iterations, print_every=10):
        print("Start training...")
        time_start = time.time()
        max_iteration_domain = 0
        adjust_subdomain_period = config_pinnadd.adjust_subdomain_period
        adjust_subdomain_start_iteration = config_pinnadd.adjust_subdomain_start_iteration
        adjust_subdomain_end_iteration = config_pinnadd.adjust_subdomain_end_iteration
        adjust_domain_times = 1
        for iter in range(iterations):
            self.aggregate_interface()
            stop_training = [False for i in range(self.domain_num)]
            for i in range(self.domain_num):
                if self.models[i].count_iteration < iterations:
                    count_iteration = self.models[i].train_sgd()
                    if self.analyze_gradient and not config_pinnadd.static_frequency:
                        loss, test_results = self.models[i].test(count_iteration)
                        print("Iteration:", count_iteration, ", Domain:", i, ", Training Loss: ", loss, ", Test Loss: ",
                              test_results)
                    elif count_iteration % print_every == 0:
                        loss, test_results = self.models[i].test(count_iteration)
                        print("Iteration:", count_iteration, ", Domain:", i, ", Training Loss: ", loss, ", Test Loss: ", test_results)
                    max_iteration_domain = max(max_iteration_domain, count_iteration)
                    if self.if_adjust_subdomain:
                        if max_iteration_domain >= adjust_domain_times * adjust_subdomain_period + adjust_subdomain_start_iteration and \
                                max_iteration_domain < adjust_subdomain_end_iteration:
                            self.adjust_subdomain()
                            adjust_domain_times += 1
                            self.aggregate_interface()
                else:
                    stop_training[i] = True

            if all(stop_training):
                break

        print("Training finished. Total iterations: ", iterations)
        loss_dict = {}
        test_result_dict = {}
        for i in range(self.domain_num):
            loss, test_results = self.models[i].test(iterations)
            print("Domain:", i, ", Training Loss: ", loss, ", Test Loss: ", test_results)
            loss_dict[i] = loss
            test_result_dict[i] = test_results

        time_end = time.time()
        train_time = time_end - time_start
        print("Total time: ", train_time)

        result_dict = {"loss": loss_dict, "test_results": test_result_dict, "train_time": train_time}

        return result_dict

    def test(self):
        test_results = []
        for i in range(self.domain_num):
            test_pts, test_vals = self.models[i].data.test_pts, self.models[i].data.test_vals
            pred_vals = utils.to_numpy(self.models[i].outputs(False, test_pts))
            test_pts = utils.to_numpy(test_pts)
            test_vals = utils.to_numpy(test_vals)
            test_results.append([test_pts, test_vals, pred_vals])

        return test_results

    def save(self, path="./", result_dict=None):
        if not os.path.exists(path):
            os.makedirs(path)
        for i in range(self.domain_num):
            self.models[i].save(path + "/model-" + str(i))

        if result_dict is not None:
            with open(path + "result.json", "w") as f:
                json.dump(result_dict, f, indent=4)


    def restore(self, path):
        for i in range(self.domain_num):
            self.models[i].restore(path + "/model-" + str(i))

    def predict(self, testset):
        if isinstance(testset, torch.Tensor):
            testset = testset.cpu().detach().numpy()

        preds = np.zeros((testset.shape[0], self.models[0].data.test_vals.shape[1]))
        for i in range(self.domain_num):
            test_pts_idx = [idx for idx in range(testset.shape[0]) if
                               (self.subdomain_polygons[i].contains(Point(testset[idx]))
                                or self.subdomain_polygons[i].boundary.contains(Point(testset[idx])))]
            test_pts = testset[test_pts_idx]
            if len(test_pts) > 0:
                pred = self.models[i].outputs(False, torch.as_tensor(test_pts))
                pred = utils.to_numpy(pred)

                preds[test_pts_idx] = pred

        return preds
