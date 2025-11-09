from shapely.geometry.multilinestring import MultiLineString

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, LineString, MultiPolygon
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull, voronoi_plot_2d
from scipy.spatial.distance import cdist
from scipy.spatial import Voronoi
import networkx as nx
import copy
from . import config_pinnadd
from .optimize_graph import divide_graph_min, divide_graph_balance

class PointSets2D:
    def __init__(self, geom, divide_config=None, random_seed=42, plot_config=None, spatial_domain_shapely=None):
        if plot_config is None:
            plot_config = {
                "points_cluster": [True, None],  # [if_plot, save_path]
                "voronoi": [True, None],
                "polygons": [True, None],
                "subdomain": [True, None],
                "filtered_points": [True, None]
            }
        self.plot_config = plot_config
        # 有如下几个图：
        # 1. 聚类后的点
        # 2. 聚类后的点+凸包+Voronoi图
        # 3. Polygon图
        # 4. Polygon图合并后的图
        # 5. 不同区域以及各种配点的图

        self.random_seed = random_seed
        self.geom = geom
        self.spatial_domain_shapely = spatial_domain_shapely

        self.divide_config = divide_config
        if self.divide_config is not None:
            self.count_subdomain = divide_config["count_subdomain"] if "count_subdomain" in divide_config else 1
            self.num_domain = divide_config["num_domain"] if "num_domain" in divide_config else 0
            self.num_boundary = divide_config["num_boundary"] if "num_boundary" in divide_config else 0
            self.mannual_centers = divide_config["mannual_centers"] if "mannual_centers" in divide_config else None
            self.mannual_scales = divide_config["mannual_scales"] if "mannual_scales" in divide_config else None
            self.n_clusters = divide_config["n_clusters"] if "n_clusters" in divide_config else 100
            self.interface_interval = divide_config["interface_interval"] if "interface_interval" in divide_config else 0.01

        self.col_points = None
        self.boundary_points = None
        self.vor = None
        self.polygons = None
        self.subdomain_polygons = None
        self.bounding_polygon = None

        self.filtered_col_points = None
        self.filtered_boundary_points = None
        self.interface_pts = None
        self.connection = None

        self.seed_points = None
        self.polygons_weight = None # 用于加权的polygon划分，等于：loss*area
        self.cluster_all_points = None
        self.cluster_all_points_inside = None
        self.cluster_all_points_boundary = None

        self.adjust_count = 0

    def set_random_seed(self, random_seed):
        self.random_seed = random_seed

    def divide(self, distribution="pseudo", divide_config=None):
        if divide_config is not None:
            self.divide_config = divide_config
            self.count_subdomain = divide_config["count_subdomain"] if "count_subdomain" in divide_config else 1
            self.num_domain = divide_config["num_domain"] if "num_domain" in divide_config else 0
            self.num_boundary = divide_config["num_boundary"] if "num_boundary" in divide_config else 0
            self.mannual_centers = divide_config["mannual_centers"] if "mannual_centers" in divide_config else None
            self.mannual_scales = divide_config["mannual_scales"] if "mannual_scales" in divide_config else None
            self.n_clusters = divide_config["n_clusters"] if "n_clusters" in divide_config else 100
            self.interface_interval = divide_config["interface_interval"] if "interface_interval" in divide_config else 0.01

        self.col_points = self.geom.random_points(self.num_domain, random=distribution)
        self.boundary_points = self.geom.random_boundary_points(self.num_boundary, random=distribution)
        train_points = np.vstack([self.col_points, self.boundary_points])
        self.vor = self.voronoi(train_points, n_clusters=self.n_clusters)
        self.polygons = self.get_polygons(self.vor)
        self.patch_center_points = self.vor.points
        self.subdomain_polygons, self.polygon_labels = self.patch_cluster_by_distance(self.count_subdomain, self.patch_center_points, self.mannual_centers, self.mannual_scales)
        self.filtered_col_points, self.filtered_boundary_points, self.interface_pts, self.connection \
            = self.filter_points_interface_line(self.subdomain_polygons, self.col_points, self.boundary_points, self.interface_interval)

    # step 1: 点聚类，计算凸包，建立Voronoi图
    def voronoi(self, points, n_clusters=20, if_plot=False):
        # 聚类
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=self.random_seed)
        kmeans.fit(points)
        labels = kmeans.labels_

        cluster_points = [[] for _ in range(kmeans.n_clusters)]
        for point, label in zip(points, labels):
            cluster_points[label].append(point)

        cluster_all_points = [[] for _ in range(kmeans.n_clusters)]
        cluster_all_points_inside = [[] for _ in range(kmeans.n_clusters)]
        cluster_all_points_boundary = [[] for _ in range(kmeans.n_clusters)]
        for i in range(kmeans.n_clusters):
            cluster_all_points[i] = np.array(cluster_points[i])
            inside_idx = self.geom.inside(cluster_all_points[i])
            boundary_idx = self.geom.on_boundary(cluster_all_points[i])
            cluster_all_points_inside[i] = cluster_all_points[i][inside_idx]
            cluster_all_points_boundary[i] = cluster_all_points[i][boundary_idx]

        self.cluster_all_points = cluster_all_points
        self.cluster_all_points_inside = cluster_all_points_inside
        self.cluster_all_points_boundary = cluster_all_points_boundary

        centroids = kmeans.cluster_centers_

        vor = Voronoi(centroids)

        # plot
        if if_plot:
            if self.plot_config["points_cluster"][0]:
                self.plot_convex_hull(points, labels, n_clusters, save_path=self.plot_config["points_cluster"][1])
            if self.plot_config["voronoi"][0]:
                self.plot_voronoi(vor, centroids, points, labels, save_path=self.plot_config["voronoi"][1])

        return vor

    def plot_convex_hull(self, points, labels, n_clusters, save_path=None):
        fig, ax = plt.subplots(figsize=(10, 10))
        # 使用不同的颜色绘制每个聚类及其凸包
        for cluster_id in range(n_clusters):
            cluster_points = points[labels == cluster_id]

            # 计算凸包
            if len(cluster_points) > 2:
                try:
                    hull = ConvexHull(cluster_points)
                    ax.fill(cluster_points[hull.vertices, 0], cluster_points[hull.vertices, 1], alpha=0.3)
                except:
                    print("Error in ConvexHull")
                    continue


        ax.scatter(points[:, 0], points[:, 1], c=labels, cmap='tab20', s=1)
        plt.title("Clustered Points with Convex Hulls")
        plt.show()
        if save_path is not None:
            fig.savefig(save_path + "/convex_hull.png")

    def plot_voronoi(self, vor, centroids, points, labels, save_path=None):
        fig, ax = plt.subplots(figsize=(10, 10))
        voronoi_plot_2d(vor, ax=ax, show_vertices=True, show_points=True)
        # 绘制聚类中心
        ax.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', label="Centroids")

        # 绘制聚类的点
        ax.scatter(points[:, 0], points[:, 1], c=labels, cmap='tab20', s=1, alpha=0.5)

        plt.title("Voronoi Diagram with Cluster Centers")
        plt.legend(loc='best')
        plt.show()
        if save_path is not None:
            fig.savefig(save_path + "/voronoi.png")

    # step 2: 根据Voronoi图建立Polygon
    def get_polygons(self, vor, if_plot=False):
        bbox = self.geom.bbox
        bounding_points = [(bbox[0][0], bbox[0][1]), (bbox[1][0], bbox[0][1]),
                           (bbox[1][0], bbox[1][1]), (bbox[0][0], bbox[1][1])]

        bounding_polygon = Polygon(bounding_points)
        self.bounding_polygon = bounding_polygon

        points = vor.points
        vertices = vor.vertices
        regions = vor.regions
        ridge_points = vor.ridge_points
        ridge_vertices = vor.ridge_vertices

        # 计算每个 Voronoi 区域
        clipped_polygons = []
        for i, region_idx in enumerate(vor.point_region):
            region = regions[region_idx]
            region_edges = []  # 存储该区域的边界点

            # 如果区域为空或包含无穷远点（-1），需要特殊处理
            if -1 in region or len(region) == 0:
                # continue
                point = points[i]  # 当前种子点

                # 遍历所有与该点关联的 ridge
                for (p1, p2), (v1, v2) in zip(ridge_points, ridge_vertices):
                    if i not in (p1, p2):
                        continue

                    # 处理包含无穷远点的边
                    if -1 in (v1, v2):
                        finite_idx = v1 if v2 == -1 else v2
                        finite_vertex = vertices[finite_idx]

                        # 计算边的中点和方向向量
                        midpoint = (points[p1] + points[p2]) / 2
                        direction = np.array([-(points[p2][1] - points[p1][1]),
                                              points[p2][0] - points[p1][0]])  # 垂直方向

                        direction = direction / np.linalg.norm(direction)

                        near_point = finite_vertex + direction * 1e-3

                        min_dist = 1e6
                        min_idx = -1
                        for k in range(vor.point_region.shape[0]):
                            cent_point_idx = k
                            cent_point = vor.points[k]
                            dist = np.linalg.norm(cent_point - near_point)
                            if dist < min_dist:
                                min_dist = dist
                                min_idx = cent_point_idx

                        if min_idx not in (p1, p2):
                            direction = -direction

                        # 将有限点沿方向延伸，构造射线
                        far_point = finite_vertex + direction * 1e6
                        edge = LineString([finite_vertex, far_point])
                        # 与求解域的边界求交点
                        intersection = edge.intersection(bounding_polygon)
                        if isinstance(intersection, Point):
                            region_edges.append(finite_vertex)
                            region_edges.append(intersection.coords[0])
                        elif isinstance(intersection, LineString):
                            region_edges.append(finite_vertex)
                            region_edges.extend(intersection.coords)
                    else:
                        # 普通边：直接裁剪
                        edge = LineString([vertices[v1], vertices[v2]])
                        intersection = edge.intersection(bounding_polygon)
                        if isinstance(intersection, LineString):
                            region_edges.extend(intersection.coords)

            else:
                # 没有无穷远点的普通区域
                region_edges = [vertices[v] for v in region]

            # 构造封闭多边形
            if len(region_edges) > 0:
                for b_pts in bounding_points:
                    min_dist = 1e6
                    min_idx = -1
                    for k in range(vor.point_region.shape[0]):
                        cent_point_idx = k
                        cent_point = vor.points[k]
                        dist = np.linalg.norm(cent_point - b_pts)
                        if dist < min_dist:
                            min_dist = dist
                            min_idx = cent_point_idx
                    if min_idx == i:
                        region_edges.append(b_pts)
                region_edges = np.array(region_edges)
                hull = ConvexHull(region_edges)
                sorted_points = region_edges[hull.vertices]
                region_polygon = Polygon(sorted_points).intersection(bounding_polygon)
                if not region_polygon.is_empty:
                    if self.spatial_domain_shapely is not None:
                        region_polygon = region_polygon.intersection(self.spatial_domain_shapely)
                    if not region_polygon.is_empty:
                        clipped_polygons.append(region_polygon)

        # plot
        if if_plot:
            if self.plot_config["polygons"][0]:
                self.plot_polygons(clipped_polygons, bounding_polygon, points, save_path=self.plot_config["polygons"][1])

        return clipped_polygons

    def plot_polygons(self, polygons, bounding_polygon, points, save_path=None):
        fig, ax = plt.subplots(figsize=(10, 10))
        x, y = bounding_polygon.exterior.xy
        ax.plot(x, y, 'k--', label="Bounding Polygon")

        for poly in polygons:
            x, y = poly.exterior.xy
            ax.fill(x, y, alpha=0.4)

        plt.scatter(points[:, 0], points[:, 1], color='red', label='Input Points')

        ax.legend(loc='best')
        ax.axis('equal')
        plt.title("Polygons")
        plt.show()
        if save_path is not None:
            fig.savefig(save_path + "/polygons.png")

    def plot_polygon_network(self, polygons, bounding_polygon, points, network, cost, save_path=None):
        fig, ax = plt.subplots(figsize=(10, 10))
        x, y = bounding_polygon.exterior.xy
        ax.plot(x, y, 'k--', label="Bounding Polygon")

        # for poly in polygons:
        #     x, y = poly.exterior.xy
        #     ax.fill(x, y, alpha=0.4, color='lightgray')
        max_cost = np.max(cost)
        cost = list( (1 - cost / max_cost) ** 10 )
        for i in range(len(polygons)):
            poly = polygons[i]
            x, y = poly.exterior.xy
            ax.fill(x, y, alpha=cost[i][0], color='gray')

        plt.scatter(points[:, 0], points[:, 1], color='red', label='Input Points')

        for u, v in network.edges():
            loc_u = [points[u, 0], points[u, 1]]
            loc_v = [points[v, 0], points[v, 1]]
            plt.plot([loc_u[0], loc_v[0]], [loc_u[1], loc_v[1]], color='blue', alpha=0.5)

        ax.axis('equal')
        plt.title("Polygon Network")
        plt.show()
        if save_path is not None:
            fig.savefig(save_path + "/polygon_network.png")

    # step 3: 对patch的中心进行聚类，得到多个subdomain
    def patch_cluster_by_distance(self, count_subdomain, patch_center_points, mannual_centers=None, mannual_scales=None, if_plot=False):
        n_clusters = count_subdomain
        if mannual_centers is not None:
            distances = cdist(patch_center_points, mannual_centers)

            if mannual_scales is not None:
                scales = np.array(mannual_scales)
            else:
                scales = np.ones((1, n_clusters))

            # 对每个中心，根据尺度参数计算高斯加权
            weights = np.exp(- (distances ** 2) / (2 * scales ** 2))

            # 为每个点选择加权距离最小的中心
            labels = np.argmax(weights, axis=1)
            centroids = np.array(mannual_centers)

        else:
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=self.random_seed)
            kmeans.fit(patch_center_points)
            labels = kmeans.labels_

            centroids = kmeans.cluster_centers_

        subdomain_polygons = []
        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            ploygons = [x for x, m in zip(self.polygons, mask) if m]
            merged_polygon = self.merge_polygons(ploygons)
            subdomain_polygons.append(merged_polygon)

        if if_plot:
            if self.plot_config["subdomain"][0]:
                self.plot_subdomains(subdomain_polygons, centroids, save_path=self.plot_config["subdomain"][1], type="distance")

        return subdomain_polygons, labels

    def patch_cluster_weighted_min(self, count_subdomain, polygon_weights, polygon_network, polygon_labels, if_plot=False):
        polygon_num = len(polygon_weights)
        # polygon_weights = polygon_weights / np.sum(polygon_weights)
        # 根据标签划分子图
        def gen_subgraphs(polygon_labels):
            subgraphs = {}
            for node, label in enumerate(polygon_labels):
                if label not in subgraphs:
                    subgraphs[label] = set()
                subgraphs[label].add(node)
            return subgraphs

        # 判断是否为边界节点
        def find_boundary_nodes(polygon_network, polygon_labels):
            is_boundary_nodes = [False for i in range(polygon_num)] # True or False
            for node_idx in range(polygon_num):
                for neighbor in polygon_network.neighbors(node_idx):
                    if polygon_labels[neighbor] != polygon_labels[node_idx]:
                        is_boundary_nodes[node_idx] = True
                        break
            return is_boundary_nodes

        # 找到每个节点的邻居中，标签不同，但该邻居所属的子图的代价更小的
        def find_min_neighbor_label(node_idx):
            current_label = polygon_labels[node_idx]
            current_cost = polygon_weights[node_idx][current_label]
            for neighbor in polygon_network.neighbors(node_idx):
                neighbor_label = polygon_labels[neighbor]
                if neighbor_label == current_label:
                    continue
                neighbor_cost = polygon_weights[node_idx][neighbor_label]
                if neighbor_cost < current_cost:
                    current_label = neighbor_label
                    current_cost = neighbor_cost
            return current_label

        # 检查移动节点后是否会破坏连通性
        def check_connectivity_after_move(graph, node, current_subgraph_nodes, target_subgraph_nodes):
            current_subgraph_nodes = copy.deepcopy(current_subgraph_nodes)
            target_subgraph_nodes = copy.deepcopy(target_subgraph_nodes)
            new_current_subgraph = current_subgraph_nodes - {node}
            if not nx.is_connected(graph.subgraph(new_current_subgraph)):
                return False  # 当前子图不再连通
            new_target_subgraph = target_subgraph_nodes | {node}
            if not nx.is_connected(graph.subgraph(new_target_subgraph)):
                return False  # 目标子图不再连通
            return True  # 不会破坏连通性

        subgraphs = gen_subgraphs(polygon_labels)
        is_boundary_nodes = find_boundary_nodes(polygon_network, polygon_labels)
        init_boundary_length = sum(is_boundary_nodes)
        max_boundary_length = init_boundary_length + config_pinnadd.boundary_length_tolerance2
        changed = True
        max_iter_times = config_pinnadd.max_iter_times_weighted_min
        iter_times = 0
        while changed and iter_times < max_iter_times:
            changed = False
            for node_idx in range(polygon_num):
                if not is_boundary_nodes[node_idx]:
                    continue
                current_label = polygon_labels[node_idx]
                min_neighbor_label = find_min_neighbor_label(node_idx)
                new_polygon_labels = copy.deepcopy(polygon_labels)
                new_polygon_labels[node_idx] = min_neighbor_label
                new_subgraphs = gen_subgraphs(new_polygon_labels)
                new_boundary_length = sum(find_boundary_nodes(polygon_network, new_polygon_labels))
                if (min_neighbor_label != current_label and
                        check_connectivity_after_move(polygon_network, node_idx, subgraphs[current_label], subgraphs[min_neighbor_label]) and
                        new_boundary_length < max_boundary_length):
                    changed = True
                    print("node {} change label from {} to {}".format(node_idx, current_label, min_neighbor_label))
                    polygon_labels = new_polygon_labels
                    subgraphs = new_subgraphs

            is_boundary_nodes = find_boundary_nodes(polygon_network, polygon_labels)
            iter_times += 1

        labels = polygon_labels
        subdomain_polygons = []
        for cluster_id in range(count_subdomain):
            mask = labels == cluster_id
            ploygons = [x for x, m in zip(self.polygons, mask) if m]
            merged_polygon = self.merge_polygons(ploygons)
            subdomain_polygons.append(merged_polygon)

        if if_plot:
            if self.plot_config["subdomain"][0]:
                self.plot_subdomains(subdomain_polygons, save_path=self.plot_config["subdomain"][1], type="adjust")

        return subdomain_polygons, labels


    def patch_cluster_weighted_balance(self, count_subdomain, polygon_weights, polygon_network, polygon_labels, if_plot=False):
        polygon_num = len(polygon_weights)
        polygon_weights = polygon_weights / np.sum(polygon_weights)

        # self.plot_polygon_network(self.polygons, self.bounding_polygon, self.patch_center_points, polygon_network, polygon_weights, save_path=self.plot_config["subdomain"][1])
        # 根据标签划分子图
        def gen_subgraphs(polygon_labels):
            subgraphs = {}
            for node, label in enumerate(polygon_labels):
                if label not in subgraphs:
                    subgraphs[label] = set()
                subgraphs[label].add(node)
            return subgraphs

        # 判断是否为边界节点
        def find_boundary_nodes(polygon_network, polygon_labels):
            is_boundary_nodes = [False for i in range(polygon_num)]  # True or False
            for node_idx in range(polygon_num):
                for neighbor in polygon_network.neighbors(node_idx):
                    if polygon_labels[neighbor] != polygon_labels[node_idx]:
                        is_boundary_nodes[node_idx] = True
                        break
            return is_boundary_nodes

        # 检查移动节点后是否会破坏连通性
        def check_connectivity_after_move(graph, node, current_subgraph_nodes, target_subgraph_nodes):
            current_subgraph_nodes = copy.deepcopy(current_subgraph_nodes)
            target_subgraph_nodes = copy.deepcopy(target_subgraph_nodes)
            new_current_subgraph = current_subgraph_nodes - {node}
            if not nx.is_connected(graph.subgraph(new_current_subgraph)):
                return False  # 当前子图不再连通
            new_target_subgraph = target_subgraph_nodes | {node}
            if not nx.is_connected(graph.subgraph(new_target_subgraph)):
                return False  # 目标子图不再连通
            return True  # 不会破坏连通性

        def compute_subgraph_cost(subgraph):
            cost = 0.0
            for node in subgraph:
                cost += polygon_weights[node][0]
            return cost

        def compute_balance_loss(subgraphs):
            cost = []
            for i in range(count_subdomain):
                cost.append(compute_subgraph_cost(subgraphs[i]))

            avg_cost = np.mean(cost)
            balance_loss = np.sum(np.abs(np.array(cost) - avg_cost))
            return balance_loss

        subgraphs = gen_subgraphs(polygon_labels)
        is_boundary_nodes = find_boundary_nodes(polygon_network, polygon_labels)
        init_boundary_length = sum(is_boundary_nodes)
        max_boundary_length = init_boundary_length + config_pinnadd.boundary_length_tolerance
        best_labels = copy.deepcopy(polygon_labels)
        best_values = np.inf
        lambda_reg = config_pinnadd.lambda1
        max_iter_times = config_pinnadd.max_iter_times_weighted_balance
        iter_times = 0
        changed = True
        while changed and iter_times < max_iter_times:
            changed = False
            for node_idx in range(polygon_num):
                if not is_boundary_nodes[node_idx]:
                    continue
                current_label = polygon_labels[node_idx]

                for new_label in range(count_subdomain):
                    if new_label == current_label:
                        continue
                    if not check_connectivity_after_move(polygon_network, node_idx, subgraphs[current_label], subgraphs[new_label]):
                        continue

                    origin_balance_loss = compute_balance_loss(subgraphs)
                    origin_boundary_length = sum(find_boundary_nodes(polygon_network, polygon_labels))
                    origin_value_func = lambda_reg * origin_boundary_length + origin_balance_loss

                    new_polygon_labels = copy.deepcopy(polygon_labels)
                    new_polygon_labels[node_idx] = new_label
                    new_subgraphs = gen_subgraphs(new_polygon_labels)
                    new_balance_loss = compute_balance_loss(new_subgraphs)
                    is_boundary_nodes_temp = find_boundary_nodes(polygon_network, new_polygon_labels)
                    boundary_length = sum(is_boundary_nodes_temp)
                    value_func = lambda_reg * boundary_length + new_balance_loss

                    if value_func < origin_value_func and boundary_length < max_boundary_length:
                        changed = True
                        print("banlance loss {} to {}".format(origin_balance_loss, new_balance_loss))
                        print("boundary_length {} to {}".format(init_boundary_length, boundary_length))
                        print("value function {} to {}".format(best_values, value_func))
                        polygon_labels = new_polygon_labels
                        subgraphs = new_subgraphs
                        best_labels = copy.deepcopy(polygon_labels)
                        best_values = value_func
                        break

            is_boundary_nodes = find_boundary_nodes(polygon_network, polygon_labels)
            iter_times += 1
            if iter_times >= max_iter_times:
                print("max iteration times reached")

        labels = best_labels
        subdomain_polygons = []
        for cluster_id in range(count_subdomain):
            mask = labels == cluster_id
            ploygons = [x for x, m in zip(self.polygons, mask) if m]
            merged_polygon = self.merge_polygons(ploygons)
            subdomain_polygons.append(merged_polygon)

        if if_plot:
            if self.plot_config["subdomain"][0]:
                self.plot_subdomains(subdomain_polygons, save_path=self.plot_config["subdomain"][1], type="balance")

        return subdomain_polygons, labels


    def merge_polygons(self, polygons_list):
        if len(polygons_list) > 0:
            polygon = polygons_list[0]
            for i in range(1, len(polygons_list)):
                polygon = polygon.union(polygons_list[i])

        else:
            raise ValueError("No polygons to merge")

        return polygon

    def plot_subdomains(self, subdomain_polygons, centroids=None, save_path=None, type="distance"):
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        fig, axs = plt.subplots(figsize=(10, 10))
        for i in range(len(subdomain_polygons)):
            subdomain = subdomain_polygons[i]
            if isinstance(subdomain, MultiPolygon):
                for poly in subdomain.geoms:
                    x, y = poly.exterior.xy
                    axs.fill(x, y, alpha=0.4, facecolor=colors[i])
                    for hole in poly.interiors:
                        x, y = hole.xy
                        axs.fill(x, y, facecolor='white')
            else:
                x, y = subdomain.exterior.xy
                axs.fill(x, y, alpha=0.4, facecolor=colors[i])
                for hole in subdomain.interiors:
                    x, y = hole.xy
                    axs.fill(x, y, facecolor='white')

        if centroids is not None:
            axs.scatter(centroids[:, 0], centroids[:, 1], color='red', label='Domain Centers')
        axs.set_title("Subdomains")
        axs.legend(loc='best')
        axs.axis('equal')
        plt.show()
        if save_path is not None:
            if type == "distance":
                fig.savefig(save_path + "/subdomains_distance.png")
            elif type == "adjust":
                fig.savefig(save_path + "/subdomains_adjust" + str(self.adjust_count) + ".png")
                self.adjust_count += 1
            elif type == "balance":
                fig.savefig(save_path + "/subdomains_balance.png")
            else:
                raise ValueError("Invalid type")

    # step 4: 过滤点，得到每个subdomain内的点，并且把所有配点都绘制出来
    def filter_points_interface_line(self, subdomain_polygons, col_points=None, boundary_points=None, interface_interval=0.01, if_plot=False):
        filtered_col_points = []
        if col_points is not None:
            for subdomain in subdomain_polygons:
                col_pts = np.array([point for point in col_points if subdomain.boundary.contains(Point(point))
                                         or subdomain.contains(Point(point))])
                col_pts = col_pts[self.geom.inside(col_pts)]
                filtered_col_points.append(col_pts)

        filtered_boundary_points = []
        if boundary_points is not None:
            for subdomain in subdomain_polygons:
                boundary_pts = np.array([point for point in boundary_points
                                         if subdomain.boundary.contains(Point(point))
                                         or subdomain.contains(Point(point))])
                boundary_pts = boundary_pts[self.geom.on_boundary(boundary_pts)]
                filtered_boundary_points.append(boundary_pts)

        connection = []
        interface_pts = []
        for i in range(len(subdomain_polygons)):
            for j in range(i+1, len(subdomain_polygons)):
                intersection = subdomain_polygons[i].intersection(subdomain_polygons[j])
                if intersection.is_empty or isinstance(intersection, Point):
                    continue
                elif isinstance(intersection, LineString):
                    connection.append([i, j])
                    interval = interface_interval  # 采样间隔
                    sampled_points = []
                    distance = 0.0

                    while distance < intersection.length:
                        sampled_points.append(intersection.interpolate(distance))
                        distance += interval

                    point_array = np.array([(point.x, point.y) for point in sampled_points])
                    interface_pts.append(point_array)

                elif isinstance(intersection, MultiLineString):
                    connection.append([i, j])
                    interval = interface_interval
                    sampled_points = []
                    for line in intersection.geoms:
                        distance = 0.0
                        while distance < line.length:
                            sampled_points.append(line.interpolate(distance))
                            distance += interval

                    point_array = np.array([(point.x, point.y) for point in sampled_points])
                    point_array = point_array[self.geom.inside(point_array)]
                    interface_pts.append(point_array)
                else:
                    raise ValueError("Invalid intersection type")

        if if_plot:
            if self.plot_config["filtered_points"][0]:
                self.plot_filtered_points(subdomain_polygons, filtered_col_points, filtered_boundary_points, interface_pts, save_path=self.plot_config["filtered_points"][1])

        return filtered_col_points, filtered_boundary_points, interface_pts, connection

    def filter_points_interface_band(self, subdomain_polygons, polygons, polygon_labels, col_points=None, boundary_points=None, if_plot=False):
        # find polygons that is at interface band
        # construct adjacent matrix for polygons
        polygons_num = len(polygons)
        G = nx.Graph()
        G.add_nodes_from(range(polygons_num))

        # adjacent_matrix = np.zeros((polygons_num, polygons_num))
        for i in range(polygons_num):
            for j in range(i+1, polygons_num):
                intersection = polygons[i].intersection(polygons[j])
                if isinstance(intersection, LineString) or isinstance(intersection, MultiLineString):
                    # adjacent_matrix[i, j] = 1
                    # adjacent_matrix[j, i] = 1
                    G.add_edge(i, j)

        node_labels = polygon_labels
        interface_polygons_idx_set = set()
        connection_dict = {}
        for node in G.nodes:
            neighbor_clusters = {node_labels[neighbor] for neighbor in G.neighbors(node)}
            if len(neighbor_clusters) > 1:
                key = list(neighbor_clusters)
                key.sort()
                key = tuple(key)
                if key not in connection_dict:
                    connection_dict[key] = [node]
                else:
                    connection_dict[key].append(node)
                interface_polygons_idx_set.add(node)

        # crop interface polygons
        num_subdomain = len(subdomain_polygons)
        cropped_subdomain_polygons = [[] for _ in range(num_subdomain)]
        for i in range(polygons_num):
            if i not in interface_polygons_idx_set:
                cropped_subdomain_polygons[polygon_labels[i]].append(polygons[i])

        subdomain_polygons = [None for _ in range(num_subdomain)]
        for i in range(num_subdomain):
            subdomain_polygons[i] = self.merge_polygons(cropped_subdomain_polygons[i])


        filtered_col_points = []
        if col_points is not None:
            for subdomain in subdomain_polygons:
                col_pts = np.array([point for point in col_points if subdomain.contains(Point(point))])
                col_pts = col_pts[self.geom.inside(col_pts)]
                filtered_col_points.append(col_pts)

        filtered_boundary_points = []
        if boundary_points is not None:
            for subdomain in subdomain_polygons:
                boundary_pts = np.array([point for point in boundary_points
                                         if subdomain.boundary.contains(Point(point))
                                         or subdomain.contains(Point(point))])
                boundary_pts = boundary_pts[self.geom.on_boundary(boundary_pts)]
                filtered_boundary_points.append(boundary_pts)

        connection = []
        interface_polygons = []
        interface_pts = []
        for key in connection_dict:
            connection.append(key)
            interface_polygons.append(self.merge_polygons([polygons[i] for i in connection_dict[key]]))

        for i in range(len(connection)):
            con = connection[i]
            boundary_pts = []
            if col_points is not None:
                col_pts = np.array([point for point in col_points if interface_polygons[i].contains(Point(point))])
                col_pts = col_pts[self.geom.inside(col_pts)]
                for c in con: # add interface points to subdomain
                    filtered_col_points[c] = np.vstack([filtered_col_points[c], col_pts])
            if boundary_points is not None:
                boundary_pts = np.array([point for point in boundary_points
                                         if interface_polygons[i].boundary.contains(Point(point))
                                         or interface_polygons[i].contains(Point(point))])
                if len(boundary_pts) > 0:
                    boundary_pts = boundary_pts[self.geom.on_boundary(boundary_pts)]
                    for c in con:
                        filtered_boundary_points[c] = np.vstack([filtered_boundary_points[c], boundary_pts])

            if len(col_pts) > 0 and len(boundary_pts) > 0:
                inter_pts = np.vstack([col_pts, boundary_pts])
            elif len(col_pts) > 0:
                inter_pts = col_pts
            else:
                inter_pts = None

            interface_pts.append(inter_pts)


        if if_plot:
            if self.plot_config["filtered_points"][0]:
                self.plot_filtered_points(subdomain_polygons, filtered_col_points, filtered_boundary_points,
                                          interface_pts, each_domain=False,
                                          save_path=self.plot_config["filtered_points"][1])
                self.plot_filtered_points(subdomain_polygons, filtered_col_points, filtered_boundary_points,
                                          interface_pts, each_domain=True, save_path=self.plot_config["filtered_points"][1])



        return filtered_col_points, filtered_boundary_points, interface_pts, connection

    def plot_filtered_points(self, subdomain_polygons, filtered_col_points, filtered_boundary_points, interface_pts, each_domain=False, save_path=None):
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        if not each_domain:
            figure, axs = plt.subplots(figsize=(10, 10))
            for i in range(len(subdomain_polygons)):
                polygon = subdomain_polygons[i]
                # 绘制多边形
                # axs.plot(*polygon.exterior.xy, label=f"Polygon {i + 1}", color=colors[i])

                if isinstance(polygon, MultiPolygon):
                    for poly in polygon.geoms:
                        x, y = poly.exterior.xy
                        axs.fill(x, y, alpha=0.4, facecolor=colors[i])
                        for hole in poly.interiors:
                            x, y = hole.xy
                            axs.fill(x, y, facecolor='white')
                else:
                    x, y = polygon.exterior.xy
                    axs.fill(x, y, alpha=0.4, facecolor=colors[i])
                    for hole in polygon.interiors:
                        x, y = hole.xy
                        axs.fill(x, y, facecolor='white')

                # 绘制多边形内部的点
                if len(filtered_col_points[i]) > 0:  # 如果多边形内有点
                    axs.scatter(filtered_col_points[i][:, 0], filtered_col_points[i][:, 1],
                                color=colors[i], label=f"Points in Polygon {i + 1}", alpha=0.6)

                if len(filtered_boundary_points) > 0 and len(filtered_boundary_points[i]) > 0:  # 如果多边形内有点
                    axs.scatter(filtered_boundary_points[i][:, 0], filtered_boundary_points[i][:, 1],
                                color=colors[i], label=f"Boundary Points in Polygon {i + 1}", alpha=0.6, marker='x')

            for i in range(len(interface_pts)):
                axs.scatter(interface_pts[i][:, 0], interface_pts[i][:, 1],
                            alpha=0.6, marker='s', color='black', label=f"Interface Points {i + 1}")

            axs.legend(loc='best')
            axs.axis('equal')
            axs.set_xlabel("X")
            axs.set_ylabel("Y")
            axs.set_title("Points in Multiple Subdomains")
            plt.show()

            if save_path is not None:
                figure.savefig(save_path + "/filtered_points.png")
        else:
            x_min = np.min([polygon.bounds[0] for polygon in subdomain_polygons])
            x_max = np.max([polygon.bounds[2] for polygon in subdomain_polygons])
            y_min = np.min([polygon.bounds[1] for polygon in subdomain_polygons])
            y_max = np.max([polygon.bounds[3] for polygon in subdomain_polygons])
            figure, axs = plt.subplots(1, len(subdomain_polygons)+1, figsize=(10 * (len(subdomain_polygons)+1), 10))
            for i in range(len(subdomain_polygons)):

                polygon = subdomain_polygons[i]
                # 绘制多边形
                if isinstance(polygon, MultiPolygon):
                    for poly in polygon.geoms:
                        x, y = poly.exterior.xy
                        axs[i].fill(x, y, alpha=0.4, facecolor=colors[i])
                        for hole in poly.interiors:
                            x, y = hole.xy
                            axs[i].fill(x, y, facecolor='white')
                else:
                    x, y = polygon.exterior.xy
                    axs[i].fill(x, y, alpha=0.4, facecolor=colors[i])
                    for hole in polygon.interiors:
                        x, y = hole.xy
                        axs[i].fill(x, y, facecolor='white')

                # 绘制多边形内部的点
                if len(filtered_col_points[i]) > 0:
                    axs[i].scatter(filtered_col_points[i][:, 0], filtered_col_points[i][:, 1],
                                color=colors[i], label=f"Points in Polygon {i + 1}", alpha=0.6)
                if len(filtered_boundary_points) > 0 and len(filtered_boundary_points[i]) > 0:
                    axs[i].scatter(filtered_boundary_points[i][:, 0], filtered_boundary_points[i][:, 1],
                                color=colors[i], label=f"Boundary Points in Polygon {i + 1}", alpha=0.6, marker='x')

                axs[i].set_xlim(x_min, x_max)  # 设置 x 轴范围
                axs[i].set_ylim(y_min, y_max)  # 设置 y 轴范围
                axs[i].legend(loc='best')  # 添加图例
                axs[i].set_aspect('equal')  # 保持比例相等
                axs[i].set_xlabel("X")  # 设置 x 轴标签
                axs[i].set_ylabel("Y")  # 设置 y 轴标签
                axs[i].set_title("Points in Subdomain " + str(i + 1))  # 设置子图标题


            # plt.figure(figsize=(10, 10))
            for i in range(len(interface_pts)):
                axs[-1].scatter(interface_pts[i][:, 0], interface_pts[i][:, 1],
                            alpha=0.6, marker='s', color=colors[i], label=f"Interface Points {i + 1}")

            axs[-1].set_xlim(x_min, x_max)  # 设置 x 轴范围
            axs[-1].set_ylim(y_min, y_max)  # 设置 y 轴范围
            axs[-1].legend(loc='best')  # 添加图例
            axs[-1].set_aspect('equal')  # 保持比例相等
            axs[-1].set_xlabel("X")  # 设置 x 轴标签
            axs[-1].set_ylabel("Y")  # 设置 y 轴标签
            axs[-1].set_title("Interface Points")  # 设置子图标题

            plt.show()

            if save_path is not None:
                figure.savefig(save_path + "/filtered_points_each_domain.png")


    def save2file(self, file_path=None):
        save_dict = {
            "col_points": self.col_points,
            "boundary_points": self.boundary_points,
            "filtered_col_points": self.filtered_col_points,
            "filtered_boundary_points": self.filtered_boundary_points,
            "interface_pts": self.interface_pts,
            "connection": self.connection,
            "subdomain_polygons": self.subdomain_polygons
        }
        if file_path is not None:
            np.save(file_path, save_dict)
            print(f"Data saved to {file_path}")

        return save_dict

# 需要哪些功能呢？
# 1. 利用端点构建一个多面体形状
# 2. 判断点是否在多面体内部、边界上；在内部和边界(面)上采点
# 3. 多面体的交界判断，交并差处理
# 4. 可视化



