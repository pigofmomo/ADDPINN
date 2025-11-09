import numpy as np
import deepxde as dde
import torch
import pinn_add
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from shapely.geometry import Polygon, Point

import config_pinnadd
import json
import os


def pde(x, y):
    u = y  # NN 输出
    u_xx = dde.grad.hessian(u, x, i=0, j=0)  # d2u/dx2
    u_yy = dde.grad.hessian(u, x, i=1, j=1)  # d2u/dy2
    return u_xx + u_yy


# 解析解 u*(x,y) = x^3 - 3 x y^2
def exact_solution(x):
    # x: (N,2)
    X = x[:, 0:1]
    Y = x[:, 1:2]
    return X**3 - 3 * X * (Y**2)

def make_star_polygon(R=0.9, r=0.45, n_arms=5, center=(0.0, 0.0), theta0=0.0):
    """
    生成 2n 个顶点的星形多边形（交替使用外半径 R 和内半径 r）。
    返回顶点列表，按逆时针排列，便于 deepxde.geometry.Polygon 使用。
    """
    cx, cy = center
    verts = []
    for k in range(2 * n_arms):
        ang = theta0 + k * np.pi / n_arms
        rad = R if (k % 2 == 0) else r
        x = cx + rad * np.cos(ang)
        y = cy + rad * np.sin(ang)
        verts.append([x, y])
    return np.array(verts, dtype=np.float64)

geom_type = "L"
if geom_type == "star":
    star = make_star_polygon(R=0.9, r=0.55, n_arms=5, theta0=np.pi / 10)
    spatial_domain = dde.geometry.Polygon(star)
    plt.figure(figsize=(5, 5))
    plt.plot(star[:, 0], star[:, 1], 'b-', linewidth=2, label='Star boundary')
    plt.fill(star[:, 0], star[:, 1], color='skyblue', alpha=0.3)
    plt.scatter(0, 0, color='red', s=40, label='Center')
    plt.axis('equal')
    plt.legend()
    plt.title("Star-shaped 2D domain")
    # 保存为文件（不显示）
    plt.savefig("star_shape_domain.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 已保存图片：star_shape_domain.png")

    spatial_domain_shapely = Polygon(star)

elif geom_type == "L":
    # bbox = [-1, 1, -1, 1]
    # bbox2 = [0, 1, 0, 1]
    # square1 = dde.geometry.Rectangle([bbox[0], bbox[2]], [bbox[1], bbox[3]])
    # square2 = dde.geometry.Rectangle([bbox2[0], bbox2[2]], [bbox2[1], bbox2[3]])
    # spatial_domain = dde.geometry.csg.CSGDifference(square1, square2)

    L_points = [(-1,-1), (1,-1), (1,0), (0,0), (0,1), (-1,1)]
    spatial_domain = dde.geometry.Polygon(L_points)
    spatial_domain_shapely = Polygon(L_points)
    
    # 可视化
    fig, ax = plt.subplots(figsize=(5,5))
    x, y = spatial_domain_shapely.exterior.xy  # ✅ Shapely Polygon有exterior属性
    ax.plot(x, y, 'b-', linewidth=2, label='L-shape boundary')
    ax.fill(x, y, color='lightgreen', alpha=0.3)
    ax.scatter(*zip(*L_points), color='red', s=30, zorder=5)

    ax.set_aspect('equal', 'box')
    ax.set_title("L-shaped 2D domain")
    ax.legend()
    plt.savefig("L_shape_domain.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 已保存图片：L_shape_domain.png")

pointset = pinn_add.PointSets2D(spatial_domain, divide_config=None, spatial_domain_shapely=spatial_domain_shapely)

# Dirichlet 边界条件：u = exact_solution 在边界上
def on_boundary(_, on_bnd):
    return on_bnd
bc = dde.icbc.DirichletBC(spatial_domain, exact_solution, on_boundary)

layers1 = [2, 30, 30, 30, 30, 1]
layers2 = [2, 30, 30, 30, 30, 1]
layers3 = [2, 30, 30, 30, 30, 1]
nets = [layers1, layers2, layers3]

print_every = 100
train_iterations = 5000

config_pinnadd.pretrain_iterations = 500
config_pinnadd.adjust_subdomain_start_iteration = 2000
config_pinnadd.adjust_subdomain_end_iteration = 3500
config_pinnadd.adjust_subdomain_period = 500

config_pinnadd.boundary_length_tolerance = 10
config_pinnadd.max_iter_times_weighted_balance = 5

config_pinnadd.max_iter_times_weighted_min = 1
config_pinnadd.boundary_length_tolerance2 = 5

num_domain = 4000
num_boundary = 800
n_clusters = 500
num_test = 10000
loss_weight_for_each_domain = [1, 1, 1, 1]

def train_pinn(if_train=True):
    net = [layers1]
    data = {"pde": pde,
            "pointset": pointset,
            "num_domain": num_domain,
            "num_boundary": num_boundary,
            "n_clusters": n_clusters,
            "icbc": [bc],
            "loss_weights": [loss_weight_for_each_domain for _ in range(1)],
            "ref_sol": exact_solution,
            "interface_type": "line"}

    save_dir = './' + geom_type + '/pinn_result/' + str(config_pinnadd.random_seed) + '/'
    pinnadd_model = pinn_add.PINNADD(net, data, save_dir=save_dir)

    if if_train:
        result_dict = pinnadd_model.train(train_iterations, print_every)
        pinnadd_model.save(save_dir, result_dict)
    else:
        pinnadd_model.restore(save_dir)

    test(pinnadd_model, save_dir)

def train_xpinn(if_train=True):
    data = {"pde": pde,
            "pointset": pointset,
            "num_domain": num_domain,
            "num_boundary": num_boundary,
            "n_clusters": n_clusters,
            "icbc": [bc],
            "loss_weights": [loss_weight_for_each_domain for _ in range(len(nets))],
            "ref_sol": exact_solution,
            "interface_type": "line"}

    save_dir = './' + geom_type +  '/xpinn_result/' + str(config_pinnadd.random_seed) + '/'
    pinnadd_model = pinn_add.PINNADD(nets, data, save_dir=save_dir)

    if if_train:
        result_dict = pinnadd_model.train(train_iterations, print_every)
        pinnadd_model.save(save_dir, result_dict)
    else:
        pinnadd_model.restore(save_dir)

    test(pinnadd_model, save_dir)

def train_pinnadd(if_train=True):
    data = {"pde": pde,
            "pointset": pointset,
            "num_domain": num_domain,
            "num_boundary": num_boundary,
            "n_clusters": n_clusters,
            "icbc": [bc],
            "loss_weights": [loss_weight_for_each_domain for _ in range(len(nets))],
            "ref_sol": exact_solution,
            "interface_type": "band",
            "ignore_inter_pde_loss": True}

    save_dir = './' + geom_type +  '/pinnadd_result/' + str(config_pinnadd.random_seed) + '/'
    pinnadd_model = pinn_add.PINNADD(nets, data, save_dir=save_dir)
    if if_train:
        result_dict = pinnadd_model.train(train_iterations, print_every)
        pinnadd_model.save(save_dir, result_dict)
    else:
        pinnadd_model.restore(save_dir)

    test(pinnadd_model, save_dir)

def train_pinnadd2(if_train=True):
    data = {"pde": pde,
            "pointset": pointset,
            "num_domain": num_domain,
            "num_boundary": num_boundary,
            "n_clusters": n_clusters,
            "icbc": [bc],
            "loss_weights": [loss_weight_for_each_domain for _ in range(len(nets))],
            "ref_sol": exact_solution,
            "interface_type": "band",
            "ignore_inter_pde_loss": True,
            "initial_subdomain_balance": True}

    save_dir = './' + geom_type +  '/pinnadd_result2/' + str(config_pinnadd.random_seed) + '/'
    pinnadd_model = pinn_add.PINNADD(nets, data, save_dir=save_dir)
    if if_train:
        result_dict = pinnadd_model.train(train_iterations, print_every)
        pinnadd_model.save(save_dir, result_dict)
    else:
        pinnadd_model.restore(save_dir)

    test(pinnadd_model, save_dir)

def train_pinnadd3(if_train=True): # 自动调节区域分割的。
    data = {"pde": pde,
            "pointset": pointset,
            "num_domain": num_domain,
            "num_boundary": num_boundary,
            "n_clusters": n_clusters,
            "icbc": [bc],
            "loss_weights": [loss_weight_for_each_domain for _ in range(len(nets))],
            "ref_sol": exact_solution,
            "interface_type": "band",
            "ignore_inter_pde_loss": True,
            "initial_subdomain_balance": True,
            "adjust_subdomain": True
            }

    save_dir = './' + geom_type +  '/pinnadd_result3/' + str(config_pinnadd.random_seed) + '/'
    pinnadd_model = pinn_add.PINNADD(nets, data, save_dir=save_dir)

    if if_train:
        result_dict = pinnadd_model.train(train_iterations, print_every)
        pinnadd_model.save(save_dir, result_dict)
    else:
        pinnadd_model.restore(save_dir)

    test(pinnadd_model, save_dir)

def test(pinnadd_model, save_dir):
    test_pts = []
    test_vals = []
    pred_vals = []
    test_results = pinnadd_model.test()
    for t in test_results:
        test_pts.append(t[0])
        test_vals.append(t[1])
        pred_vals.append(t[2])
    
    test_pts = np.vstack(test_pts)
    test_vals = np.vstack(test_vals)
    pred_vals = np.vstack(pred_vals)

    metric_list = ["MSE", "l2 relative error"]
    errors = []
    for metric in metric_list:
        metric_fn = dde.metrics.get(metric)
        err = metric_fn(test_vals, pred_vals).astype(float)
        errors.append(err)
        print(f"{metric}: {errors[-1]}")
    
    np.savetxt(save_dir + "metric.txt", np.array([errors]), fmt="%.6e",
               delimiter=",", header=" ".join(metric_list),comments="")

    X_test = test_pts
    u_pred = pinnadd_model.predict(X_test)
    u_true = exact_solution(X_test)
    error = np.abs(u_pred - u_true)
    # l2_rel = dde.metrics.l2_relative_error(u_true, u_pred)
    # print("L2 relative error on test:", l2_rel)
    XY_in = X_test
    boundary_xy = np.array(spatial_domain.vertices)
    boundary_xy = np.vstack([boundary_xy, boundary_xy[0]])  # 闭合边界
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    p0 = axs[0].scatter(XY_in[:, 0], XY_in[:, 1], c=u_pred, s=6, cmap='viridis', marker='s', rasterized=True)
    axs[0].plot(*boundary_xy.T, 'k-', lw=1)
    axs[0].set_aspect('equal'); axs[0].set_title("PINN predicted")
    plt.colorbar(p0, ax=axs[0])

    p1 = axs[1].scatter(XY_in[:, 0], XY_in[:, 1], c=u_true, s=6, cmap='viridis', marker='s', rasterized=True)
    axs[1].plot(*boundary_xy.T, 'k-', lw=1)
    axs[1].set_aspect('equal'); axs[1].set_title("Analytical")
    plt.colorbar(p1, ax=axs[1])

    p2 = axs[2].scatter(XY_in[:, 0], XY_in[:, 1], c=error,  s=6, cmap='inferno', marker='s', rasterized=True)
    axs[2].plot(*boundary_xy.T, 'k-', lw=1)
    axs[2].set_aspect('equal'); axs[2].set_title("|Error|")
    plt.colorbar(p2, ax=axs[2])

    plt.tight_layout()
    plt.savefig(save_dir + "result.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ 已保存：pinn_star_solution_scatter.png")
    plt.show()

def summary(geom_type="star"):

    dir_list = ["./" + geom_type + "/pinn_result/",
                "./" + geom_type + "/xpinn_result/",
                "./" + geom_type + "/pinnadd_result/",
                # "./" + geom_type + "/pinnadd_result2/",
                # "./" + geom_type + "/pinnadd_result3/"
                ]

    result_str_list = ["mse(std), l2 relative error(std), train time(std)"]
    subdomain_num = 3
    line_styles = ['-', '--', '-.', ':', '-']
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    for dir in dir_list:
        subfolders = [os.path.join(dir, d) for d in os.listdir(dir) if
                      os.path.isdir(os.path.join(dir, d))]
        metrics_each_run = []
        train_time_each_run = []
        for d in subfolders:
            metrics = np.loadtxt(d + "/metric.txt", skiprows=1, delimiter=",")
            metrics_each_run.append(metrics)
            with open(d + "/result.json", "r") as f:
                result_dict = json.load(f)
            train_time = result_dict["train_time"]
            train_time_each_run.append([train_time])

            subdomain_loss_curves = []
            for i in range(subdomain_num):
                if os.path.exists(d + f"/model-{i}-history.txt"):
                    subdomain_loss = np.loadtxt(d + f"/model-{i}-history.txt", delimiter=",")
                    subdomain_loss_curves.append(subdomain_loss)

            # plot loss
            loss_legend = ["Pde loss", "BC loss", "Interface average loss", "Interface pde loss"]
            fig, axs = plt.subplots(1, 4, figsize=(20, 5))
            for i in range(len(subdomain_loss_curves)):
                subdomain_loss = subdomain_loss_curves[i]
                iterations = subdomain_loss[:, 0]  # 迭代次数
                loss = subdomain_loss[:, 1:5]  # Loss
                for j in range(loss.shape[1]):
                    axs[j].plot(iterations, loss[:, j], label=f'Model {i} - ' + f"{loss_legend[j]}", color=colors[i],
                             linestyle=line_styles[j])

            for j in range(loss.shape[1]):
                axs[j].set_xlabel("Iteration", fontsize=20)
                axs[j].set_ylabel("Loss", fontsize=20)
                axs[j].tick_params(axis='y', labelsize=15)
                axs[j].set_title(loss_legend[j], fontsize=20)
                axs[j].legend(loc='best')
                axs[j].grid(True)

            fig.suptitle("Loss for Multiple Subdomains", fontsize=15)
            plt.tight_layout()
            plt.show()
            fig.savefig(d + "loss.png")

            # plot metrics
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            for i in range(len(subdomain_loss_curves)):
                subdomain_loss = subdomain_loss_curves[i]
                iterations = subdomain_loss[:, 0]
                mse = subdomain_loss[:, 5]
                l2_relative_error = subdomain_loss[:, 6]
                axs[0].plot(iterations, mse, label=f'Model {i} - MSE', color=colors[i], linestyle=line_styles[i])
                axs[1].plot(iterations, l2_relative_error, label=f'Model {i} - L2 relative error', color=colors[i],
                            linestyle=line_styles[i], marker='o')

            axs[0].set_xlabel("Iteration", fontsize=20)
            axs[0].set_ylabel("MSE", fontsize=20)
            axs[0].tick_params(axis='y', labelsize=15)
            axs[0].set_title("MSE", fontsize=20)
            axs[0].legend(loc='best')
            axs[0].grid(True)

            axs[1].set_xlabel("Iteration", fontsize=20)
            axs[1].set_ylabel("L2 Relative Error", fontsize=20)
            axs[1].tick_params(axis='y', labelsize=15)
            axs[1].set_title("L2 Relative Error", fontsize=20)
            axs[1].legend(loc='best')
            axs[1].grid(True)
            axs[1].set_ylim([0, 1])

            fig.suptitle("Metrics for Multiple Subdomains", fontsize=15)
            plt.tight_layout()
            plt.show()
            fig.savefig(d + "metrics.png")

        metrics_each_run = np.array(metrics_each_run)
        metrics_mean = np.mean(metrics_each_run, axis=0)
        metrics_std = np.std(metrics_each_run, axis=0)
        train_time_each_run = np.array(train_time_each_run)
        train_time_mean = np.mean(train_time_each_run, axis=0)
        train_time_std = np.std(train_time_each_run, axis=0)
        result_str = f"{metrics_mean[0]/1e-5:.3f} \\pm {metrics_std[0]/1e-5:.3f}, {metrics_mean[1]/1e-2:.3f} \\pm {metrics_std[1]/1e-2:.3f}, " + \
                     f"{train_time_mean[0]:.1f} \\pm {train_time_std[0]:.1f}"
        result_str_list.append(result_str)

    result_str_list = "\n".join(result_str_list)
    print(result_str_list)
    with open("metric_results.txt", "w") as f:
        f.write(result_str_list)

if __name__ == "__main__":

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    seeds = [0, 1, 2]
    # seeds = [0]
    for i in range(len(seeds)):
        if seeds[i] is None:
            continue
        print("random seed: ", seeds[i])
        config_pinnadd.random_seed = seeds[i]
        train_pinn(if_train=True)
        train_xpinn(if_train=True)
        train_pinnadd(if_train=True)
        # train_pinnadd2(if_train=True)
        # train_pinnadd3(if_train=True)
