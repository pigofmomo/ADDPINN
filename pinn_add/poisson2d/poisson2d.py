import deepxde as dde
import numpy as np
import torch
import pinn_add
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from shapely.geometry import Polygon, Point

import config_pinnadd
import json
import os

def pde(x, u):
    k = 8
    mu=(1, 4)
    mu1 = mu[0]
    mu2 = mu[1]
    A = 10

    u_xx = dde.grad.hessian(u, x, i=0, j=0)
    u_yy = dde.grad.hessian(u, x, i=1, j=1)

    def f(xy):
        x, y = xy[:, 0:1], xy[:, 1:2]
        return A * (mu1 ** 2 + x ** 2 + mu2 ** 2 + y ** 2) \
            * torch.sin(mu1 * torch.pi * x) * torch.sin(mu2 * torch.pi * y)

    return -(u_xx + u_yy) + k ** 2 * u - f(x)

bbox = [-1, 1, -1, 1]
circ = [(0.5, 0.5, 0.2), (0.4, -0.4, 0.4), (-0.2, -0.7, 0.1), (-0.6, 0.5, 0.3)]

spatial_domain_shapely = Polygon([(bbox[0], bbox[2]), (bbox[1], bbox[2]), (bbox[1], bbox[3]), (bbox[0], bbox[3])])
for c in circ:
    spatial_domain_shapely = spatial_domain_shapely.difference(Point(c[0], c[1]).buffer(c[2]))

spatial_domain = dde.geometry.Rectangle(xmin=[bbox[0], bbox[2]], xmax=[bbox[1], bbox[3]])
for i in range(len(circ)):
    disk = dde.geometry.Disk(circ[i][0:2], circ[i][2])
    spatial_domain = dde.geometry.csg.CSGDifference(spatial_domain, disk)

pointset = pinn_add.PointSets2D(spatial_domain, divide_config=None, spatial_domain_shapely=spatial_domain_shapely)


def boundary_rec(x, on_boundary):
    return on_boundary and (
            np.isclose(x[0], bbox[0]) or np.isclose(x[0], bbox[1]) or np.isclose(x[1], bbox[2]) or np.isclose(
        x[1], bbox[3]))


def boundary_circle(x, on_boundary):
    return on_boundary and not boundary_rec(x, on_boundary)


bc_rec = dde.DirichletBC(spatial_domain, lambda x: 0.2, boundary_rec, component=0)
bc_circ = dde.DirichletBC(spatial_domain, lambda x: 1.0, boundary_circle, component=0)

ref_data = np.loadtxt("./poisson_boltzmann2d.dat", comments="%", encoding='utf-8')

layers1 = [2, 50, 50, 50, 50, 1]
layers2 = [2, 50, 50, 50, 50, 1]
layers3 = [2, 50, 50, 50, 50, 1]
nets = [layers1, layers2, layers3]

print_every = 100
train_iterations = 20000
config_pinnadd.adjust_subdomain_start_iteration = 5000
config_pinnadd.adjust_subdomain_end_iteration = 10000
config_pinnadd.adjust_subdomain_period = 2000

config_pinnadd.boundary_length_tolerance = 10
config_pinnadd.max_iter_times_weighted_balance = 5

config_pinnadd.max_iter_times_weighted_min = 1
config_pinnadd.boundary_length_tolerance2 = 5

num_domain = 10000
num_boundary = 10000
n_clusters = 500
loss_weight_for_each_domain = [1, 80000, 80000, 10]

def train_pinn(if_train=True):
    net = [layers1]
    data = {"pde": pde,
            "pointset": pointset,
            # "num_domain": int(num_domain/3),
            # "num_boundary": int(num_boundary/3),
            "num_domain": num_domain,
            "num_boundary": num_boundary,
            "n_clusters": n_clusters,
            "icbc": [bc_rec, bc_circ],
            "loss_weights": [loss_weight_for_each_domain for _ in range(1)],
            "ref_data": ref_data,
            "interface_type": "line"}

    save_dir = './pinn_result/' + str(config_pinnadd.random_seed) + '/'
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
            "icbc": [bc_rec, bc_circ],
            "loss_weights": [loss_weight_for_each_domain for _ in range(len(nets))],
            "ref_data": ref_data,
            "interface_type": "line"}

    save_dir = './xpinn_result/' + str(config_pinnadd.random_seed) + '/'
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
            "icbc": [bc_rec, bc_circ],
            "loss_weights": [loss_weight_for_each_domain for _ in range(len(nets))],
            "ref_data": ref_data,
            "interface_type": "band",
            "ignore_inter_pde_loss": True}

    save_dir = './pinnadd_result/' + str(config_pinnadd.random_seed) + '/'
    pinnadd_model = pinn_add.PINNADD(nets, data, save_dir=save_dir)
    if if_train:
        result_dict = pinnadd_model.train(train_iterations, print_every)
        pinnadd_model.save(save_dir, result_dict)
    else:
        pinnadd_model.restore(save_dir)

    test(pinnadd_model, save_dir)

def train_pinnadd2(if_train=True): # 自动调节区域分割的。
    data = {"pde": pde,
            "pointset": pointset,
            "num_domain": num_domain,
            "num_boundary": num_boundary,
            "n_clusters": n_clusters,
            "icbc": [bc_rec, bc_circ],
            "loss_weights": [loss_weight_for_each_domain for _ in range(len(nets))],
            "ref_data": ref_data,
            "interface_type": "band",
            "ignore_inter_pde_loss": True,
            "initial_subdomain_balance": True,
            }

    save_dir = './pinnadd_result2/' + str(config_pinnadd.random_seed) + '/'
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
            "icbc": [bc_rec, bc_circ],
            "loss_weights": [loss_weight_for_each_domain for _ in range(len(nets))],
            "ref_data": ref_data,
            "interface_type": "band",
            "ignore_inter_pde_loss": True,
            # "initial_subdomain_balance": True,
            "adjust_subdomain": True
            }

    save_dir = './pinnadd_result3/' + str(config_pinnadd.random_seed) + '/'
    pinnadd_model = pinn_add.PINNADD(nets, data, save_dir=save_dir)

    if if_train:
        result_dict = pinnadd_model.train(train_iterations, print_every)
        pinnadd_model.save(save_dir, result_dict)
    else:
        pinnadd_model.restore(save_dir)

    test(pinnadd_model, save_dir)

def train_pinnadd4(if_train=True): # 自动调节区域分割的。
    data = {"pde": pde,
            "pointset": pointset,
            "num_domain": num_domain,
            "num_boundary": num_boundary,
            "n_clusters": n_clusters,
            "icbc": [bc_rec, bc_circ],
            "loss_weights": [loss_weight_for_each_domain for _ in range(len(nets))],
            "ref_data": ref_data,
            "interface_type": "band",
            "ignore_inter_pde_loss": True,
            "analyze_gradient": True}

    config_pinnadd.random_seed = 0
    config_pinnadd.min_iteration = 10
    config_pinnadd.max_iteration = 50
    config_pinnadd.window_length = 5
    config_pinnadd.probe_grad_period = 2
    config_pinnadd.ratio_threshold = 0.7
    save_dir = './pinnadd_result4/' + str(config_pinnadd.ratio_threshold) + '/'
    pinnadd_model = pinn_add.PINNADD(nets, data, save_dir=save_dir)

    if if_train:
        result_dict = pinnadd_model.train(train_iterations, print_every)
        pinnadd_model.save(save_dir, result_dict)
    else:
        pinnadd_model.restore(save_dir)

    test(pinnadd_model, save_dir)

def train_pinnadd5(if_train=True): # 自动调节区域分割的。
    data = {"pde": pde,
            "pointset": pointset,
            "num_domain": num_domain,
            "num_boundary": num_boundary,
            "n_clusters": n_clusters,
            "icbc": [bc_rec, bc_circ],
            "loss_weights": [loss_weight_for_each_domain for _ in range(len(nets))],
            "ref_data": ref_data,
            "interface_type": "band",
            "ignore_inter_pde_loss": True,
            "analyze_gradient": True}

    # 1 2.966198e-05,1.039432e-02
    # 2 5.688415e-05,1.439433e-02
    # 5 6.594669e-05,1.549860e-02
    # 10 4.838756e-05,1.327586e-02
    # 20 3.449658e-05,1.120944e-02
    # 50 4.575967e-05,1.291033e-02
    # 100 6.588780e-04,4.898898e-02
    config_pinnadd.random_seed = 0
    config_pinnadd.frequency = 1
    config_pinnadd.static_frequency = True
    config_pinnadd.min_iteration = config_pinnadd.frequency
    config_pinnadd.max_iteration = config_pinnadd.frequency
    config_pinnadd.probe_grad_period = config_pinnadd.frequency
    save_dir = './pinnadd_result5/' + str(config_pinnadd.frequency) + '/'
    pinnadd_model = pinn_add.PINNADD(nets, data, save_dir=save_dir)

    if if_train:
        result_dict = pinnadd_model.train(train_iterations, print_every)
        pinnadd_model.save(save_dir, result_dict)
    else:
        pinnadd_model.restore(save_dir)

    test(pinnadd_model, save_dir)

def to_analyze_gradient(if_train=True):
    data = {"pde": pde,
            "pointset": pointset,
            "num_domain": num_domain,
            "num_boundary": num_boundary,
            "n_clusters": n_clusters,
            "icbc": [bc_rec, bc_circ],
            "loss_weights": [loss_weight_for_each_domain for _ in range(len(nets))],
            "ref_data": ref_data,
            "interface_type": "band",
            "ignore_inter_pde_loss": True,
            "analyze_gradient": True}

    config_pinnadd.max_iteration = 100
    config_pinnadd.probe_grad_period = 1
    config_pinnadd.ratio_threshold = 0.0


    save_dir = './to_analyze_gradient/'
    pinnadd_model = pinn_add.PINNADD(nets, data, save_dir=save_dir)
    if if_train:
        result_dict = pinnadd_model.train(10000, print_every)
        pinnadd_model.save(save_dir, result_dict)
        for i in range(3):
            grads_history = np.array(pinnadd_model.models[i].grads_history)
            np.save(save_dir + f"grads_history_{i}.npy", grads_history)
    else:
        # pinnadd_model.restore(save_dir)
        begin = 190
        end = 1990
        grads_history = np.load(save_dir + "grads_history_0.npy")
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        idxs = np.arange(grads_history.shape[0])
        axs[0].plot(idxs[begin:end], grads_history[begin:end, 0], label="Interior Gradients")
        axs[0].plot(idxs[begin:end], grads_history[begin:end, 1], label="Interface Gradients")
        axs[1].plot(idxs[begin:end], grads_history[begin:end, 2], label="Ratio")
        axs[0].set_xlabel("Iteration")
        axs[0].set_ylabel("Gradient")
        axs[0].legend()
        axs[0].grid(True)
        axs[1].set_xlabel("Iteration")
        axs[1].set_ylabel("Ratio")
        axs[1].legend()
        axs[1].grid(True)
        fig.suptitle("Gradients Analysis")
        plt.show()
        fig.savefig(save_dir + "grads_history.png")


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
    x = np.linspace(-1.0, 1.0, 100)
    y = np.linspace(-1.0, 1.0, 100)
    X, Y = np.meshgrid(x, y)
    X_star = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))


    U_true = griddata(ref_data[:, 0:2], ref_data[:, 2:3], X_star, method='cubic')
    U_true[~spatial_domain.inside(X_star)] = 0.0
    U_true_grid = U_true.reshape(X.shape)
    U_pred = pinnadd_model.predict(X_star)
    U_pred[~spatial_domain.inside(X_star)] = 0.0
    U_pred_grid = U_pred.reshape(X.shape)

    fig, axs = plt.subplots(1, 3, figsize=(18, 4))
    extent = (-1.0, 1.0, -1.0, 1.0)

    # prediction
    h = axs[0].imshow(U_pred_grid, interpolation='nearest', cmap='rainbow', extent=extent,
                      origin='lower', aspect='auto')

    for c in circ:
        circle = plt.Circle((c[0], c[1]), c[2], color='black', fill=True, fc='white')
        axs[0].add_artist(circle)

    axs[0].set_xlabel(r'$x$')
    axs[0].set_ylabel(r'$y$')
    axs[0].legend(frameon=False, loc='best')
    axs[0].set_title(r'predicted $u(t,x)$', fontsize=10)
    fig.colorbar(h, ax=axs[0])

    # ground truth
    h = axs[1].imshow(U_true_grid, interpolation='nearest', cmap='rainbow', extent=extent,
                      origin='lower', aspect='auto')

    for c in circ:
        circle = plt.Circle((c[0], c[1]), c[2], color='black', fill=True, fc='white')
        axs[1].add_artist(circle)
    axs[1].set_xlabel(r'$x$')
    axs[1].set_ylabel(r'$y$')
    axs[1].legend(frameon=False, loc='best')
    axs[1].set_title(r'true $u(t,x)$', fontsize=10)
    fig.colorbar(h, ax=axs[1])

    # absolute error
    U_abs_res = np.abs(U_true_grid - U_pred_grid)
    h = axs[2].imshow(U_abs_res, interpolation='nearest', cmap='rainbow', extent=extent,
                      origin='lower', aspect='auto')

    for c in circ:
        circle = plt.Circle((c[0], c[1]), c[2], color='black', fill=True, fc='white')
        axs[2].add_artist(circle)
    axs[2].set_xlabel(r'$x$')
    axs[2].set_ylabel(r'$y$')
    axs[2].legend(frameon=False, loc='best')
    axs[2].set_title(r'res $u(t,x)$', fontsize=10)
    fig.colorbar(h, ax=axs[2])

    plt.show()
    fig.savefig(save_dir + "result.png")

def summary():
    dir_list = ["./pinn_result/",
                "./xpinn_result/",
                "./pinnadd_result/",
                "./pinnadd_result2/",
                "./pinnadd_result3/",
                # "./pinnadd_result4/"
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

def summary_frequency():
    # fixed 1
    # 2.966198e-05,1.039432e-02
    # 736s / 2852s
    # adaptive 0.7
    # 3.862823e-05,1.186174e-02
    # 42s / 1952s

    # 1 2.966198e-05,1.039432e-02
    # 2 5.688415e-05,1.439433e-02
    # 5 6.594669e-05,1.549860e-02
    # 10 4.838756e-05,1.327586e-02
    # 20 3.449658e-05,1.120944e-02
    # 50 4.575967e-05,1.291033e-02
    # 100 6.588780e-04,4.898898e-02
    labels1 = ['1', '2', '5', '10', '20', '50', '100']  # 横轴分类标签
    mse1 = [2.966198e-05, 5.688415e-05, 6.594669e-05, 4.838756e-05, 3.449658e-05, 4.575967e-05, 6.588780e-04]  # 左轴数据：MSE
    l2_err1 = [1.039432e-02, 1.439433e-02, 1.549860e-02, 1.327586e-02, 1.120944e-02, 1.291033e-02, 4.898898e-02]  # 右轴数据：L2相对误差

    # 0.3 5.018648e-05,1.352039e-02
    # 0.5 4.305434e-05,1.252288e-02
    # 0.7 3.862823e-05,1.186174e-02
    labels2 = ['0.3', '0.5', '0.7']  # 横轴分类标签
    mse2 = [5.018648e-05, 4.305434e-05, 3.862823e-05]  # 左轴数据：MSE
    l2_err2 = [1.352039e-02, 1.252288e-02, 1.186174e-02]  # 右轴数据：L2相对误差

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    line1 = ax1.plot(labels1, mse1, 'o-', color='k', label='MSE')
    ax1.set_ylabel('MSE')
    line2 = ax2.plot(labels1, l2_err1, 's--', color='k', label='L2 relative error')
    ax2.set_ylabel('L2 relative error')
    ax1.set_xlabel('Step Interval')
    # ax1.set_yscale('log')
    ax1.set_ylim(0, 0.001)
    ax2.set_ylim(0, 0.06)
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best')
    plt.title("Fixed Synchronization Frequency")
    plt.savefig("./pinnadd_result5/fixed_frequency.png")
    plt.show()

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    line1 = ax1.plot(labels2, mse2, 'o-', color='k', label='MSE')
    ax1.set_ylabel('MSE')
    line2 = ax2.plot(labels2, l2_err2, 's--', color='k', label='L2 relative error')
    ax2.set_ylabel('L2 relative error')
    ax1.set_xlabel('Ratio')
    # ax1.set_yscale('log')
    ax1.set_ylim(0, 0.001)
    ax2.set_ylim(0, 0.06)
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best')
    plt.title("Adaptive Synchronization Frequency")
    plt.savefig("./pinnadd_result4/adaptive_frequency.png")
    plt.show()

    dir = "./pinnadd_result4/0.7/"
    step_intervals = []
    mid_steps = []
    for i in range(3):
        if os.path.exists(dir + f"/model-{i}-history.txt"):
            subdomain_loss = np.loadtxt(dir + f"/model-{i}-history.txt", delimiter=",")
            subdomain_loss = subdomain_loss[:-1, :]
            steps = subdomain_loss[:, 0]
            step_interval = np.diff(steps)
            mid_step = [(steps[i] + steps[i + 1]) / 2 for i in range(len(step_interval))]
            step_intervals.append(step_interval)
            mid_steps.append(mid_step)
            pass

    fig, axs = plt.subplots(3, 1)
    axs[0].plot(mid_steps[0], step_intervals[0], 'o-', color='r', label='model 0')
    axs[1].plot(mid_steps[1], step_intervals[1], 'o-', color='g', label='model 1')
    axs[2].plot(mid_steps[2], step_intervals[2], 'o-', color='b', label='model 2')
    # ax1.set_ylabel('step')
    # ax1.set_xlabel('interval')
    # ax1.legend(loc='best')
    plt.title("Step Intervals")
    plt.savefig("./pinnadd_result4/adaptive_steps.png")
    plt.show()

if __name__ == "__main__":

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    seeds = [None, None, 4]
    for i in range(len(seeds)):
        if seeds[i] is None:
            continue
        print("random seed: ", seeds[i])
        config_pinnadd.random_seed = seeds[i]

        # train_pinn(if_train=True)
        # train_xpinn(if_train=True)
        # train_pinnadd(if_train=True)
        # train_pinnadd2(if_train=True)
        # train_pinnadd3(if_train=True)

    # train_pinnadd4(if_train=True)
    # train_pinnadd5(if_train=True)
    # summary()

    # to_analyze_gradient(if_train=False)

    # pinnadd2 total: 1969s, aggregate interface: 530s, train sgd: 1319
    # pinnadd4 total: 1573s, aggregate interface: 26s, train sgd: 1546

    # pinn total: 259s train_sgd: 230s # 数据集减少为1/3后是226s
    # xpinn total: 1895s train_sgd: 1400s, agregate interface: 348s

    # summary_frequency()



