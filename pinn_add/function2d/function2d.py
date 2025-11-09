import os.path
import sys

import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import pinn_add
import torch
from .. import config_pinnadd
import json

def func2d(X):
    x1, x2 = X[:, 0:1], X[:, 1:2]
    center1 = np.array([-0.5, 0.5])
    center2 = np.array([0.5, 0.5])
    center3 = np.array([0, -0.5])
    a = 0.01
    sigma = 5
    component1 = 100*np.sqrt(np.square(x1-center1[0:1]) + np.square(x2-center1[1:2]))
    component11 = (np.sin(component1) - 0.5) / np.square(1 + a * component1) * np.exp(-(component1) / (2 * sigma**2))
    component2 = 100*np.sqrt(np.square(x1-center2[0:1]) + np.square(x2-center2[1:2]))
    component22 = (np.sin(component2) - 0.5) / np.square(1 + a * component2) * np.exp(-(component2) / (2 * sigma**2))
    component3 = 100*np.sqrt(np.square(x1-center3[0:1]) + np.square(x2-center3[1:2]))
    component33 = (np.sin(component3) - 0.5) / np.square(1 + a * component3) * np.exp(-(component3) / (2 * sigma**2))
    y = 20*component11 + 5*component22 + component33

    return y

def visualize_func2d():

    x = np.linspace(-1, 1, 200)
    y = np.linspace(-1, 1, 200)
    X, Y = np.meshgrid(x, y)
    Z = func2d(np.hstack((X.flatten()[:, None], Y.flatten()[:, None])))
    Z = Z.reshape(X.shape)
    plt.contourf(X, Y, Z, levels=40)
    plt.colorbar()
    plt.show()

spatial_domain = dde.geometry.Rectangle([-1, -1], [1, 1])
pointset = pinn_add.PointSets2D(spatial_domain, divide_config=None)

layers1 = [2, 50, 50, 50, 50, 1]
layers2 = [2, 50, 50, 50, 50, 1]
layers3 = [2, 50, 50, 50, 50, 1]
nets = [layers1, layers2, layers3]

print_every = 100
train_iterations = 20000
config_pinnadd.pretrain_iterations = 1000
config_pinnadd.adjust_subdomain_start_iteration = 5000
config_pinnadd.adjust_subdomain_end_iteration = 10000
config_pinnadd.adjust_subdomain_period = 2000

config_pinnadd.boundary_length_tolerance = 10
config_pinnadd.max_iter_times_weighted_balance = 5
config_pinnadd.lambda1 = 0.001

config_pinnadd.max_iter_times_weighted_min = 2
config_pinnadd.boundary_length_tolerance2 = 5

num_domain = 10000
n_clusters = 500
num_test = 20000

def train_pinn(if_train=True):
    net = [layers1]
    data = {
        "function": func2d,
        "pointset": pointset,
        "num_domain": num_domain,
        "n_clusters": n_clusters,
        "num_test": num_test
    }

    save_dir = './pinn_result/' + str(config_pinnadd.random_seed) + '/'
    pinnadd_model = pinn_add.PINNADD(net, data, save_dir=save_dir)

    if if_train:
        result_dict = pinnadd_model.train(train_iterations, print_every)
        pinnadd_model.save(save_dir, result_dict)
    else:
        pinnadd_model.restore(save_dir)

    test(pinnadd_model, save_dir)

def train_xpinn(if_train=True):
    data = {
        "function": func2d,
        "pointset": pointset,
        "num_domain": num_domain,
        "n_clusters": n_clusters,
        "num_test": num_test,
        "interface_type": "line"
    }

    save_dir = './xpinn_result/' + str(config_pinnadd.random_seed) + '/'
    pinnadd_model = pinn_add.PINNADD(nets, data, save_dir=save_dir)

    if if_train:
        result_dict = pinnadd_model.train(train_iterations, print_every)
        pinnadd_model.save(save_dir, result_dict)
    else:
        pinnadd_model.restore(save_dir)

    test(pinnadd_model, save_dir)

def train_pinnadd(if_train=True):
    data = {
        "function": func2d,
        "pointset": pointset,
        "num_domain": num_domain,
        "n_clusters": n_clusters,
        "num_test": num_test,
        "interface_type": "band"
    }

    save_dir = './pinnadd_result/' + str(config_pinnadd.random_seed) + '/'
    pinnadd_model = pinn_add.PINNADD(nets, data, save_dir=save_dir)

    if if_train:
        result_dict = pinnadd_model.train(train_iterations, print_every)
        pinnadd_model.save(save_dir, result_dict)
    else:
        pinnadd_model.restore(save_dir)

    test(pinnadd_model, save_dir)

def train_pinnadd2(if_train=True): # 更好的初始化的
    data = {
        "function": func2d,
        "pointset": pointset,
        "num_domain": num_domain,
        "n_clusters": n_clusters,
        "num_test": num_test,
        "interface_type": "band",
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
    data = {
        "function": func2d,
        "pointset": pointset,
        "num_domain": num_domain,
        "n_clusters": n_clusters,
        "num_test": num_test,
        "interface_type": "band",
        "initial_subdomain_balance": True,
        "adjust_subdomain": True,
    }

    save_dir = './pinnadd_result3/' + str(config_pinnadd.random_seed) + '/'
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
               delimiter=",", header=" ".join(metric_list), comments="")

    x = np.linspace(-1.0, 1.0, 200)
    y = np.linspace(-1.0, 1.0, 200)
    X, Y = np.meshgrid(x, y)
    X_star = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))
    U_pred = pinnadd_model.predict(X_star)
    U_pred = U_pred.reshape(X.shape)
    U_true = func2d(X_star)
    U_true = U_true.reshape(X.shape)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    extent = (-1.0, 1.0, -1.0, 1.0)

    h = axs[0].imshow(U_pred, interpolation='nearest', cmap='rainbow', extent=extent,
                      origin='lower', aspect='auto', vmax=10, vmin=-25)
    axs[0].set_xlabel(r'$x$')
    axs[0].set_ylabel(r'$y$')
    axs[0].legend(frameon=False, loc='best')
    axs[0].set_title(r'predicted $u(t,x)$', fontsize=10)
    fig.colorbar(h, ax=axs[0])


    h = axs[1].imshow(U_true, interpolation='nearest', cmap='rainbow', extent=extent,
                      origin='lower', aspect='auto', vmax=10, vmin=-25)
    axs[1].set_xlabel(r'$x$')
    axs[1].set_ylabel(r'$y$')
    axs[1].legend(frameon=False, loc='best')
    axs[1].set_title(r'true $u(t,x)$', fontsize=10)
    fig.colorbar(h, ax=axs[1])

    U_abs_res = np.abs(U_true - U_pred)
    h = axs[2].imshow(U_abs_res, interpolation='nearest', cmap='rainbow', extent=extent,
                      origin='lower', aspect='auto', vmax=10, vmin=-25)

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
                # "./pinnadd_result2/",
                "./pinnadd_result3/"
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
            loss_legend = ["Total Loss"]
            fig, axs = plt.subplots(1, 1, figsize=(6, 6))
            for i in range(len(subdomain_loss_curves)):
                subdomain_loss = subdomain_loss_curves[i]
                iterations = subdomain_loss[:, 0]  # 迭代次数
                loss = subdomain_loss[:, 1:2]  # Loss
                for j in range(loss.shape[1]):
                    axs.plot(iterations, loss[:, j], label=f'Model {i} - ' + f"{loss_legend[j]}", color=colors[i], linestyle=line_styles[j])

            axs.set_xlabel("Iteration", fontsize=20)
            axs.set_ylabel("Loss", fontsize=20)
            axs.tick_params(axis='y', labelsize=15)
            axs.set_title("Loss", fontsize=20)
            axs.legend(loc='best')
            axs.grid(True)
            fig.suptitle("Loss for Multiple Subdomains", fontsize=15)
            plt.tight_layout()
            plt.show()
            fig.savefig(d + "loss.png")

            # plot metrics
            fig, axs = plt.subplots(1, 2, figsize=(10, 6))
            for i in range(len(subdomain_loss_curves)):
                subdomain_loss = subdomain_loss_curves[i]
                iterations = subdomain_loss[:, 0]  # 迭代次数
                mse = subdomain_loss[:, 2]  # MSE
                l2_relative_error = subdomain_loss[:, 3]
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
        result_str = f"{metrics_mean[0]/1e-1:.3f} \\pm {metrics_std[0]/1e-1:.3f}, {metrics_mean[1]/1e-2:.3f} \\pm {metrics_std[1]/1e-2:.3f}, " + \
                        f"{train_time_mean[0]:.1f} \\pm {train_time_std[0]:.1f}"
        result_str_list.append(result_str)

    result_str_list = "\n".join(result_str_list)
    print(result_str_list)
    with open("metric_results.txt", "w") as f:
        f.write(result_str_list)


if __name__ == "__main__":
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    # visualize_func2d()

    seeds = [2, 5, 6]
    for i in range(len(seeds)):
        if seeds[i] is None:
            continue
        # print("random seed: ", seeds[i])
        # config_pinnadd.random_seed = seeds[i]
        # train_pinn(if_train=True)
        train_xpinn(if_train=True)
        # train_pinnadd(if_train=True)
        # train_pinnadd2(if_train=True)
        # train_pinnadd3(if_train=True)

    # summary()