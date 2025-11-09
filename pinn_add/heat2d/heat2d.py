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

pde_coef = (1 / np.square(500 * np.pi), 1 / np.square(np.pi))
init_coef=(20 * np.pi, np.pi)


def pde(x, u):
    u_xx = dde.grad.hessian(u, x, i=0, j=0)
    u_yy = dde.grad.hessian(u, x, i=1, j=1)
    u_t = dde.grad.jacobian(u, x, j=2)

    return [u_t - pde_coef[0] * u_xx - pde_coef[1] * u_yy]

def ref_sol(xt):
    return np.sin(init_coef[0] * xt[:, 0:1]) * np.sin(init_coef[1] * xt[:, 1:2]) * \
        np.exp(-(pde_coef[0] * init_coef[0] ** 2 + pde_coef[1] * init_coef[1] ** 2) * xt[:, 2:3])

t_end = 1
bbox = [0, 1, 0, 1, 0, t_end]

geom = dde.geometry.Rectangle(xmin=(bbox[0], bbox[2]), xmax=(bbox[1], bbox[3]))
timedomain = dde.geometry.TimeDomain(bbox[4], bbox[5])
geomtime = dde.geometry.GeometryXTime(geom, timedomain)
bc = dde.DirichletBC(geomtime, lambda x: 0, (lambda _, on_boundary: on_boundary), component=0)
ic = dde.IC(geomtime, ref_sol, (lambda _, on_initial: on_initial), component=0)

pointset = pinn_add.PointSets2D(geom, divide_config=None)

layers1 = [3, 30, 30, 30, 30, 1]
layers2 = [3, 30, 30, 30, 30, 1]
layers3 = [3, 30, 30, 30, 30, 1]
layers4 = [3, 30, 30, 30, 30, 1]
layers5 = [3, 30, 30, 30, 30, 1]
nets = [layers1, layers2, layers3, layers4, layers5]

t_span = [0, t_end]
t_steps = 30
n_clusters = 500
num_domain = 10000
num_boundary = 10000
loss_weight_for_each_domain = [1, 10, 100, 10, 1]
mannual_centers = [[0.1, 0.5], [0.3, 0.5], [0.5, 0.5], [0.7, 0.5], [0.9, 0.5]]

print_every = 100
train_iterations = 20000
config_pinnadd.adjust_subdomain_start_iteration = 6000
config_pinnadd.adjust_subdomain_end_iteration = 9000
config_pinnadd.adjust_subdomain_period = 2000

config_pinnadd.boundary_length_tolerance = 5
config_pinnadd.max_iter_times_weighted_balance = 2

config_pinnadd.max_iter_times_weighted_min = 1
config_pinnadd.boundary_length_tolerance2 = 2

config_pinnadd.lr_decay = ["step", 2000, 0.9]

def train_pinn(if_train=True):
    net = [layers1]
    data = {"pde": pde,
            "pointset": pointset,
            "num_domain": num_domain,
            "num_boundary": num_boundary,
            "t_span": t_span,
            "t_steps": t_steps,
            "n_clusters": n_clusters,
            "icbc": [bc, ic],
            "loss_weights": [loss_weight_for_each_domain for _ in range(1)],
            "ref_sol": ref_sol,
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
            "t_span": t_span,
            "t_steps": t_steps,
            "n_clusters": n_clusters,
            "icbc": [bc, ic],
            "loss_weights": [loss_weight_for_each_domain for _ in range(len(nets))],
            "ref_sol": ref_sol,
            "interface_type": "line",
            "mannual_centers": mannual_centers}

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
            "t_span": t_span,
            "t_steps": t_steps,
            "n_clusters": n_clusters,
            "icbc": [bc, ic],
            "loss_weights": [loss_weight_for_each_domain for _ in range(len(nets))],
            "ref_sol": ref_sol,
            "mannual_centers": mannual_centers,
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

def train_pinnadd2(if_train=True):
    data = {"pde": pde,
            "pointset": pointset,
            "num_domain": num_domain,
            "num_boundary": num_boundary,
            "t_span": t_span,
            "t_steps": t_steps,
            "n_clusters": n_clusters,
            "icbc": [bc, ic],
            "loss_weights": [loss_weight_for_each_domain for _ in range(len(nets))],
            "ref_sol": ref_sol,
            "mannual_centers": mannual_centers,
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

def train_pinnadd3(if_train=True):
    data = {"pde": pde,
            "pointset": pointset,
            "num_domain": num_domain,
            "num_boundary": num_boundary,
            "t_span": t_span,
            "t_steps": t_steps,
            "n_clusters": n_clusters,
            "icbc": [bc, ic],
            "loss_weights": [loss_weight_for_each_domain for _ in range(len(nets))],
            "ref_sol": ref_sol,
            "mannual_centers": mannual_centers,
            "interface_type": "band",
            "ignore_inter_pde_loss": True,
            "initial_subdomain_balance": True,
            "adjust_subdomain": True
            }

    config_pinnadd.pretrain_iterations = 100
    save_dir = './pinnadd_result3/' + str(config_pinnadd.random_seed) + '/'
    pinnadd_model = pinn_add.PINNADD(nets, data, save_dir=save_dir)
    if if_train:
        result_dict = pinnadd_model.train(train_iterations, print_every)
        pinnadd_model.save(save_dir, result_dict)
    else:
        pinnadd_model.restore(save_dir)

    test(pinnadd_model, save_dir)

def train_pinnadd4(if_train=True):
    data = {"pde": pde,
            "pointset": pointset,
            "num_domain": num_domain,
            "num_boundary": num_boundary,
            "t_span": t_span,
            "t_steps": t_steps,
            "n_clusters": n_clusters,
            "icbc": [bc, ic],
            "loss_weights": [loss_weight_for_each_domain for _ in range(len(nets))],
            "ref_sol": ref_sol,
            "mannual_centers": mannual_centers,
            "interface_type": "band",
            "ignore_inter_pde_loss": True,
            "analyze_gradient": True
            }

    config_pinnadd.random_seed = 0
    config_pinnadd.min_iteration = 100
    config_pinnadd.max_iteration = 500
    config_pinnadd.probe_grad_period = 50
    config_pinnadd.ratio_threshold = 0.5
    save_dir = './pinnadd_result4/' + str(config_pinnadd.ratio_threshold) + '/'
    pinnadd_model = pinn_add.PINNADD(nets, data, save_dir=save_dir)
    if if_train:
        result_dict = pinnadd_model.train(train_iterations, print_every)
        pinnadd_model.save(save_dir, result_dict)
    else:
        pinnadd_model.restore(save_dir)

    test(pinnadd_model, save_dir)

def train_pinnadd5(if_train=True, frequencey=1):
    data = {"pde": pde,
            "pointset": pointset,
            "num_domain": num_domain,
            "num_boundary": num_boundary,
            "t_span": t_span,
            "t_steps": t_steps,
            "n_clusters": n_clusters,
            "icbc": [bc, ic],
            "loss_weights": [loss_weight_for_each_domain for _ in range(len(nets))],
            "ref_sol": ref_sol,
            "mannual_centers": mannual_centers,
            "interface_type": "band",
            "ignore_inter_pde_loss": True,
            "analyze_gradient": True
            }

    # 1 1.021703e-04,3.069959e-02
    # 5 1.289956e-04,3.449512e-02
    # 10 1.176848e-04,3.294811e-02
    # 50 1.178529e-04,3.297163e-02
    # 100 8.218048e-05,2.753307e-02
    # 500 1.054188e-04,3.118382e-02
    # 1000 1.388349e-04,3.578653e-02
    # 2000 7.561256e-04,8.351553e-02
    # 5000 8.159662e-03,2.743509e-01
    config_pinnadd.random_seed = 0
    config_pinnadd.frequency = frequencey
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
            "t_span": t_span,
            "t_steps": t_steps,
            "n_clusters": n_clusters,
            "icbc": [bc, ic],
            "loss_weights": [loss_weight_for_each_domain for _ in range(len(nets))],
            "ref_sol": ref_sol,
            "mannual_centers": mannual_centers,
            "interface_type": "band",
            "ignore_inter_pde_loss": True,
            "analyze_gradient": True
            }

    config_pinnadd.max_iteration = 500
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
        end = 9990
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
               delimiter=",", header=" ".join(metric_list), comments="")

    x = np.linspace(0, 1, 101)
    y = np.linspace(0, 1, 101)
    X, Y = np.meshgrid(x, y)
    X_star = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))
    t_slice_num = 3
    t = np.linspace(0, t_end, t_slice_num)
    fig, axs = plt.subplots(2, t.shape[0], figsize=(t_slice_num*3, 6))
    extent = [0, 1, 0, 1]
    for i in range(t.shape[0]):
        t_slice = t[i]

        U_input = np.hstack((X_star, t_slice * np.ones((X_star.shape[0], 1))))
        U_pred = pinnadd_model.predict(U_input)
        U_true = ref_sol(U_input)
        metric_list = ["MSE", "l2 relative error"]
        errors = []
        print(f"t = {t[i]}")
        for metric in metric_list:
            metric_fn = dde.metrics.get(metric)
            u_err = metric_fn(U_pred, U_true).astype(float)
            errors.append(u_err)
            print(f"{metric}: {errors[-1]}")


        U_pred = U_pred.reshape(X.shape)
        U_true = U_true.reshape(X.shape)

        axs[0, i].contourf(X, Y, U_pred, cmap="coolwarm")
        axs[0, i].set_title(f"t = {t[i]} predicted")
        axs[1, i].contourf(X, Y, U_true, cmap="coolwarm")
        axs[1, i].set_title(f"t = {t[i]} true")

        # h = axs[0, i].imshow(U_pred, interpolation='nearest', cmap='rainbow', extent=extent,
        #                      origin='lower', aspect='auto')
        # axs[0, i].set_xlabel(r'$x$')
        # axs[0, i].set_ylabel(r'$y$')
        # axs[0, i].legend(frameon=False, loc='best')
        # axs[0, i].set_title(r'predicted $u(x,y)$, t=' + f"{t_slice}", fontsize=10)
        # fig.colorbar(h, ax=axs[0, i])
        #
        # h = axs[1, i].imshow(U_true, interpolation='nearest', cmap='rainbow', extent=extent,
        #                      origin='lower', aspect='auto')
        # axs[1, i].set_xlabel(r'$x$')
        # axs[1, i].set_ylabel(r'$y$')
        # axs[1, i].legend(frameon=False, loc='best')
        # axs[1, i].set_title(r'true $u(x,y)$, t=' + f"{t_slice}", fontsize=10)
        # fig.colorbar(h, ax=axs[1, i])

    plt.show()
    fig.savefig(save_dir + "result.png")

def summary():
    dir_list = ["./pinn_result/",
                "./xpinn_result/",
                "./pinnadd_result/",
                "./pinnadd_result2/",
                "./pinnadd_result3/"
                ]
    result_str_list = ["mse(std), l2 relative error(std), train time(std)"]
    subdomain_num = 5
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
            loss_legend = ["Pde loss", "BC loss", "IC loss", "Interface average loss", "Interface pde loss"]
            fig, axs = plt.subplots(1, 5, figsize=(25, 5))
            for i in range(len(subdomain_loss_curves)):
                subdomain_loss = subdomain_loss_curves[i]
                iterations = subdomain_loss[:, 0]  # 迭代次数
                loss = subdomain_loss[:, 1:6]  # Loss
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
                mse = subdomain_loss[:, 6]
                l2_relative_error = subdomain_loss[:, 7]
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
    # 1.021703e-04,3.069959e-02
    # 403s / 5307s
    # adaptive 0.5
    # 7.735363e-05,2.671226e-02
    # 4s / 4519s

    # 1 1.021703e-04,3.069959e-02
    # 5 1.289956e-04,3.449512e-02
    # 10 1.176848e-04,3.294811e-02
    # 50 1.178529e-04,3.297163e-02
    # 100 8.218048e-05,2.753307e-02
    # 500 1.054188e-04,3.118382e-02
    # 1000 1.388349e-04,3.578653e-02
    # 2000 7.561256e-04,8.351553e-02
    # 5000 8.159662e-03,2.743509e-01
    labels1 = ['1', '5', '10', '50', '100', '500', '1000', '2000', '5000']  # 横轴分类标签
    mse1 = [1.021703e-04, 1.289956e-04, 1.176848e-04, 1.178529e-04, 8.218048e-05, 1.054188e-04, 1.388349e-04, 7.561256e-04, 8.159662e-03]  # 左轴数据：MSE
    l2_err1 = [3.069959e-02, 3.449512e-02, 3.294811e-02, 3.297163e-02, 2.753307e-02, 3.118382e-02, 3.578653e-02, 8.351553e-02, 2.743509e-01]  # 右轴数据：L2相对误差

    # 0.3 1.572927e-04,3.809118e-02
    # 0.5 7.735363e-05,2.671226e-02
    # 0.7 1.047366e-04,3.108275e-02
    labels2 = ['0.3', '0.5', '0.7']  # 横轴分类标签
    mse2 = [1.572927e-04, 7.735363e-05, 1.047366e-04]  # 左轴数据：MSE
    l2_err2 = [3.809118e-02, 2.671226e-02, 3.108275e-02]  # 右轴数据：L2相对误差

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    line1 = ax1.plot(labels1, mse1, 'o-', color='k', label='MSE')
    ax1.set_ylabel('MSE')
    line2 = ax2.plot(labels1, l2_err1, 's--', color='k', label='L2 relative error')
    ax2.set_ylabel('L2 relative error')
    ax1.set_xlabel('Step Interval')
    ax1.set_yscale('log')
    ax1.set_ylim(5e-5, 1e-2)
    ax2.set_ylim(0, 0.3)
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
    ax1.set_yscale('log')
    ax1.set_ylim(5e-5, 1e-2)
    ax2.set_ylim(0, 0.3)
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best')
    plt.title("Adaptive Synchronization Frequency")
    plt.savefig("./pinnadd_result4/adaptive_frequency.png")
    plt.show()

    # dir = "./pinnadd_result4/0.3/"
    # step_intervals = []
    # mid_steps = []
    # for i in range(5):
    #     if os.path.exists(dir + f"/model-{i}-history.txt"):
    #         subdomain_loss = np.loadtxt(dir + f"/model-{i}-history.txt", delimiter=",")
    #         subdomain_loss = subdomain_loss[:-1, :]
    #         steps = subdomain_loss[:, 0]
    #         step_interval = np.diff(steps)
    #         mid_step = [(steps[i] + steps[i + 1]) / 2 for i in range(len(step_interval))]
    #         step_intervals.append(step_interval)
    #         mid_steps.append(mid_step)
    #         pass
    #
    # fig, axs = plt.subplots(5, 1)
    # axs[0].plot(mid_steps[0], step_intervals[0], 'o-', color='r', label='model 0')
    # axs[1].plot(mid_steps[1], step_intervals[1], 'o-', color='g', label='model 1')
    # axs[2].plot(mid_steps[2], step_intervals[2], 'o-', color='b', label='model 2')
    # axs[3].plot(mid_steps[3], step_intervals[3], 'o-', color='k', label='model 3')
    # axs[4].plot(mid_steps[4], step_intervals[4], 'o-', color='y', label='model 4')
    # # ax1.set_ylabel('step')
    # # ax1.set_xlabel('interval')
    # # ax1.legend(loc='best')
    # plt.title("Step Intervals")
    # plt.savefig("./pinnadd_result4/adaptive_steps.png")
    # plt.show()

if __name__ == "__main__":
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    # seeds = [0, 1, 2]
    # for i in range(len(seeds)):
    #     if seeds[i] is None:
    #         continue
    #     print("random seed: ", seeds[i])
    #     config_pinnadd.random_seed = seeds[i]
        # train_pinn(if_train=True)
        # train_xpinn(if_train=True)
        # train_pinnadd(if_train=True)
        # train_pinnadd2(if_train=True)
        # train_pinnadd3(if_train=True)

    # train_pinnadd4(if_train=True)
    # train_pinnadd5(if_train=True)

    # for frequency in [2000]:
    #     print("frequency: ", frequency)
    #     train_pinnadd5(if_train=True, frequencey=frequency)

    summary()
    # summary_frequency()
    # to_analyze_gradient(if_train=False)