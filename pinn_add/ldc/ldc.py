import deepxde as dde
import numpy as np
import torch
import pinn_add
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from shapely.geometry import Polygon, Point
import config_pinnadd
import os, json

def pde(x, u):
    Re = 100
    u_vel, v_vel, p = u[:, 0:1], u[:, 1:2], u[:, 2:]
    u_vel_x = dde.grad.jacobian(u, x, i=0, j=0)
    u_vel_y = dde.grad.jacobian(u, x, i=0, j=1)
    u_vel_xx = dde.grad.hessian(u, x, component=0, i=0, j=0)
    u_vel_yy = dde.grad.hessian(u, x, component=0, i=1, j=1)

    v_vel_x = dde.grad.jacobian(u, x, i=1, j=0)
    v_vel_y = dde.grad.jacobian(u, x, i=1, j=1)
    v_vel_xx = dde.grad.hessian(u, x, component=1, i=0, j=0)
    v_vel_yy = dde.grad.hessian(u, x, component=1, i=1, j=1)

    p_x = dde.grad.jacobian(u, x, i=2, j=0)
    p_y = dde.grad.jacobian(u, x, i=2, j=1)

    momentum_x = (
        u_vel * u_vel_x + v_vel * u_vel_y + p_x - 1 / Re * (u_vel_xx + u_vel_yy)
    )
    momentum_y = (
        u_vel * v_vel_x + v_vel * v_vel_y + p_y - 1 / Re * (v_vel_xx + v_vel_yy)
    )
    continuity = u_vel_x + v_vel_y

    return [momentum_x, momentum_y, continuity]

spatial_domain = dde.geometry.Rectangle([0, 0], [1, 1])
pointset = pinn_add.PointSets2D(spatial_domain, divide_config=None)


def boundary_top(x, on_boundary):
    return on_boundary and np.isclose(x[1], 1)


def boundary_not_top(x, on_boundary):
    return on_boundary and not np.isclose(x[1], 1)


a = 4
bc_u_top = dde.DirichletBC(spatial_domain, lambda x: a * x[:, 0:1] * (1 - x[:, 0:1]), boundary_top, component=0)
bc_v_top = dde.DirichletBC(spatial_domain, lambda _: 0, boundary_top, component=1)
bc_u_not_top = dde.DirichletBC(spatial_domain, lambda _: 0, boundary_not_top, component=0)
bc_v_not_top = dde.DirichletBC(spatial_domain, lambda _: 0, boundary_not_top, component=1)
bc_p = dde.PointSetBC(np.array([[0, 0]]), np.array([[0]]), component=2)

ref_data = np.loadtxt("./lid_driven_a" + str(a) + ".dat", comments="%")

layers1 = [2, 30, 30, 30, 30, 3]
layers2 = [2, 30, 30, 30, 30, 3]
layers3 = [2, 30, 30, 30, 30, 3]
nets = [layers1, layers2, layers3]

print_every = 100
train_iterations = 20000
config_pinnadd.adjust_subdomain_start_iteration = 10000
config_pinnadd.adjust_subdomain_end_iteration = 15000
config_pinnadd.adjust_subdomain_period = 2000

config_pinnadd.boundary_length_tolerance = 10
config_pinnadd.max_iter_times_weighted_balance = 5

config_pinnadd.max_iter_times_weighted_min = 2
config_pinnadd.boundary_length_tolerance2 = 5


num_domain = 10000
num_boundary = 400
n_clusters = 500
loss_weight_for_each_domain = [1, 1, 1, 1]



def train_pinn(if_train=True):
    net = [layers1]
    data = {"pde": pde,
            "pointset": pointset,
            "num_domain": num_domain,
            "num_boundary": num_boundary,
            "n_clusters": n_clusters,
            "icbc": [bc_u_top, bc_v_top, bc_u_not_top, bc_v_not_top, bc_p],
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
            "icbc": [bc_u_top, bc_v_top, bc_u_not_top, bc_v_not_top, bc_p],
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
            "icbc": [bc_u_top, bc_v_top, bc_u_not_top, bc_v_not_top, bc_p],
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

def train_pinnadd2(if_train=True):
    data = {"pde": pde,
            "pointset": pointset,
            "num_domain": num_domain,
            "num_boundary": num_boundary,
            "n_clusters": n_clusters,
            "icbc": [bc_u_top, bc_v_top, bc_u_not_top, bc_v_not_top, bc_p],
            "loss_weights": [loss_weight_for_each_domain for _ in range(len(nets))],
            "ref_data": ref_data,
            "interface_type": "band",
            "ignore_inter_pde_loss": True,
            "initial_subdomain_balance": True}

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
            "n_clusters": n_clusters,
            "icbc": [bc_u_top, bc_v_top, bc_u_not_top, bc_v_not_top, bc_p],
            "loss_weights": [loss_weight_for_each_domain for _ in range(len(nets))],
            "ref_data": ref_data,
            "interface_type": "band",
            "ignore_inter_pde_loss": True,
            "initial_subdomain_balance": True,
            "adjust_subdomain": True}

    save_dir = './pinnadd_result3/' + str(config_pinnadd.random_seed) + '/'
    pinnadd_model = pinn_add.PINNADD(nets, data, save_dir=save_dir)
    if if_train:
        result_dict = pinnadd_model.train(train_iterations, print_every)
        pinnadd_model.save(save_dir, result_dict)
    else:
        pinnadd_model.restore(save_dir)

    test(pinnadd_model, save_dir)

def train_pinnadd4(if_train=True): #
    data = {"pde": pde,
            "pointset": pointset,
            "num_domain": num_domain,
            "num_boundary": num_boundary,
            "n_clusters": n_clusters,
            "icbc": [bc_u_top, bc_v_top, bc_u_not_top, bc_v_not_top, bc_p],
            "loss_weights": [loss_weight_for_each_domain for _ in range(len(nets))],
            "ref_data": ref_data,
            "interface_type": "band",
            "ignore_inter_pde_loss": True,
            "analyze_gradient": True}

    config_pinnadd.random_seed = 0
    config_pinnadd.min_iteration = 2
    config_pinnadd.max_iteration = 10
    config_pinnadd.probe_grad_period = 1
    config_pinnadd.ratio_threshold = 0.7

    save_dir = './pinnadd_result4/' + str(config_pinnadd.ratio_threshold) + '/'
    pinnadd_model = pinn_add.PINNADD(nets, data, save_dir=save_dir)
    if if_train:
        result_dict = pinnadd_model.train(train_iterations, print_every)
        pinnadd_model.save(save_dir, result_dict)
    else:
        pinnadd_model.restore(save_dir)

    test(pinnadd_model, save_dir)

def train_pinnadd5(if_train=True):
    data = {"pde": pde,
            "pointset": pointset,
            "num_domain": num_domain,
            "num_boundary": num_boundary,
            "n_clusters": n_clusters,
            "icbc": [bc_u_top, bc_v_top, bc_u_not_top, bc_v_not_top, bc_p],
            "loss_weights": [loss_weight_for_each_domain for _ in range(len(nets))],
            "ref_data": ref_data,
            "interface_type": "band",
            "ignore_inter_pde_loss": True,
            "analyze_gradient": True}

    # 1 4.327200e-04 5.306888e-04 8.344711e-05
    # 5 5.651196e-04 7.168719e-04 9.980212e-05
    # 10 8.147169e-04 8.720270e-04 2.481099e-04
    # 50 1.179899e-03 1.208461e-03 3.269763e-03
    # 100 1.489544e-03 1.544679e-03 6.003713e-03
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
            "icbc": [bc_u_top, bc_v_top, bc_u_not_top, bc_v_not_top, bc_p],
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

    u_true = test_vals[:, 0]
    v_true = test_vals[:, 1]
    p_true = test_vals[:, 2]
    u_pred = pred_vals[:, 0]
    v_pred = pred_vals[:, 1]
    p_pred = pred_vals[:, 2]
    metric_list = ["MSE"]
    errors = []
    for metric in metric_list:
        # mse = dde.metrics.get("MSE")
        metric_fn = dde.metrics.get(metric)
        u_err = metric_fn(u_true, u_pred).astype(float)
        v_err = metric_fn(v_true, v_pred).astype(float)
        p_err = metric_fn(p_true, p_pred).astype(float)
        errors.append([u_err, v_err, p_err])
        print(f"{metric}: {errors[-1]}")

    np.savetxt(save_dir + "metric.txt", np.array(errors).T, fmt="%.6e",
               delimiter=",", header=" ".join(metric_list), comments="")

    x = np.linspace(0., 1.0, 100)
    y = np.linspace(0., 1.0, 100)
    X, Y = np.meshgrid(x, y)
    X_star = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))

    U_true = griddata(ref_data[:, 0:2], ref_data[:, 2:5], X_star, method='cubic')
    u_true = U_true[:, 0]
    v_true = U_true[:, 1]
    p_true = U_true[:, 2]
    u_true_grid = u_true.reshape(X.shape)
    v_true_grid = v_true.reshape(X.shape)
    p_true_grid = p_true.reshape(X.shape)
    U_pred = pinnadd_model.predict(X_star)
    u_pred = U_pred[:, 0]
    v_pred = U_pred[:, 1]
    p_pred = U_pred[:, 2]
    u_pred_grid = u_pred.reshape(X.shape)
    v_pred_grid = v_pred.reshape(X.shape)
    p_pred_grid = p_pred.reshape(X.shape)

    fig, axs = plt.subplots(3, 4, figsize=(20, 12))
    extent = (0.0, 1.0, 0.0, 1.0)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    h = axs[0, 0].imshow(u_pred_grid, interpolation='nearest', cmap='rainbow', extent=extent,
                         origin='lower', aspect='auto')
    axs[0, 0].set_xlabel(r'$x$')
    axs[0, 0].set_ylabel(r'$y$')
    axs[0, 0].legend(frameon=False, loc='best')
    axs[0, 0].set_title(r'predicted $u(x,y)$', fontsize=10)
    fig.colorbar(h, ax=axs[0, 0])

    h = axs[0, 1].imshow(v_pred_grid, interpolation='nearest', cmap='rainbow', extent=extent,
                         origin='lower', aspect='auto')
    axs[0, 1].set_xlabel(r'$x$')
    axs[0, 1].set_ylabel(r'$y$')
    axs[0, 1].legend(frameon=False, loc='best')
    axs[0, 1].set_title(r'predicted $v(x,y)$', fontsize=10)
    fig.colorbar(h, ax=axs[0, 1])

    h = axs[0, 2].imshow(p_pred_grid, interpolation='nearest', cmap='rainbow', extent=extent,
                         origin='lower', aspect='auto')
    axs[0, 2].set_xlabel(r'$x$')
    axs[0, 2].set_ylabel(r'$y$')
    axs[0, 2].legend(frameon=False, loc='best')
    axs[0, 2].set_title(r'predicted $p(x,y)$', fontsize=10)
    fig.colorbar(h, ax=axs[0, 2])

    strm = axs[0, 3].streamplot(X, Y, u_pred_grid, v_pred_grid, color='k', linewidth=1, density=3)
    axs[0, 3].set_xlabel(r'$x$')
    axs[0, 3].set_ylabel(r'$y$')
    axs[0, 3].set_title('predicted velocity field', fontsize=10)

    h = axs[1, 0].imshow(u_true_grid, interpolation='nearest', cmap='rainbow', extent=extent,
                         origin='lower', aspect='auto')
    axs[1, 0].set_xlabel(r'$x$')
    axs[1, 0].set_ylabel(r'$y$')
    axs[1, 0].legend(frameon=False, loc='best')
    axs[1, 0].set_title(r'reference $u(x,y)$', fontsize=10)
    fig.colorbar(h, ax=axs[1, 0])

    h = axs[1, 1].imshow(v_true_grid, interpolation='nearest', cmap='rainbow', extent=extent,
                         origin='lower', aspect='auto')
    axs[1, 1].set_xlabel(r'$x$')
    axs[1, 1].set_ylabel(r'$y$')
    axs[1, 1].legend(frameon=False, loc='best')
    axs[1, 1].set_title(r'reference $v(x,y)$', fontsize=10)
    fig.colorbar(h, ax=axs[1, 1])

    h = axs[1, 2].imshow(p_true_grid, interpolation='nearest', cmap='rainbow', extent=extent,
                         origin='lower', aspect='auto')
    axs[1, 2].set_xlabel(r'$x$')
    axs[1, 2].set_ylabel(r'$y$')
    axs[1, 2].legend(frameon=False, loc='best')
    axs[1, 2].set_title(r'reference $p(x,y)$', fontsize=10)
    fig.colorbar(h, ax=axs[1, 2])

    strm = axs[1, 3].streamplot(X, Y, u_true_grid, v_true_grid, color='k', linewidth=1, density=3)
    axs[1, 3].set_xlabel(r'$x$')
    axs[1, 3].set_ylabel(r'$y$')
    axs[1, 3].set_title('reference velocity field', fontsize=10)

    h = axs[2, 0].imshow(np.abs(u_pred_grid - u_true_grid), interpolation='nearest', cmap='rainbow', extent=extent,
                         origin='lower', aspect='auto')
    axs[2, 0].set_xlabel(r'$x$')
    axs[2, 0].set_ylabel(r'$y$')
    axs[2, 0].legend(frameon=False, loc='best')
    axs[2, 0].set_title(r'absolute error $u(x,y)$', fontsize=10)
    fig.colorbar(h, ax=axs[2, 0])

    h = axs[2, 1].imshow(np.abs(v_pred_grid - v_true_grid), interpolation='nearest', cmap='rainbow', extent=extent,
                         origin='lower', aspect='auto')
    axs[2, 1].set_xlabel(r'$x$')
    axs[2, 1].set_ylabel(r'$y$')
    axs[2, 1].legend(frameon=False, loc='best')
    axs[2, 1].set_title(r'absolute error $v(x,y)$', fontsize=10)
    fig.colorbar(h, ax=axs[2, 1])

    h = axs[2, 2].imshow(np.abs(p_pred_grid - p_true_grid), interpolation='nearest', cmap='rainbow', extent=extent,
                         origin='lower', aspect='auto')
    axs[2, 2].set_xlabel(r'$x$')
    axs[2, 2].set_ylabel(r'$y$')
    axs[2, 2].legend(frameon=False, loc='best')
    axs[2, 2].set_title(r'absolute error $p(x,y)$', fontsize=10)
    fig.colorbar(h, ax=axs[2, 2])

    fig.suptitle(r'Lid Driven Cavity', fontsize=16)
    # 希望matplotlib的subplots某个子图是空的，不要坐标轴
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

    result_str_list = ["mse_u(std), mse_v(std), mse_p(std), train time(std)"]
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
            fig, axs = plt.subplots(1, 2, figsize=(20, 10))
            metrics_legend = ["u", "v", "p"]
            markers = ['o', '*', 's']
            for i in range(len(subdomain_loss_curves)):
                subdomain_loss = subdomain_loss_curves[i]
                iterations = subdomain_loss[:, 0]
                mse = subdomain_loss[:, 5:8]
                l2_relative_error = subdomain_loss[:, 8:11]
                for j in range(mse.shape[1]):
                    axs[0].plot(iterations, mse[:,j], label=f'Model {i} - MSE ' + metrics_legend[j], color=colors[j], linestyle=line_styles[i])
                for j in range(l2_relative_error.shape[1]):
                    axs[1].plot(iterations, l2_relative_error[:,j], label=f'Model {i} - L2 relative error ' + metrics_legend[j], color=colors[j],
                            linestyle=line_styles[i], marker=markers[j], markersize=3)

            axs[0].set_xlabel("Iteration", fontsize=40)
            axs[0].set_ylabel("MSE", fontsize=40)
            axs[0].tick_params(axis='y', labelsize=30)
            axs[0].set_title("MSE", fontsize=40)
            axs[0].legend(loc='best')
            axs[0].grid(True)

            axs[1].set_xlabel("Iteration", fontsize=40)
            axs[1].set_ylabel("L2 Relative Error", fontsize=40)
            axs[1].tick_params(axis='y', labelsize=30)
            axs[1].set_title("L2 Relative Error", fontsize=40)
            axs[1].legend(loc='best')
            axs[1].grid(True)
            axs[1].set_ylim([0, 1])

            fig.suptitle("Metrics for Multiple Subdomains", fontsize=30)
            plt.tight_layout()
            plt.show()
            fig.savefig(d + "metrics.png")

        metrics_each_run = np.array(metrics_each_run)
        metrics_mean = np.mean(metrics_each_run, axis=0)
        metrics_std = np.std(metrics_each_run, axis=0)
        train_time_each_run = np.array(train_time_each_run)
        train_time_mean = np.mean(train_time_each_run, axis=0)
        train_time_std = np.std(train_time_each_run, axis=0)
        result_str = f"{metrics_mean[0]/1e-4:.3f} \\pm {metrics_std[0]/1e-4:.3f}, {metrics_mean[1]/1e-4:.3f} \\pm {metrics_std[1]/1e-4:.3f}, " + \
                     f"{metrics_mean[2]/1e-4:.3f} \\pm {metrics_std[2]/1e-4:.3f}, {train_time_mean[0]:.1f} \\pm {train_time_std[0]:.1f}"
        result_str_list.append(result_str)

    result_str_list = "\n".join(result_str_list)
    print(result_str_list)
    with open("metric_results.txt", "w") as f:
        f.write(result_str_list)

def summary_frequency():
    # fixed 1
    # 4.327200e-04 5.306888e-04 8.344711e-05
    # 1059s / 3847s
    # adaptive 0.7
    # 8.518004e-04 9.731136e-04 1.561509e-04
    # 226s / 4391s

    # 1 4.327200e-04 5.306888e-04 8.344711e-05
    # 5 5.651196e-04 7.168719e-04 9.980212e-05
    # 10 8.147169e-04 8.720270e-04 2.481099e-04
    # 50 1.179899e-03 1.208461e-03 3.269763e-03
    # 100 1.489544e-03 1.544679e-03 6.003713e-03
    labels1 = ['1', '5', '10', '50', '100']  # 横轴分类标签
    mse_u1 = [4.327200e-04, 5.651196e-04, 8.147169e-04, 1.179899e-03, 1.489544e-03]
    mse_v1 = [5.306888e-04, 7.168719e-04, 8.720270e-04, 1.208461e-03, 1.544679e-03]
    mse_p1 = [8.344711e-05, 9.980212e-05, 2.481099e-04, 3.269763e-03, 6.003713e-03]

    # 0.3 8.074249e-04 8.672091e-04 1.446745e-03
    # 0.5 4.748159e-04 5.945358e-04 9.827131e-04
    # 0.7 8.518004e-04 9.731136e-04 1.561509e-04
    labels2 = ['0.3', '0.5', '0.7']  # 横轴分类标签
    mse_u2 = [8.074249e-04, 4.748159e-04, 8.518004e-04]
    mse_v2 = [8.672091e-04, 5.945358e-04, 9.731136e-04]
    mse_p2 = [1.446745e-03, 9.827131e-04, 1.561509e-04]

    fig, ax1 = plt.subplots()
    ax1.plot(labels1, mse_u1, 'o-', color='r', label='MSE u')
    ax1.plot(labels1, mse_v1, 'o-', color='g', label='MSE v')
    ax1.plot(labels1, mse_p1, 'o-', color='b', label='MSE p')
    ax1.set_ylabel('MSE')
    ax1.set_xlabel('Frequency')
    ax1.set_yscale('log')
    ax1.set_ylim(5e-5, 0.01)
    ax1.legend(loc='best')
    plt.title("Fixed Synchronization Frequency")
    plt.savefig("./pinnadd_result5/fixed_frequency.png")
    plt.show()

    fig, ax1 = plt.subplots()
    ax1.plot(labels2, mse_u2, 'o-', color='r', label='MSE u')
    ax1.plot(labels2, mse_v2, 'o-', color='g', label='MSE v')
    ax1.plot(labels2, mse_p2, 'o-', color='b', label='MSE p')
    ax1.set_ylabel('MSE')
    ax1.set_xlabel('Step Interval')
    ax1.set_yscale('log')
    ax1.set_ylim(5e-5, 0.01)
    ax1.legend(loc='best')
    plt.title("Adaptive Synchronization Frequency")
    plt.savefig("./pinnadd_result4/adaptive_frequency.png")
    plt.show()

    dir = "./pinnadd_result4/0.3/"
    step_intervals = []
    mid_steps = []
    for i in range(3):
        if os.path.exists(dir + f"/model-{i}-history.txt"):
            subdomain_loss = np.loadtxt(dir + f"/model-{i}-history.txt", delimiter=",")
            subdomain_loss = subdomain_loss[:-1,:]
            steps = subdomain_loss[:, 0]
            step_interval = np.diff(steps)
            mid_step = [(steps[i] + steps[i + 1]) / 2 for i in range(len(step_interval))]
            step_intervals.append(step_interval)
            mid_steps.append(mid_step)
            pass

    fig, axs = plt.subplots(3,1)
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

    # seeds = [None, None, 4]
    # for i in range(len(seeds)):
    #     if seeds[i] is None:
    #         continue
    #     print("random seed: ", seeds[i])
    #     config_pinnadd.random_seed = seeds[i]

        # train_pinn()
        # train_xpinn(if_train=True)
        # train_pinnadd(if_train=True)
        # train_pinnadd2(if_train=True)
        # train_pinnadd3(if_train=True)


    # train_pinnadd4(if_train=True)
    # train_pinnadd5(if_train=True)

    summary()
    # summary_frequency()
    # to_analyze_gradient(if_train=False)