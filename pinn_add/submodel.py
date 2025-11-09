from distutils.command.config import config

import torch
import numpy as np

# from config_pinnadd import window_length
from deepxde import gradients as grad
from deepxde import losses as losses_module
from deepxde import metrics as metrics_module
from deepxde import optimizers
from deepxde import utils
from .subdata import Function, PDE
from . import config_pinnadd
from profilehooks import profile

class SubModel:
    def __init__(self, data, net):
        self.data = data
        self.net = net

        self.opt_name = None
        self.lr = None
        self.decay = None
        self.opt = None
        self.lr_scheduler = None
        self.loss_name = None
        self.loss_fn = None
        self.metrics = None
        self.test_epochs = []
        self.loss_history = []
        self.metrics_history = []

        self.pde_loss = None
        self.ic_loss = None
        self.bc_loss = None
        self.inter_avg_loss = None
        self.inter_pde_loss = None
        if hasattr(self.data, "inter_pts"):
            inter_num = len(self.data.inter_pts)
            self.aggregate_interface = [None for i in range(inter_num)]
        else:
            self.aggregate_interface = []
        self.total_loss = None
        self.ignore_inter_pde_loss = data.ignore_inter_pde_loss if isinstance(data, PDE) else False
        self.count_iteration = 0
        self.grads_history = []
        self.compile(decay=config_pinnadd.lr_decay)

    def compile(self, opt_name = "adam", lr = 0.001, decay = None, loss_name = "mse", metrics=("mse", "l2 relative error")):
        self.opt_name = opt_name
        self.lr = lr
        self.decay = decay
        trainable_variables = list(self.net.parameters())
        self.opt, self.lr_scheduler = optimizers.get(
            trainable_variables, self.opt_name, learning_rate=self.lr, decay=self.decay
        )
        self.loss_name = loss_name
        self.loss_fn = losses_module.get(self.loss_name)
        self.metrics = metrics
        self.count_iteration = 0

    # return list of interface outputs (u and F), tensor
    # with no grad

    def outputs_inter(self):
        inter_outputs = []
        for inter in self.data.inter_pts:
            if inter is not None:
                u_inter = self.outputs(False, inter)
                F_inter = self.outputs(False, inter, operator=self.data.pde)
                inter_outputs.append([u_inter, F_inter])
            else:
                inter_outputs.append(None)

        grad.clear()
        return inter_outputs

    def outputs(self, training, inputs, operator=None):
        self.net.train(mode=training)
        with torch.no_grad():
            inputs = torch.as_tensor(inputs)
            inputs.requires_grad_()

        if operator is None:
            outputs = self.net(inputs)
        else:
            outputs = operator(inputs, self.net(inputs))

        return outputs

    def calculate_losses(self):

        # if data is a function, only calculate 1 loss
        if isinstance(self.data, Function):
            outputs = self.outputs(True, self.data.col_pts)
            self.total_loss = self.data.function_loss(outputs, self.loss_fn)
            return self.total_loss

        if self.data.bc_pts is not None:
            self.bc_loss = self.data.bc_loss(self.outputs(True, self.data.bc_pts), self.loss_fn)
        else:
            self.bc_loss = torch.as_tensor(0.0)

        if self.data.ic_pts is not None:
            self.ic_loss = self.data.ic_loss(self.outputs(True, self.data.ic_pts), self.loss_fn)
        else:
            self.ic_loss = torch.as_tensor(0.0)


        pde_loss = self.outputs(True, self.data.col_pts, operator=self.data.pde)
        if isinstance(pde_loss, list):
            pde_loss = torch.vstack(pde_loss)
        self.pde_loss = self.loss_fn(torch.zeros_like(pde_loss), pde_loss)

        inter_avg_loss = []
        inter_pde_loss = []
        for i in range(len(self.data.inter_pts)):
            inter_pts = self.data.inter_pts[i]
            if inter_pts is not None:
                u_inter = self.outputs(True, inter_pts)
                F_inter = self.outputs(True, inter_pts, operator=self.data.pde)
                u_inter_avg = self.aggregate_interface[i][0]
                F_inter_adjacent = self.aggregate_interface[i][1]
                inter_avg_loss.append(self.loss_fn(u_inter_avg, u_inter))

                if isinstance(F_inter_adjacent, list):
                    F_inter_adjacent = torch.vstack(F_inter_adjacent)
                    F_inter = torch.vstack(F_inter)
                inter_pde_loss.append(self.loss_fn(F_inter_adjacent, F_inter))

        if len(inter_avg_loss) == 0:
            self.inter_avg_loss = torch.as_tensor(0.0)
            self.inter_pde_loss = torch.as_tensor(0.0)
        else:
            self.inter_avg_loss = torch.sum(torch.stack(inter_avg_loss))
            self.inter_pde_loss = torch.sum(torch.stack(inter_pde_loss))

        grad.clear()

        if self.ignore_inter_pde_loss:
            self.inter_pde_loss = torch.as_tensor(0.0)

        if self.data.ic_pts is None:
            total_loss = torch.stack([self.pde_loss, self.bc_loss, self.inter_avg_loss, self.inter_pde_loss])
        else:
            total_loss = torch.stack([self.pde_loss, self.bc_loss, self.ic_loss, self.inter_avg_loss, self.inter_pde_loss])

        if self.data.loss_weights is not None:
            self.total_loss = total_loss * torch.as_tensor(self.data.loss_weights)

        if self.data.analyze_gradient:
            self.total_loss = self.total_loss # Tensor
        else:
            self.total_loss = torch.sum(self.total_loss)

        return self.total_loss

    # @profile
    def train_sgd(self):
        grad.clear()
        anaylze_gradient = self.data.analyze_gradient if isinstance(self.data, PDE) else False
        min_iteration = config_pinnadd.min_iteration
        max_iteration = config_pinnadd.max_iteration
        probe_grad_period = config_pinnadd.probe_grad_period
        ratio_threshold = config_pinnadd.ratio_threshold
        iterations = max_iteration if anaylze_gradient else 1

        count_this_epoch = 0
        for i in range(iterations):
            stop_early = False
            self.calculate_losses()
            self.opt.zero_grad()
            ratio_list = []
            if anaylze_gradient and (count_this_epoch+1) % probe_grad_period == 0 and not config_pinnadd.static_frequency:
                num_interior_loss_term = 2 if self.data.ic_pts is None else 3  # time pde has 3 terms: col, bc, ic
                mean_grads = []

                for j in range(len(self.total_loss)):
                    self.total_loss[j].backward(retain_graph=True)
                    m_grad = []
                    for group in self.opt.param_groups:
                        for p in group['params']:
                            if p.grad is None:
                                m_grad.append(torch.zeros(p.size))
                            else:
                                m_grad.append(torch.abs(p.grad).reshape(-1))
                    m_grad = torch.cat(m_grad) ** 2
                    # m_grad = torch.abs(torch.cat(m_grad))
                    m_grad = torch.sum(m_grad) / m_grad.shape[0]
                    m_grad = m_grad.item()
                    mean_grads.append(m_grad)
                    self.opt.zero_grad()

                # print("iteration", self.count_iteration, "mean grads", mean_grads, "loss", self.total_loss.cpu().detach().numpy().astype(float).tolist())

                self.total_loss = torch.sum(self.total_loss)
                interior_grads = sum(mean_grads[:num_interior_loss_term])
                interface_grads = sum(mean_grads[num_interior_loss_term:])
                ratio = interface_grads / (interior_grads + interface_grads)
                self.grads_history.append([interior_grads, interface_grads, ratio])

                window_length = config_pinnadd.window_length
                ratio_list.append(ratio)
                if count_this_epoch > min_iteration:
                    ratio_avg = sum(ratio_list[-window_length:]) / len(ratio_list[-window_length:])
                    if ratio_avg < ratio_threshold and count_this_epoch > min_iteration:
                        stop_early = True
                if not count_this_epoch < max_iteration:
                    stop_early = True

            if anaylze_gradient:
                self.total_loss = torch.sum(self.total_loss)

            self.total_loss.backward()
            self.opt.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.count_iteration += 1
            count_this_epoch +=1

            # 判断是否提前结束训练
            if stop_early:
                # print("Early stop at iteration", self.count_iteration)
                break

        return self.count_iteration

    def train_lbfgs(self): # todo: closure() for lbfgs
        pass

    def test(self, iteration=None):

        if isinstance(self.data, Function):
            test_pts = self.data.test_pts
            test_vals = self.data.test_vals
            component_num = test_vals.shape[1]
            test_results = []
            for met in self.metrics:
                metric = metrics_module.get(met)
                tst_vals = utils.to_numpy(test_vals)
                pred_vals = utils.to_numpy(self.outputs(False, test_pts))
                tst_results = []
                for c in range(component_num):
                    result = metric(tst_vals[:, c], pred_vals[:, c])
                    tst_results.append(result)
                test_results.extend(tst_results)
            loss = self.total_loss.cpu().detach().numpy().astype(float).tolist()
            if iteration is not None:
                self.test_epochs.append(iteration)
            self.loss_history.append([loss])
            self.metrics_history.append(test_results)
            return loss, test_results

        test_pts = self.data.test_pts
        if test_pts is not None:
            test_vals = self.data.test_vals
            component_num = test_vals.shape[1]
            test_results = []
            for met in self.metrics:
                metric = metrics_module.get(met)
                tst_vals = utils.to_numpy(test_vals)
                pred_vals = utils.to_numpy(self.outputs(False, test_pts))
                tst_results = []
                for c in range(component_num):
                    result = metric(tst_vals[:, c], pred_vals[:, c])
                    tst_results.append(result)
                test_results.extend(tst_results)
        else:
            test_results = None

        if self.data.ic_pts is None:
            loss = np.hstack((utils.to_numpy(self.pde_loss), utils.to_numpy(self.bc_loss), utils.to_numpy(self.inter_avg_loss), utils.to_numpy(self.inter_pde_loss)))
        else:
            loss = np.hstack((utils.to_numpy(self.pde_loss), utils.to_numpy(self.bc_loss), utils.to_numpy(self.ic_loss), utils.to_numpy(self.inter_avg_loss), utils.to_numpy(self.inter_pde_loss)))

        loss = loss.astype(float).tolist()
        if iteration is not None:
            self.test_epochs.append(iteration)
        self.loss_history.append(loss)
        self.metrics_history.append(test_results)

        return loss, test_results


    def save(self, save_path):
        torch.save(self.net.state_dict(), f"{save_path}.pt")
        test_epochs = np.array(self.test_epochs, dtype=int).reshape(-1, 1)
        loss_history = np.array(self.loss_history)
        metrics_history = np.array(self.metrics_history)
        txt = np.hstack([test_epochs, loss_history, metrics_history])
        headers = "  ".join(["iteration", "loss"] + list(self.metrics))
        np.savetxt(f"{save_path}-history.txt", txt, delimiter=",", header=headers, fmt="%.6e")


    def restore(self, save_path):
        self.net.load_state_dict(torch.load(f"{save_path}.pt"))
        print("Model restored from", f"{save_path}.pt")