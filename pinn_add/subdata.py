import numpy as np
import torch
import deepxde.icbc
from scipy.interpolate import griddata
from deepxde import gradients as grad

class PDE:
    def __init__(self,
                 col_pts,
                 pde_func,
                 bc_pts=None,
                 inter_pts=None,
                 icbc=None,
                 loss_weights=None,
                 pointset_bc=None,
                 ref_data=None,
                 ref_sol=None,
                 t_span=None,
                 t_steps=None,
                 ignore_inter_pde_loss=False,
                 analyze_gradient=False
                 ):

        self.pde = pde_func
        self.ignore_inter_pde_loss = ignore_inter_pde_loss
        self.analyze_gradient = analyze_gradient

        # numpy arrays to tensors
        self.col_pts = torch.as_tensor(col_pts)
        for i in range(len(inter_pts)):
            if inter_pts[i] is not None:
                inter_pts[i] = torch.as_tensor(inter_pts[i])
        self.inter_pts = inter_pts
        self.bc_pts = torch.as_tensor(bc_pts) if bc_pts is not None else None
        self.loss_weights = torch.as_tensor(loss_weights) if loss_weights is not None else None
        self.icbc = icbc
        self.ic = None

        # pointset_bc
        self.pointset_bc = pointset_bc
        self.pointset_bc_pts = None
        self.pointset_bc_vals = None
        self.pointset_bc_idx = None
        if self.pointset_bc is not None:
            self.pointset_bc_pts = self.pointset_bc[0]
            self.pointset_bc_vals = torch.as_tensor(self.pointset_bc[1])
            bc_pts = np.vstack([bc_pts, self.pointset_bc_pts])
            indices = []
            for row in self.pointset_bc_pts:
                match = np.where((bc_pts == row).all(axis=1))[0]  # 找到匹配行
                indices.append(match[0] if match.size > 0 else -1)
            self.pointset_bc_idx = np.array(indices)
            self.bc_pts = torch.as_tensor(bc_pts)
            self.pointset_bc_pts = torch.as_tensor(self.pointset_bc_pts)

        # filter different types of icbc
        self.bc_pts_filtered = []
        self.bc_vals_idxs = []
        for constraint in self.icbc:
            if isinstance(constraint, deepxde.icbc.DirichletBC):
                if hasattr(constraint.geom, "geometry"): # is time pde
                    bc_vals_idx = constraint.filter_idx(np.hstack((bc_pts, np.zeros((bc_pts.shape[0], 1)))))
                else:
                    bc_vals_idx = constraint.filter_idx(bc_pts)
                bc_inputs = bc_pts[bc_vals_idx]
                if len(bc_inputs) > 0:
                    bc_inputs = torch.as_tensor(bc_inputs)
                else:
                    bc_inputs = None
                self.bc_pts_filtered.append(bc_inputs)
                self.bc_vals_idxs.append(bc_vals_idx)
            elif isinstance(constraint, deepxde.icbc.IC):
                self.ic = constraint


        # is time pde
        self.ic_pts = None
        if t_span is not None and t_steps is not None and self.ic is not None:
            self.t_span = t_span
            self.t_steps = t_steps
            self.t_pts = torch.as_tensor(np.linspace(t_span[0], t_span[1], t_steps))

            self.col_pts = self.multiply_x_t(self.col_pts, self.t_pts)


            self.ic_pts = torch.as_tensor(np.hstack((col_pts, np.zeros((col_pts.shape[0], 1)))))

            for i in range(len(self.inter_pts)):
                if self.inter_pts[i] is not None:
                    self.inter_pts[i] = self.multiply_x_t(self.inter_pts[i], self.t_pts)

            if self.pointset_bc is not None:
                self.pointset_bc_pts = self.multiply_x_t(self.pointset_bc_pts, self.t_pts)
                self.pointset_bc_vals = self.multiply_x_t(self.pointset_bc_vals, self.t_pts)
                self.pointset_bc_idx = np.tile(self.pointset_bc_idx, (1, self.t_steps))

            if bc_pts is not None:
                self.bc_pts = self.multiply_x_t(self.bc_pts, self.t_pts)
                for i in range(len(self.bc_pts_filtered)):
                    if self.bc_pts_filtered[i] is not None:
                        self.bc_pts_filtered[i] = self.multiply_x_t(self.bc_pts_filtered[i], self.t_pts)
                        self.bc_vals_idxs[i] = np.tile(self.bc_vals_idxs[i], (1, self.t_steps))


        self.ref_data = ref_data
        self.ref_sol = ref_sol
        self.test_pts = None
        self.test_vals = None
        if self.ref_data is not None:
            self.test_pts = self.col_pts.cpu().detach().numpy()
            input_dim = self.test_pts.shape[1]
            test_pts = ref_data[:, 0:input_dim]
            test_vals = ref_data[:, input_dim:]
            if input_dim > 2:
                method = 'nearest'
            else:
                method = 'cubic'
            self.test_vals = griddata(test_pts, test_vals, self.test_pts, method=method)
            self.test_vals = torch.as_tensor(self.test_vals)
            self.test_pts = torch.as_tensor(self.test_pts)
        elif self.ref_sol is not None:
            self.test_pts = self.col_pts.cpu().detach().numpy()
            self.test_vals = self.ref_sol(self.test_pts)
            self.test_vals = torch.as_tensor(self.test_vals)
            self.test_pts = torch.as_tensor(self.test_pts)

    def bc_loss(self, model_outputs, loss_fn):
        loss = []
        pointset_component = 0
        for i in range(len(self.icbc)):
            constraint = self.icbc[i]
            if isinstance(constraint, deepxde.icbc.PointSetBC):
                pointset_component = constraint.component #todo: if there are multiple pointset components

            elif isinstance(constraint, deepxde.icbc.DirichletBC):
                if self.bc_pts_filtered[i] is None:
                    error = torch.tensor([[0.0]])
                else:
                    X = self.bc_pts_filtered[i].cpu().detach().numpy()
                    bc_inputs = self.bc_pts_filtered[i]
                    beg = 0
                    end = bc_inputs.shape[0]
                    bc_outputs = model_outputs[self.bc_vals_idxs[i]]
                    error = constraint.error(X, bc_inputs, bc_outputs, beg, end)
                loss.append(error)

            elif isinstance(constraint, deepxde.icbc.IC):
               continue

        if self.pointset_bc is not None:
            error = self.pointset_bc_vals - model_outputs[self.pointset_bc_idx][:,
                                            pointset_component:pointset_component + 1]
            loss.append(error)

        loss = torch.concatenate(loss, dim=0)
        loss = loss_fn(loss, torch.zeros_like(loss))
        return loss

    def ic_loss(self, model_outputs, loss_fn):
        if self.ic is None:
            raise ValueError("No initial condition is provided.")
        X = self.ic_pts.cpu().detach().numpy()
        error = self.ic.error(X, self.ic_pts, model_outputs, 0, len(X))
        loss = loss_fn(error, torch.zeros_like(error))
        return loss

    # tensor
    def multiply_x_t(self, x, t):
        result_list = []
        for row in t:
            repeated_row = row.repeat(x.size(0), 1)  # shape: (a行数, b列数)
            combined = torch.cat((x, repeated_row), dim=1)  # shape: (a行数, a列数 + b列数)
            result_list.append(combined)

        final_result = torch.cat(result_list, dim=0)

        return final_result

    def loss_all_for_inputs(self, model, loss_fn, col_points, bc_points=None):

        col_points_np = col_points
        col_points = torch.as_tensor(col_points)
        # bc_points_np = bc_points
        # bc_points = torch.as_tensor(bc_points) if bc_points is not None else None

        if self.ic_pts is not None:
            col_points = self.multiply_x_t(col_points, self.t_pts)
            # ic_pts_np = np.hstack((col_points_np, np.zeros((col_points_np.shape[0], 1))))
            # ic_pts = torch.as_tensor(ic_pts_np)
            # ic_pts_outputs = model.outputs(False, ic_pts)
            # ic_loss = self.ic.error(ic_pts_np, ic_pts, ic_pts_outputs, 0, len(ic_pts_np))
            # if bc_points_np is not None:
            #     bc_points = self.multiply_x_t(bc_points, self.t_pts)

        # if bc_points_np is not None:
        #     loss_bc_list = []
        #     for i in range(len(self.icbc)):
        #         constraint = self.icbc[i]
        #         if isinstance(constraint, deepxde.icbc.IC):
        #             continue
        #         if isinstance(constraint, deepxde.icbc.PointSetBC):
        #             pointset_bc_pts = self.pointset_bc[0].cpu().detach().numpy()
        #             pointset_bc_vals = self.pointset_bc[1].cpu().detach().numpy()
        #             if hasattr(constraint.geom, "geometry"):  # is time pde
        #                 pointset_bc_pts_idx = constraint.filter_idx(np.hstack((bc_points_np, np.zeros((bc_points_np.shape[0], 1)))))
        #             else:
        #                 bc_pts_idx = constraint.filter_idx(bc_points_np)
        #
        #             pointset_component = constraint.component
        #             pointset_bc_pts = torch.as_tensor(self.pointset_bc[0])
        #             pointset_bc_vals = torch.as_tensor(self.pointset_bc[1])
        #             if self.ic_pts is not None:
        #                 pointset_bc_pts = self.multiply_x_t(pointset_bc_pts, self.t_pts)
        #                 pointset_bc_vals = self.multiply_x_t(pointset_bc_vals, self.t_pts)
        #             pointset_outputs = model.outputs(False, pointset_bc_pts)
        #             error = pointset_bc_vals - pointset_outputs[:, pointset_component:pointset_component + 1]
        #             loss_bc_list.append(error)
        #
        #         elif isinstance(constraint, deepxde.icbc.DirichletBC):
        #             if hasattr(constraint.geom, "geometry"):  # is time pde
        #                 bc_pts_idx = constraint.filter_idx(np.hstack((bc_points_np, np.zeros((bc_points_np.shape[0], 1)))))
        #             else:
        #                 bc_pts_idx = constraint.filter_idx(bc_points_np)
        #
        #             if len(bc_pts_idx) == 0:
        #                 error = torch.tensor([[0.0]])
        #             else:
        #                 bc_pts = bc_points_np[bc_pts_idx]
        #                 X = bc_pts
        #                 bc_inputs = torch.as_tensor(bc_pts)
        #                 beg = 0
        #                 end = bc_inputs.shape[0]
        #                 bc_outputs = model.outputs(False, bc_inputs)
        #                 error = constraint.error(X, bc_inputs, bc_outputs, beg, end)
        #             loss_bc_list.append(error)
        #
        #         bc_loss = torch.concatenate(loss_bc_list, dim=0)
        #         bc_loss = loss_fn(bc_loss, torch.zeros_like(bc_loss))

        grad.clear()
        torch.cuda.empty_cache()
        pde_loss = model.outputs(False, col_points, operator=self.pde)
        if isinstance(pde_loss, list):
            pde_loss = torch.vstack(pde_loss)
        pde_loss = loss_fn(torch.zeros_like(pde_loss), pde_loss)

        return pde_loss
        # if self.loss_weights is None:
        #     return loss_fn(model_outputs, torch.zeros_like(model_outputs))
        # else:
        #     error = self.pde(model_outputs)
        #     loss = torch.sum(self.loss_weights * error)
        #     return loss

class Function:
    def __init__(self, col_pts, func):

        self.col_pts = torch.as_tensor(col_pts)
        self.func = func
        self.test_pts = col_pts
        self.test_vals = self.func(self.test_pts)
        self.test_vals = torch.as_tensor(self.test_vals)
        self.test_pts = torch.as_tensor(self.test_pts)

        self.col_pts_val = torch.as_tensor(self.func(col_pts))

    def function_loss(self, model_outputs, loss_fn):
        error = self.col_pts_val - model_outputs
        loss = loss_fn(error, torch.zeros_like(error))
        return loss

    def loss_all_for_inputs(self, model, loss_fn, col_points):
        outputs = model.outputs(False, torch.as_tensor(col_points))
        error = torch.as_tensor(self.func(col_points)) - outputs
        loss = loss_fn(error, torch.zeros_like(error))
        return loss