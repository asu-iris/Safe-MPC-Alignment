import numpy as np
import casadi as cd
import torch
from torch.autograd.functional import jacobian

class cutter_torch(object):
    def __init__(self, name: str) -> None:
        self.name = name

    def set_state_dim(self, x_dim: int):
        self.x_dim = x_dim

    def set_ctrl_dim(self, u_dim: int):
        self.u_dim = u_dim

    def set_dyn(self, dyn_f: cd.Function):
        self.dyn_f = dyn_f

    def set_step_cost(self, step_cost: cd.Function):
        self.step_cost = step_cost

    def set_term_cost(self, term_cost: cd.Function):
        self.terminal_cost = term_cost

    def from_controller(self, controller):
        self.set_state_dim(controller.x_dim)
        self.set_ctrl_dim(controller.u_dim)
        self.set_dyn(controller.dyn_f)
        self.set_step_cost(controller.step_cost)
        self.set_term_cost(controller.terminal_cost)

    def construct_graph(self, horizon):
        assert hasattr(self, 'x_dim'), "missing x_dim"
        assert hasattr(self, 'u_dim'), "missing u_dim"
        assert hasattr(self, 'dyn_f'), "missing dyn_f"
        assert hasattr(self, 'step_cost'), "missing step_cost"
        assert hasattr(self, 'terminal_cost'), "terminal_cost"

        self.horizon = horizon
        init_state = cd.SX.sym('init_x', self.x_dim)

        target_r=cd.SX.sym('target_r', 3)

        # pre-define lists for casadi solver, joint optimization of x,u
        u_mx_list = []
        traj_xu = []

        J = 0

        # Xk = cd.MX.sym('X_0', self.x_dim) #Xk stands for the state in kth time step
        Xk = init_state
        traj_xu.append(Xk)

        for k in range(horizon):  # at step k, construct u_k and x_{k+1}
            Uk = cd.SX.sym('U_' + str(k), self.u_dim)
            # add u to var list
            u_mx_list.append(Uk)
            traj_xu.append(Uk)

            # dynamics
            Xk_1 = self.dyn_f(Xk, Uk)
            traj_xu.append(Xk_1)
            # J
            Ck = self.step_cost(Xk, Uk, target_r)
            J = J + Ck
            # update xk
            Xk = Xk_1
        # define trajectory
        traj_xu_flat = cd.vcat(traj_xu)
        traj_u_flat = cd.vcat(u_mx_list)
        # finish J
        J = J + self.terminal_cost(Xk_1, target_r)
        # Barrier

        J_grad = cd.gradient(J, traj_u_flat)
        self.J_grad_func = cd.Function('J_grad', [init_state, traj_u_flat, target_r], [J_grad])

    def calc_planes(self, init_state, traj_xu, human_corr, target_r, phi, phi_jacobi,gamma):
        traj_u = []
        for i in range(self.horizon):
            start_idx = i * (self.x_dim + self.u_dim) + self.x_dim
            end_idx = start_idx + self.u_dim
            traj_u += list(traj_xu[start_idx:end_idx])
        traj_u = np.array(traj_u)

        # print("grad shape", self.J_grad_func(init_state, traj_u, target_r).shape)
        h = -np.dot(human_corr.T, self.J_grad_func(init_state, traj_u, target_r)) * phi[1:]
        # print("h1 shape",h.shape)
        phi_jacobi_1_ = phi_jacobi[1:, :]
        h = h.flatten() + gamma * (phi_jacobi_1_ @ human_corr).flatten()
        print("h2 shape",h.shape)

        b = np.dot(human_corr.T, self.J_grad_func(init_state, traj_u, target_r)) * phi[0]
        phi_grad_0 = phi_jacobi[0, :]
        b = b - gamma * (phi_grad_0 @ human_corr)
        b = b.item()

        # theta^T h_phi <= b_phi
        h_phi = phi[1:]
        b_phi = -phi[0]

        return h, b, h_phi, b_phi
    
class Neural_phi(object):
    def __init__(self, phi_func, dyn_func, x_dim, u_dim, Horizon) -> None:
        self.phi_func = phi_func
        self.dyn_func = dyn_func
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.Horizon = Horizon
    
    def calc_phi(self,traj): #calculate feature and corresponding jacobian
        return self.phi_func(traj)

    def calc_phi_from_u(self,x_0,u_vec):
        calc_traj = [x_0]
        x = x_0
        for i in range(self.Horizon):
            u_i = u_vec[i*self.u_dim: (i+1) * self.u_dim]
            x = self.dyn_func(x,u_i)
            calc_traj.append(u_i)
            calc_traj.append(x)

        phi = self.phi_func(torch.concat(calc_traj))
        return phi
    
    def calc_phi_jac(self,traj):
        x_0 = traj[0:self.x_dim]
        u_seq = []
        for i in range(self.Horizon):
            bias = i * (self.x_dim + self.u_dim)
            u_i = traj[bias+self.x_dim: bias + self.x_dim+ self.u_dim]
            u_seq.append(u_i)

        u_vec = torch.concatenate(u_seq)
        u_vec.requires_grad= True

        jacobians = jacobian(self.calc_phi_from_u, inputs=(x_0,u_vec))
        jac_u = jacobians[1]
        return jac_u

        

            


        
