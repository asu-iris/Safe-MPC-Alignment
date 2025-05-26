import numpy as np
import casadi as cd
# Optimal control Solver using casadi

class ocsolver_lin(object):
    def __init__(self, name: str) -> None:
        self.name = name
        self.print_level = 0

        self.opt_traj = None

    def set_state_param(self, x_dim: int, x_lb=None, x_ub=None):
        self.x_dim = x_dim
        if x_lb:
            self.x_lb = x_lb
        else:
            self.x_lb = self.x_dim * [-1e20]
        if x_ub:
            self.x_ub = x_ub
        else:
            self.x_ub = self.x_dim * [1e20]

    def set_ctrl_param(self, u_dim: int, u_lb: list, u_ub: list):
        self.u_dim = u_dim
        self.u_lb = u_lb.copy()
        self.u_ub = u_ub.copy()

    def set_dyn(self, dyn_f: cd.Function):
        self.dyn_f = dyn_f

    def set_step_cost(self, step_cost: cd.Function):
        self.step_cost = step_cost

    def set_term_cost(self, term_cost: cd.Function):
        self.terminal_cost = term_cost

    def set_gamma(self, gamma=0.001):
        self.gamma = gamma

    def construct_prob(self, horizon):  # joint optimization
        assert hasattr(self, 'x_dim'), "missing x_dim"
        assert hasattr(self, 'u_dim'), "missing u_dim"
        assert hasattr(self, 'u_lb'), "missing u_lb"
        assert hasattr(self, 'u_ub'), "missing u_ub"
        assert hasattr(self, 'dyn_f'), "missing dyn_f"
        assert hasattr(self, 'step_cost'), "missing step_cost"
        assert hasattr(self, 'terminal_cost'), "terminal_cost"

        # define param for cost function
        target_r = cd.SX.sym('target_r', 3)

        # pre-define lists for casadi solver, joint optimization of x,u
        self.w = []

        self.w_0 = []
        self.w_lb = []
        self.w_ub = []

        self.g = []
        self.g_lb = []
        self.g_ub = []

        J = 0

        x0 = cd.SX.sym('X_t_0', self.x_dim)  # Xk stands for the state in kth time step
        Xk = x0
        for k in range(horizon):  # at step k, construct u_k and x_{k+1}
            Uk = cd.SX.sym('U_t_' + str(k), self.u_dim)

            # add u to var list
            self.w.append(Uk)
            self.w_lb += self.u_lb
            self.w_ub += self.u_ub
            self.w_0 += [0.5 * (x + y) for x, y in zip(self.u_lb, self.u_ub)]

            # X
            Xk_1 = cd.SX.sym('X_t_' + str(k + 1), self.x_dim)
            self.w.append(Xk_1)
            self.w_lb += self.x_lb
            self.w_ub += self.x_ub
            self.w_0 += [0.5 * x + 0.5 * y for x, y in zip(self.x_lb, self.x_ub)]

            # dynamics
            X_ref = self.dyn_f(Xk, Uk)
            self.g.append(X_ref - Xk_1)
            self.g_lb += self.x_dim * [0]
            self.g_ub += self.x_dim * [0]

            # J
            Ck = self.step_cost(Xk, Uk, target_r)
            J = J + Ck

            # update xk
            Xk = Xk_1

        # finish J
        J = J + self.terminal_cost(Xk_1, target_r)

        # compute constraint g
        traj_xu = cd.vcat([x0] + self.w)
        

        g_center = cd.SX.sym('g_center', 1)

        traj_center = cd.SX.sym('traj_center', traj_xu.shape[0])

        g_grad_vec = cd.SX.sym('g_grad', traj_xu.shape[0])
        
        g_theta = cd.dot(g_grad_vec, traj_xu - traj_center) + g_center

        # Barrier
        B = J - self.gamma * cd.log(-g_theta)

        self.g_func = cd.Function('g_func', [x0, cd.vcat(self.w),traj_center, g_center, g_grad_vec], [g_theta])

        # self.B_func = cd.Function('B_func', [x0, cd.vcat(self.w), g_center, g_grad_vec, target_r], [B])

        param = cd.vertcat(x0, traj_center, g_center, g_grad_vec, target_r)
        self.param_func = cd.Function('param_func', [x0, traj_center ,g_center, g_grad_vec, target_r], [param])

        # constuct solver
        opts = {'ipopt.print_level': self.print_level, 'ipopt.sb': 'yes', 'print_time': self.print_level}
        prob = {'f': B, 'x': cd.vertcat(*self.w),
                'g': cd.vertcat(*self.g), 'p': param}
        self.solver_func = cd.nlpsol('solver', 'ipopt', prob, opts)

    def solve(self, init_state: np.ndarray, traj_center, g_center, g_grad, target_r: np.ndarray=0.0, initial_guess = None):

        # if hasattr(self, "warm_start_sol") and self.warm_start_sol is not None:
        #     self.initial_guess = self.warm_start_sol
        # else:
        self.initial_guess = initial_guess

        sol = self.solver_func(x0=self.initial_guess,
                               lbx=self.w_lb, ubx=self.w_ub,
                               lbg=self.g_lb,
                               ubg=self.g_ub,
                               p=self.param_func(init_state, traj_center, g_center, g_grad, target_r))

        w_opt = sol['x'].full().flatten()  # w_opt:[u_0,x_1,...,x_k-1]
        self.warm_start_sol = w_opt
        self.opt_traj = cd.vertcat(init_state, w_opt).full().flatten()

        g_value = self.g_func(init_state, w_opt, traj_center, g_center, g_grad)
        print("g value", g_value)
        if g_value > 0:
            print('g_value', g_value)
            raise Exception("violation appears in trajectory solved")
        # print('g_value', g_value)
        return w_opt

    def control(self, init_state, traj_center, g_center, g_grad, target_r,initial_guess):
        w_opt = self.solve(init_state, traj_center, g_center, g_grad, target_r,initial_guess)
        return w_opt[0:self.u_dim]

    def reset_warmstart(self):
        self.warm_start_sol = None
