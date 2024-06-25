import numpy as np
import casadi as cd
# Optimal control Solver using casadi
class ocsolver(object):
    def __init__(self,name:str) -> None:
        self.name=name
        self.print_level=0

    def set_state_param(self,x_dim:int,x_lb=None,x_ub=None):
        self.x_dim=x_dim
        if x_lb:
            self.x_lb = x_lb
        else:
            self.x_lb = self.x_dim * [-1e20]
        if x_ub:
            self.x_ub = x_ub
        else:
            self.x_ub = self.x_dim * [1e20]

    def set_ctrl_param(self,u_dim:int,u_lb:list,u_ub:list):
        self.u_dim=u_dim
        self.u_lb=u_lb.copy()
        self.u_ub=u_ub.copy()

    def set_dyn(self,dyn_f:cd.Function):
        self.dyn_f=dyn_f

    def set_step_cost(self,step_cost:cd.Function):
        self.step_cost=step_cost

    def set_term_cost(self,term_cost:cd.Function):
        self.terminal_cost=term_cost

    def set_g(self,Features:cd.Function,weights,gamma=0.001):
        self.features = Features
        self.weights=cd.MX(weights)
        self.gamma=gamma
        self.g_flag=True

    def solve(self,init_state,horizon):
        assert hasattr(self,'x_dim'), "missing x_dim"
        assert hasattr(self,'u_dim'), "missing u_dim"
        assert hasattr(self,'u_lb'), "missing u_lb"
        assert hasattr(self,'u_ub'), "missing u_ub"
        assert hasattr(self,'dyn_f'), "missing dyn_f"
        assert hasattr(self,'step_cost'), "missing step_cost"
        assert hasattr(self,'terminal_cost'), "terminal_cost"
        
        #if type(init_state) == numpy.ndarray:
            #init_state = init_state.flatten().tolist()
        
        #pre-define lists for casadi solver, joint optimization of x,u
        self.u_mx_list=[]
        self.x_mx_list=[]
        traj_xu=[]

        opt_mx_list=[]
        opt_mid_list=[]
        opt_lb_list=[]
        opt_ub_list=[]

        dyn_rel_list=[]
        dyn_lb_list=[]
        dyn_ub_list=[]

        J=0

        #Xk = cd.MX.sym('X_0', self.x_dim) #Xk stands for the state in kth time step
        Xk = cd.MX(init_state)
        self.x_mx_list.append(Xk)
        traj_xu.append(Xk)

        #opt_mx_list.append(Xk)
        #opt_lb_list += init_state
        #opt_ub_list += init_state
        #opt_mid_list += init_state

        for k in range(horizon): #at step k, construct u_k and x_{k+1}
            Uk=cd.MX.sym('U_' + str(k), self.u_dim)
            #add u to var list
            opt_mx_list.append(Uk)
            self.u_mx_list.append(Uk)
            traj_xu.append(Uk)
            opt_lb_list+=self.u_lb
            opt_ub_list+=self.u_ub
            opt_mid_list+= [0.5 * (x + y) - 2 for x, y in zip(self.u_lb, self.u_ub)]
            #add x to var list
            #Xk_1=cd.MX.sym('X_' + str(k+1), self.x_dim)
            #dynamics
            Xk_1=self.dyn_f(Xk,Uk)
            self.x_mx_list.append(Xk_1)
            traj_xu.append(Xk_1)
            #opt_mx_list.append(Xk_1)
            #opt_lb_list += self.x_lb
            #opt_ub_list += self.x_ub
            #opt_mid_list+= [0.5 * (x + y) for x, y in zip(self.x_lb, self.x_ub)]
            #dynamics
            #X_ref=self.dyn_f(Xk,Uk)
            #dyn_rel_list.append(X_ref - Xk_1)
            #dyn_lb_list += self.x_dim * [0]
            #dyn_ub_list += self.x_dim * [0]
            #J
            Ck=self.step_cost(Xk,Uk)
            J=J+Ck
            #update xk
            Xk=Xk_1
        #define trajectory
        traj_xu_flat=cd.vertcat(*traj_xu)
        traj_u_flat=cd.vertcat(*self.u_mx_list)
        #finish J
        J=J+self.terminal_cost(Xk_1)
        #Barrier
        B=J
        if hasattr(self,'g_flag'):
            expand_w=cd.vertcat(cd.MX([1]),self.weights)
            phi=self.features(traj_xu_flat)
            g_theta=cd.dot(expand_w,phi)
            B=B-self.gamma*cd.log(-g_theta)

        opts = {'ipopt.print_level': self.print_level, 'ipopt.sb': 'yes', 'print_time': self.print_level}
        prob = {'f': B, 'x': cd.vertcat(*opt_mx_list), 'g': cd.vertcat(*dyn_rel_list)}
        solver = cd.nlpsol('solver', 'ipopt', prob, opts)
        sol = solver(x0=opt_mid_list, lbx=opt_lb_list, ubx=opt_ub_list, lbg=dyn_lb_list, ubg=dyn_ub_list)
        w_opt = sol['x'].full().flatten() #w_opt:[u_0,u_1,...,u_k-1]
        x=self.dyn_f(init_state,w_opt[0])
        print(x)
        print('g', cd.Function('g_theta',[traj_u_flat],[g_theta])(w_opt))
        return w_opt
    
    def control(self,init_state,horizon):
        self.opt_traj=self.solve(init_state,horizon)
        return self.opt_traj[0:self.u_dim]

class ocsolver_fast(object):
    def __init__(self,name:str) -> None:
        self.name=name
        self.print_level=0

    def set_state_param(self,x_dim:int,x_lb=None,x_ub=None):
        self.x_dim=x_dim
        if x_lb:
            self.x_lb = x_lb
        else:
            self.x_lb = self.x_dim * [-1e20]
        if x_ub:
            self.x_ub = x_ub
        else:
            self.x_ub = self.x_dim * [1e20]

    def set_ctrl_param(self,u_dim:int,u_lb:list,u_ub:list):
        self.u_dim=u_dim
        self.u_lb=u_lb.copy()
        self.u_ub=u_ub.copy()

    def set_dyn(self,dyn_f:cd.Function):
        self.dyn_f=dyn_f

    def set_step_cost(self,step_cost:cd.Function):
        self.step_cost=step_cost

    def set_term_cost(self,term_cost:cd.Function):
        self.terminal_cost=term_cost

    def set_g(self,Features:cd.Function,weights,gamma=0.001):
        self.features = Features
        self.weights=cd.DM(weights)
        self.gamma=gamma
        self.g_flag=True
    
    def construct_prob(self,horizon): #joint optimization
        assert hasattr(self,'x_dim'), "missing x_dim"
        assert hasattr(self,'u_dim'), "missing u_dim"
        assert hasattr(self,'u_lb'), "missing u_lb"
        assert hasattr(self,'u_ub'), "missing u_ub"
        assert hasattr(self,'dyn_f'), "missing dyn_f"
        assert hasattr(self,'step_cost'), "missing step_cost"
        assert hasattr(self,'terminal_cost'), "terminal_cost"
        
        #self.init_state_t=SX.sym('init_x_t',self.x_dim)

        #pre-define lists for casadi solver, joint optimization of x,u
        self.u_sx_list_t=[]
        self.x_sx_list_t=[]
        self.traj_xu_t=[]

        #self.opt_mx_list_t=[]
        self.opt_mid_list_t=[]
        self.opt_lb_list_t=[]
        self.opt_ub_list_t=[]

        self.dyn_list_t=[]
        self.dyn_lb_list_t=[]
        self.dyn_ub_list_t=[]
        self.J_t=0

        Xk = cd.SX.sym('X_t_0', self.x_dim) #Xk stands for the state in kth time step
        #Xk = self.init_state_t
        #self.opt_mx_list_t.append(Xk)
        self.x_sx_list_t.append(Xk)
        self.traj_xu_t.append(Xk)
        #will modify the range of x0 when there is actual init state
        self.opt_lb_list_t+=self.x_lb
        self.opt_ub_list_t+=self.x_ub
        self.opt_mid_list_t+= [0.5 * (x + y) for x, y in zip(self.x_lb, self.x_ub)]

        for k in range(horizon): #at step k, construct u_k and x_{k+1}
            Uk=cd.SX.sym('U_t_' + str(k), self.u_dim)
            #add u to var list
            self.u_sx_list_t.append(Uk)
            self.traj_xu_t.append(Uk)
            self.opt_lb_list_t+=self.u_lb
            self.opt_ub_list_t+=self.u_ub
            self.opt_mid_list_t+= [0.5 * (x + y) for x, y in zip(self.u_lb, self.u_ub)]
            
            #X
            Xk_1=cd.SX.sym('X_t_' + str(k+1), self.x_dim)
            self.x_sx_list_t.append(Xk_1)
            self.traj_xu_t.append(Xk_1)

            self.opt_lb_list_t+=self.x_lb
            self.opt_ub_list_t+=self.x_ub
            self.opt_mid_list_t+= [0.5 * x + 0.5 * y  for x, y in zip(self.x_lb, self.x_ub)]
            #dynamics
            X_ref=self.dyn_f(Xk,Uk)
            self.dyn_list_t.append(X_ref-Xk_1)
            self.dyn_lb_list_t += self.x_dim * [0]
            self.dyn_ub_list_t += self.x_dim * [0]
            #J
            Ck=self.step_cost(Xk,Uk)
            self.J_t=self.J_t+Ck
            #update xk
            Xk=Xk_1
        #define trajectory
        self.traj_xu_flat_t=cd.vertcat(*self.traj_xu_t)
        self.traj_u_flat_t=cd.vertcat(*self.u_sx_list_t)
        #finish J
        self.J_t=self.J_t+self.terminal_cost(Xk_1)
        #Barrier
        self.B_t=self.J_t
        if hasattr(self,'g_flag'):
            expand_w=cd.vertcat(cd.DM([1]),self.weights)
            phi=self.features(self.traj_xu_flat_t)
            self.g_theta_t=cd.dot(expand_w,phi)
            self.B_t=self.B_t-self.gamma*cd.log(-self.g_theta_t)

        self.B_func_t=cd.Function('B_func',[self.traj_xu_flat_t],[self.B_t])
            
    def solve(self,init_state:np.ndarray):
        assert hasattr(self,'B_t'), "construct the problem first"
        #modify the range for x0
        self.opt_lb_list_t[0:self.x_dim]=list(init_state.flatten())
        self.opt_ub_list_t[0:self.x_dim]=list(init_state.flatten())
        self.opt_mid_list_t[0:self.x_dim]=list(init_state.flatten())

        obj_B=self.B_func_t(self.traj_xu_flat_t)

        opts = {'ipopt.print_level': self.print_level, 'ipopt.sb': 'yes', 'print_time': self.print_level}
        prob = {'f': obj_B, 'x': self.traj_xu_flat_t, 'g': cd.vertcat(*self.dyn_list_t)}
        solver = cd.nlpsol('solver', 'ipopt', prob, opts)
        if hasattr(self, "warm_start_sol") and self.warm_start_sol is not None:
            self.initial_guess=self.opt_mid_list_t.copy()
            self.initial_guess[self.x_dim:]=list(self.warm_start_sol)[self.x_dim:]
            #self.initial_guess=list(self.warm_start_sol)
        else:
            self.initial_guess=self.opt_mid_list_t
        sol = solver(x0=self.initial_guess, lbx=self.opt_lb_list_t, ubx=self.opt_ub_list_t, lbg=self.dyn_lb_list_t, ubg=self.dyn_ub_list_t)
        w_opt = sol['x'].full().flatten() #w_opt:[u_0,x_1,...,x_k-1]
        x=self.dyn_f(init_state,w_opt[0])
        #print(x)
        #g_val= cd.Function('g_theta',[self.traj_xu_flat_t],[self.g_theta_t])(w_opt)
        #print('g', g_val)
        #print('gamma*ln(-g)', self.gamma*np.log(-g_val))
        #print('B',sol['f'])
        if hasattr(self,'g_flag') and cd.Function('g_theta',[self.traj_xu_flat_t],[self.g_theta_t])(w_opt)>0:
            print(cd.Function('g_theta',[self.traj_xu_flat_t],[self.g_theta_t])(w_opt))
            raise Exception("violation appears in trajectory solved")
        self.warm_start_sol=w_opt
        return w_opt
    
    def control(self,init_state):
        self.opt_traj_t=self.solve(init_state)
        return self.opt_traj_t[self.x_dim:self.x_dim+self.u_dim]
    
    def reset_warmstart(self):
        self.warm_start_sol=None

class ocsolver_inner_Barrier(object):
    def __init__(self,name:str) -> None:
        self.name=name
        self.print_level=0

    def set_state_param(self,x_dim:int,x_lb=None,x_ub=None):
        self.x_dim=x_dim
        if x_lb:
            self.x_lb = x_lb
        else:
            self.x_lb = self.x_dim * [-1e20]
        if x_ub:
            self.x_ub = x_ub
        else:
            self.x_ub = self.x_dim * [1e20]

    def set_ctrl_param(self,u_dim:int,u_lb:list,u_ub:list):
        self.u_dim=u_dim
        self.u_lb=u_lb.copy()
        self.u_ub=u_ub.copy()

    def set_dyn(self,dyn_f:cd.Function):
        self.dyn_f=dyn_f

    def set_step_cost(self,step_cost:cd.Function):
        self.step_cost=step_cost

    def set_term_cost(self,term_cost:cd.Function):
        self.terminal_cost=term_cost

    def set_g(self,Features:cd.Function,weights,gamma=0.001):
        self.features = Features
        self.weights=cd.SX(weights)
        self.gamma=gamma
        self.g_flag=True
    
    def construct_prob(self,horizon): #joint optimization
        assert hasattr(self,'x_dim'), "missing x_dim"
        assert hasattr(self,'u_dim'), "missing u_dim"
        assert hasattr(self,'u_lb'), "missing u_lb"
        assert hasattr(self,'u_ub'), "missing u_ub"
        assert hasattr(self,'dyn_f'), "missing dyn_f"
        assert hasattr(self,'step_cost'), "missing step_cost"
        assert hasattr(self,'terminal_cost'), "terminal_cost"
        
        #self.init_state_t=SX.sym('init_x_t',self.x_dim)

        #pre-define lists for casadi solver, joint optimization of x,u
        self.u_mx_list_t=[]
        self.x_mx_list_t=[]
        self.traj_xu_t=[]

        #self.opt_mx_list_t=[]
        self.opt_mid_list_t=[]
        self.opt_lb_list_t=[]
        self.opt_ub_list_t=[]

        self.constraints_t=[]
        self.constraints_lb_t=[]
        self.constraints_ub_t=[]
        self.J_t=0

        Xk = cd.SX.sym('X_t_0', self.x_dim) #Xk stands for the state in kth time step
        #Xk = self.init_state_t
        #self.opt_mx_list_t.append(Xk)
        self.x_mx_list_t.append(Xk)
        self.traj_xu_t.append(Xk)
        #will modify the range of x0 when there is actual init state
        self.opt_lb_list_t+=self.x_lb
        self.opt_ub_list_t+=self.x_ub
        self.opt_mid_list_t+= [0.5 * (x + y) for x, y in zip(self.x_lb, self.x_ub)]

        for k in range(horizon): #at step k, construct u_k and x_{k+1}
            Uk=cd.SX.sym('U_t_' + str(k), self.u_dim)
            #add u to var list
            self.u_mx_list_t.append(Uk)
            self.traj_xu_t.append(Uk)
            self.opt_lb_list_t+=self.u_lb
            self.opt_ub_list_t+=self.u_ub
            self.opt_mid_list_t+= [0.5 * (x + y) for x, y in zip(self.u_lb, self.u_ub)]
            
            #X
            Xk_1=cd.SX.sym('X_t_' + str(k+1), self.x_dim)
            self.x_mx_list_t.append(Xk_1)
            self.traj_xu_t.append(Xk_1)

            self.opt_lb_list_t+=self.x_lb
            self.opt_ub_list_t+=self.x_ub
            self.opt_mid_list_t+= [0.5 * x + 0.5 * y  for x, y in zip(self.x_lb, self.x_ub)]
            #dynamics
            X_ref=self.dyn_f(Xk,Uk)
            self.constraints_t.append(X_ref-Xk_1)
            self.constraints_lb_t += self.x_dim * [0]
            self.constraints_ub_t += self.x_dim * [0]
            #J
            Ck=self.step_cost(Xk,Uk)
            self.J_t=self.J_t+Ck
            #update xk
            Xk=Xk_1
        #define trajectory
        self.traj_xu_flat_t=cd.vertcat(*self.traj_xu_t)
        self.traj_u_flat_t=cd.vertcat(*self.u_mx_list_t)
        #finish J
        self.J_t=self.J_t+self.terminal_cost(Xk_1)
        
        #Inner Barrier
        if hasattr(self,'g_flag'):
            expand_w=cd.vertcat(cd.DM([1]),self.weights)
            phi=self.features(self.traj_xu_flat_t)
            self.g_theta_t=cd.dot(expand_w,phi)
            #self.B_t=self.B_t-self.gamma*cd.log(-self.g_theta_t)
            self.constraints_t.append(self.g_theta_t)
            self.constraints_lb_t += [-np.inf]
            self.constraints_ub_t += [0]

        self.J_func_t=cd.Function('J_func',[self.traj_xu_flat_t],[self.J_t])
            
    def solve(self,init_state:np.ndarray):
        assert hasattr(self,'J_t'), "construct the problem first"
        #modify the range for x0
        self.opt_lb_list_t[0:self.x_dim]=list(init_state.flatten())
        self.opt_ub_list_t[0:self.x_dim]=list(init_state.flatten())
        self.opt_mid_list_t[0:self.x_dim]=list(init_state.flatten())

        obj_B=self.J_func_t(self.traj_xu_flat_t)
        if hasattr(self,'g_flag'):
            opts = {'ipopt.print_level': self.print_level, 'ipopt.sb': 'yes', 'ipopt.mu_target': self.gamma, 'print_time': self.print_level}
        else:
            opts = {'ipopt.print_level': self.print_level, 'ipopt.sb': 'yes', 'print_time': self.print_level}
        prob = {'f': obj_B, 'x': self.traj_xu_flat_t, 'g': cd.vertcat(*self.constraints_t)}
        solver = cd.nlpsol('solver', 'ipopt', prob, opts)
        if hasattr(self, "warm_start_sol") and self.warm_start_sol is not None:
            self.initial_guess=self.warm_start_sol
        else:
            self.initial_guess=self.opt_mid_list_t
        sol = solver(x0=self.initial_guess, lbx=self.opt_lb_list_t, ubx=self.opt_ub_list_t, lbg=self.constraints_lb_t, ubg=self.constraints_ub_t)
        w_opt = sol['x'].full().flatten() #w_opt:[u_0,x_1,...,x_k-1]
        x=self.dyn_f(init_state,w_opt[0])
        print(x)
        g_val= cd.Function('g_theta',[self.traj_xu_flat_t],[self.g_theta_t])(w_opt)
        print('g', g_val)
        print('gamma*ln(-g)', self.gamma*np.log(-g_val))
        print('B',sol['f'])
        if hasattr(self,'g_flag') and cd.Function('g_theta',[self.traj_xu_flat_t],[self.g_theta_t])(w_opt)>0:
            raise Exception("violation appears in trajectory solved")
        self.warm_start_sol=w_opt
        return w_opt
    
    def control(self,init_state):
        self.opt_traj_t=self.solve(init_state)
        return self.opt_traj_t[self.x_dim:self.x_dim+self.u_dim]
    
    def reset_warmstart(self):
        self.warm_start_sol=None

class ocsolver_v2(object):
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

    def set_g(self, Features: cd.Function, gamma=0.001):
        self.features = Features
        self.gamma = gamma
        self.g_flag = True

    def construct_prob(self, horizon):  # joint optimization
        assert hasattr(self, 'x_dim'), "missing x_dim"
        assert hasattr(self, 'u_dim'), "missing u_dim"
        assert hasattr(self, 'u_lb'), "missing u_lb"
        assert hasattr(self, 'u_ub'), "missing u_ub"
        assert hasattr(self, 'dyn_f'), "missing dyn_f"
        assert hasattr(self, 'step_cost'), "missing step_cost"
        assert hasattr(self, 'terminal_cost'), "terminal_cost"

        # self.init_state_t=SX.sym('init_x_t',self.x_dim)

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
            Ck = self.step_cost(Xk, Uk)
            J = J + Ck

            # update xk
            Xk = Xk_1

        # finish J
        J = J + self.terminal_cost(Xk_1)

        # compute constraint g
        traj_xu = cd.vcat([x0] + self.w)
        # Barrier
        B = J

        if hasattr(self, 'g_flag'):
            weight_param = cd.SX.sym('feature_weights', self.features.numel_out(0) - 1)
            feature_weights = cd.vertcat(1, weight_param)
            g_theta = cd.dot(self.features(traj_xu), feature_weights)
            B = B - self.gamma * cd.log(-g_theta)
        else:
            weight_param = cd.SX.sym('feature_weights', 0)
            g_theta = -10000.0

        self.g_func = cd.Function('g_func', [x0, cd.vcat(self.w), weight_param], [g_theta])

        self.B_func = cd.Function('B_func', [x0, cd.vcat(self.w), weight_param], [B])

        param = cd.vertcat(x0, weight_param)
        self.param_func = cd.Function('param_func', [x0, weight_param], [param])

        # constuct solver
        opts = {'ipopt.print_level': self.print_level, 'ipopt.sb': 'yes', 'print_time': self.print_level}
        prob = {'f': B, 'x': cd.vertcat(*self.w),
                'g': cd.vertcat(*self.g), 'p': param}
        self.solver_func = cd.nlpsol('solver', 'ipopt', prob, opts)

    def solve(self, init_state: np.ndarray, weights: np.ndarray = 0.0):

        if hasattr(self, "warm_start_sol") and self.warm_start_sol is not None:
            self.initial_guess = self.warm_start_sol
        else:
            self.initial_guess = cd.vcat(self.w_0)

        sol = self.solver_func(x0=self.initial_guess,
                               lbx=self.w_lb, ubx=self.w_ub,
                               lbg=self.g_lb,
                               ubg=self.g_ub,
                               p=self.param_func(init_state, weights))

        w_opt = sol['x'].full().flatten()  # w_opt:[u_0,x_1,...,x_k-1]
        self.warm_start_sol = w_opt
        self.opt_traj = cd.vertcat(init_state, w_opt).full().flatten()

        # x = self.dyn_f(init_state, w_opt[0])
        # print(x)
        # g_val= cd.Function('g_theta',[self.traj_xu_flat_t],[self.g_theta_t])(w_opt)
        # print('g', g_val)
        # print('gamma*ln(-g)', self.gamma*np.log(-g_val))
        # print('B',sol['f'])
        g_value = self.g_func(init_state, w_opt, weights)
        if hasattr(self, 'g_flag') and g_value > 0:
            print('g_value', g_value)
            raise Exception("violation appears in trajectory solved")
        # print('g_value', g_value)
        return w_opt

    def control(self, init_state, weights=0.0):
        w_opt= self.solve(init_state, weights)
        return w_opt[0:self.u_dim]

    def reset_warmstart(self):
        self.warm_start_sol = None

class ocsolver_v3(object):
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

    def set_g(self, Features: cd.Function, gamma=0.001):
        self.features = Features
        self.gamma = gamma
        self.g_flag = True

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
        # Barrier
        B = J

        if hasattr(self, 'g_flag'):
            weight_param = cd.SX.sym('feature_weights', self.features.numel_out(0) - 1)
            feature_weights = cd.vertcat(1, weight_param)
            g_theta = cd.dot(self.features(traj_xu), feature_weights)
            B = B - self.gamma * cd.log(-g_theta)
        else:
            weight_param = cd.SX.sym('feature_weights', 0)
            g_theta = -10000.0

        self.g_func = cd.Function('g_func', [x0, cd.vcat(self.w), weight_param], [g_theta])

        self.B_func = cd.Function('B_func', [x0, cd.vcat(self.w), weight_param, target_r], [B])

        param = cd.vertcat(x0, weight_param, target_r)
        self.param_func = cd.Function('param_func', [x0, weight_param, target_r], [param])

        # constuct solver
        opts = {'ipopt.print_level': self.print_level, 'ipopt.sb': 'yes', 'print_time': self.print_level}
        prob = {'f': B, 'x': cd.vertcat(*self.w),
                'g': cd.vertcat(*self.g), 'p': param}
        self.solver_func = cd.nlpsol('solver', 'ipopt', prob, opts)

    def solve(self, init_state: np.ndarray, weights: np.ndarray = 0.0, target_r: np.ndarray=0.0):

        if hasattr(self, "warm_start_sol") and self.warm_start_sol is not None:
            self.initial_guess = self.warm_start_sol
        else:
            self.initial_guess = cd.vcat(self.w_0)

        sol = self.solver_func(x0=self.initial_guess,
                               lbx=self.w_lb, ubx=self.w_ub,
                               lbg=self.g_lb,
                               ubg=self.g_ub,
                               p=self.param_func(init_state, weights, target_r))

        w_opt = sol['x'].full().flatten()  # w_opt:[u_0,x_1,...,x_k-1]
        self.warm_start_sol = w_opt
        self.opt_traj = cd.vertcat(init_state, w_opt).full().flatten()

        # x = self.dyn_f(init_state, w_opt[0])
        # print(x)
        # g_val= cd.Function('g_theta',[self.traj_xu_flat_t],[self.g_theta_t])(w_opt)
        # print('g', g_val)
        # print('gamma*ln(-g)', self.gamma*np.log(-g_val))
        # print('B',sol['f'])
        g_value = self.g_func(init_state, w_opt, weights)
        if hasattr(self, 'g_flag') and g_value > 0:
            print('g_value', g_value)
            raise Exception("violation appears in trajectory solved")
        # print('g_value', g_value)
        return w_opt

    def control(self, init_state, weights=0.0, target_r=0.0):
        w_opt = self.solve(init_state, weights, target_r)
        return w_opt[0:self.u_dim]

    def reset_warmstart(self):
        self.warm_start_sol = None

class ocsolver_v4(object):
    def __init__(self, name: str) -> None:
        self.name = name
        self.print_level = 0

        self.opt_traj = None

    def set_state_param(self, x_dim: int, x_lb=None, x_ub=None): #dim and bound for x
        self.x_dim = x_dim
        if x_lb:
            self.x_lb = x_lb
        else:
            self.x_lb = self.x_dim * [-1e20]
        if x_ub:
            self.x_ub = x_ub
        else:
            self.x_ub = self.x_dim * [1e20]

    def set_ctrl_param(self, u_dim: int, u_lb: list, u_ub: list): #dim and bound for u
        self.u_dim = u_dim
        self.u_lb = u_lb.copy()
        self.u_ub = u_ub.copy()

    def set_dyn(self, dyn_f: cd.Function): #dynamics
        self.dyn_f = dyn_f

    def set_step_cost(self, step_cost: cd.Function):
        self.step_cost = step_cost

    def set_term_cost(self, term_cost: cd.Function):
        self.terminal_cost = term_cost

    def set_g(self, Features: cd.Function, gamma=0.001): # safety constraint
        self.features = Features
        self.gamma = gamma
        self.g_flag = True

    def construct_prob(self, horizon):  # joint optimization, construct optimal control problem
        assert hasattr(self, 'x_dim'), "missing x_dim"
        assert hasattr(self, 'u_dim'), "missing u_dim"
        assert hasattr(self, 'u_lb'), "missing u_lb"
        assert hasattr(self, 'u_ub'), "missing u_ub"
        assert hasattr(self, 'dyn_f'), "missing dyn_f"
        assert hasattr(self, 'step_cost'), "missing step_cost"
        assert hasattr(self, 'terminal_cost'), "terminal_cost"

        # define param for cost function
        target_x = cd.SX.sym('target_x', self.x_dim)

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
            Ck = self.step_cost(Xk, Uk, target_x)
            J = J + Ck

            # update xk
            Xk = Xk_1

        # finish J
        J = J + self.terminal_cost(Xk_1, target_x)

        # compute constraint g
        traj_xu = cd.vcat([x0] + self.w)
        # Barrier
        B = J

        if hasattr(self, 'g_flag'):
            weight_param = cd.SX.sym('feature_weights', self.features.numel_out(0) - 1)
            feature_weights = cd.vertcat(1, weight_param)
            g_theta = cd.dot(self.features(traj_xu), feature_weights)
            B = B - self.gamma * cd.log(-g_theta)
        else:
            weight_param = cd.SX.sym('feature_weights', 0)
            g_theta = -10000.0

        self.g_func = cd.Function('g_func', [x0, cd.vcat(self.w), weight_param], [g_theta])

        self.B_func = cd.Function('B_func', [x0, cd.vcat(self.w), weight_param, target_x], [B])

        param = cd.vertcat(x0, weight_param, target_x)
        self.param_func = cd.Function('param_func', [x0, weight_param, target_x], [param])

        # constuct solver
        opts = {'ipopt.print_level': self.print_level, 'ipopt.sb': 'yes', 'print_time': self.print_level}
        prob = {'f': B, 'x': cd.vertcat(*self.w),
                'g': cd.vertcat(*self.g), 'p': param}
        self.solver_func = cd.nlpsol('solver', 'ipopt', prob, opts)

    def solve(self, init_state: np.ndarray, weights: np.ndarray = 0.0, target_x: np.ndarray=0.0): # solve the problem

        if hasattr(self, "warm_start_sol") and self.warm_start_sol is not None:
            self.initial_guess = self.warm_start_sol
        else:
            self.initial_guess = cd.vcat(self.w_0)

        sol = self.solver_func(x0=self.initial_guess,
                               lbx=self.w_lb, ubx=self.w_ub,
                               lbg=self.g_lb,
                               ubg=self.g_ub,
                               p=self.param_func(init_state, weights, target_x))

        w_opt = sol['x'].full().flatten()  # w_opt:[u_0,x_1,...,x_k-1]
        self.warm_start_sol = w_opt
        self.opt_traj = cd.vertcat(init_state, w_opt).full().flatten()

        # x = self.dyn_f(init_state, w_opt[0])
        # print(x)
        # g_val= cd.Function('g_theta',[self.traj_xu_flat_t],[self.g_theta_t])(w_opt)
        # print('g', g_val)
        # print('gamma*ln(-g)', self.gamma*np.log(-g_val))
        # print('B',sol['f'])
        g_value = self.g_func(init_state, w_opt, weights)
        #phi = self.features(self.opt_traj)
        # print('phi', phi)
        if hasattr(self, 'g_flag') and g_value > 0:
            print('g_value', g_value)
            print('phi', phi)
            raise Exception("violation appears in trajectory solved")
        # print('g_value', g_value)
        return w_opt

    def control(self, init_state, weights=0.0, target_x=0.0): #wrapper for solve, only pick the first control value
        w_opt = self.solve(init_state, weights, target_x)
        return w_opt[0:self.u_dim]

    def reset_warmstart(self):
        self.warm_start_sol = None
