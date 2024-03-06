import numpy as np
from casadi import *

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

    def set_dyn(self,dyn_f:Function):
        self.dyn_f=dyn_f

    def set_step_cost(self,step_cost:Function):
        self.step_cost=step_cost

    def set_term_cost(self,term_cost:Function):
        self.terminal_cost=term_cost

    def solve(self,init_state,horizon):
        assert hasattr(self,'x_dim'), "missing x_dim"
        assert hasattr(self,'u_dim'), "missing u_dim"
        assert hasattr(self,'u_lb'), "missing u_lb"
        assert hasattr(self,'u_ub'), "missing u_ub"
        assert hasattr(self,'dyn_f'), "missing dyn_f"
        assert hasattr(self,'step_cost'), "missing step_cost"
        assert hasattr(self,'terminal_cost'), "terminal_cost"
        
        if type(init_state) == numpy.ndarray:
            init_state = init_state.flatten().tolist()
        
        #pre-define lists for casadi solver, joint optimization of x,u
        u_mx_list=[]
        x_mx_list=[]

        opt_mx_list=[]
        opt_mid_list=[]
        opt_lb_list=[]
        opt_ub_list=[]

        dyn_rel_list=[]
        dyn_lb_list=[]
        dyn_ub_list=[]

        J=0

        Xk = MX.sym('X_0', self.x_dim) #Xk stands for the state in kth time step
        x_mx_list.append(Xk)

        opt_mx_list.append(Xk)
        opt_lb_list += init_state
        opt_ub_list += init_state
        opt_mid_list += init_state

        for k in range(horizon): #at step k, construct u_k and x_{k+1}
            Uk=MX.sym('U_' + str(k), self.u_dim)
            #add u to var list
            opt_mx_list.append(Uk)
            u_mx_list.append(Uk)
            opt_lb_list+=self.u_lb
            opt_ub_list+=self.u_ub
            opt_mid_list+= [0.5 * (x + y) for x, y in zip(self.u_lb, self.u_ub)]
            #add x to var list
            Xk_1=MX.sym('X_' + str(k+1), self.x_dim)
            x_mx_list.append(Xk_1)
            opt_mx_list.append(Xk_1)
            opt_lb_list += self.x_lb
            opt_ub_list += self.x_ub
            opt_mid_list+= [0.5 * (x + y) for x, y in zip(self.x_lb, self.x_ub)]
            #dynamics
            X_ref=self.dyn_f(Xk,Uk)
            dyn_rel_list.append(X_ref - Xk_1)
            dyn_lb_list += self.x_dim * [0]
            dyn_ub_list += self.x_dim * [0]
            #J
            Ck=self.step_cost(Xk,Uk)
            J=J+Ck
            #update xk
            Xk=Xk_1

        #finish J
        J=J+self.terminal_cost(Xk_1)

        opts = {'ipopt.print_level': self.print_level, 'ipopt.sb': 'yes', 'print_time': self.print_level}
        prob = {'f': J, 'x': vertcat(*opt_mx_list), 'g': vertcat(*dyn_rel_list)}
        solver = nlpsol('solver', 'ipopt', prob, opts)
        sol = solver(x0=opt_mid_list, lbx=opt_lb_list, ubx=opt_ub_list, lbg=dyn_lb_list, ubg=dyn_ub_list)
        w_opt = sol['x'].full().flatten() #w_opt:[x_0,u_0,x_1,..u_{k-1},x_k]

        return w_opt
    
    def control(self,init_state,horizon):
        opt_traj=self.solve(init_state,horizon)
        return opt_traj[self.x_dim:self.x_dim+self.u_dim]







