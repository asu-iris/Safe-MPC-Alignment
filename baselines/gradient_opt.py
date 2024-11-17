import numpy as np
import casadi as cd
from baselines.adam import AdamWithGrad

class gd_optimizer(object):
    def __init__(self,name:str,eta=0.01) -> None:
        self.name=name
        self.eta = eta
        self.optimizer = AdamWithGrad(eta,0.9,0.999)

    def set_state_dim(self,x_dim:int):
        self.x_dim=x_dim

    def set_ctrl_dim(self,u_dim:int):
        self.u_dim=u_dim

    def set_dyn(self,dyn_f:cd.Function):
        self.dyn_f=dyn_f

    def set_step_cost(self,step_cost:cd.Function):
        self.step_cost=step_cost

    def set_term_cost(self,term_cost:cd.Function):
        self.terminal_cost=term_cost

    def set_g(self,Features:cd.Function,gamma=0.001):
        self.features = Features
        self.gamma=gamma
        self.g_flag=True
        
    def from_controller(self,controller):
        self.set_state_dim(controller.x_dim)
        self.set_ctrl_dim(controller.u_dim)
        self.set_dyn(controller.dyn_f)
        self.set_step_cost(controller.step_cost)
        self.set_term_cost(controller.terminal_cost)
        assert hasattr(controller,'g_flag'),'no safety constraint in controller'
        self.set_g(controller.features,controller.gamma)

    def construct_graph(self,horizon):
        assert hasattr(self,'x_dim'), "missing x_dim"
        assert hasattr(self,'u_dim'), "missing u_dim"
        assert hasattr(self,'dyn_f'), "missing dyn_f"
        assert hasattr(self,'step_cost'), "missing step_cost"
        assert hasattr(self,'terminal_cost'), "terminal_cost"
        
        self.horizon=horizon
        init_state=cd.SX.sym('init_x',self.x_dim)
        weights=cd.SX.sym('weights',self.features.numel_out(0)-1)
        #pre-define lists for casadi solver, joint optimization of x,u
        u_mx_list=[]
        traj_xu=[]

        J=0

        #Xk = cd.MX.sym('X_0', self.x_dim) #Xk stands for the state in kth time step
        Xk = init_state
        traj_xu.append(Xk)

        for k in range(horizon): #at step k, construct u_k and x_{k+1}
            Uk=cd.SX.sym('U_' + str(k), self.u_dim) 
            #add u to var list
            u_mx_list.append(Uk)
            traj_xu.append(Uk)
            
            #dynamics
            Xk_1=self.dyn_f(Xk,Uk)
            traj_xu.append(Xk_1)
            #J
            Ck=self.step_cost(Xk,Uk)
            J=J+Ck 
            #update xk
            Xk=Xk_1
        #define trajectory
        traj_xu_flat=cd.vcat(traj_xu)
        traj_u_flat=cd.vcat(u_mx_list)
        #finish J
        J=J+self.terminal_cost(Xk_1)
        #Barrier
        B=J
        if hasattr(self,'g_flag'):
            expand_w=cd.vertcat(cd.DM([1]),weights)
            phi=self.features(traj_xu_flat)
            g_theta=cd.dot(expand_w,phi)
            B=B-self.gamma*cd.log(-g_theta)

        self.g_func=cd.Function('g_func',[weights,init_state,traj_u_flat],[g_theta])


        marginal=-cd.log(-g_theta) * self.gamma# type: ignore
        m_grad=cd.gradient(marginal,traj_u_flat)
        self.m_grad_func = cd.Function('u_grad',[weights,init_state,traj_u_flat],[m_grad])

        B_grad=cd.gradient(B,traj_u_flat)
        self.B_grad_func = cd.Function('u_grad',[weights,init_state,traj_u_flat],[B_grad])

        a=cd.SX.sym('human_corr',horizon*self.u_dim) # type: ignore
        #object: aligning the negative gradient of the margin to the correction
        obj = cd.sumsqr(a + B_grad)

        obj_grad = cd.gradient(obj,weights)
        self.obj_grad_func = cd.Function('weight_grad',[weights,init_state,traj_u_flat,a],[obj_grad])

    def init_optimizer(self,init_weights):
        self.optimizer.reset(init_weights)

    def step(self,weights,init_state,traj_xu,human_corr):
        traj_u=[]
        for i in range(self.horizon):
            start_idx=i*(self.x_dim+self.u_dim)+self.x_dim
            end_idx=start_idx+self.u_dim
            traj_u+=list(traj_xu[start_idx:end_idx])
        traj_u=np.array(traj_u)

        m_grad = self.m_grad_func(weights,init_state,traj_u)
        #print('m grad',m_grad)

        grad = self.obj_grad_func(weights,init_state,traj_u,human_corr).full().flatten()
        #print('grad',type(grad))

        #clip the gradient to avoid numerical issue
        grad = np.clip(grad,-0.01,0.01)
        #print('grad clip',grad)
        return self.optimizer.step(grad)

class gd_optimizer_nolog(gd_optimizer):
    def __init__(self, name: str, eta=0.01) -> None:
        super().__init__(name, eta)
    
    def construct_graph(self,horizon):
        assert hasattr(self,'x_dim'), "missing x_dim"
        assert hasattr(self,'u_dim'), "missing u_dim"
        assert hasattr(self,'dyn_f'), "missing dyn_f"
        assert hasattr(self,'step_cost'), "missing step_cost"
        assert hasattr(self,'terminal_cost'), "terminal_cost"
        
        self.horizon=horizon
        init_state=cd.SX.sym('init_x',self.x_dim)
        weights=cd.SX.sym('weights',self.features.numel_out(0)-1)
        #pre-define lists for casadi solver, joint optimization of x,u
        u_mx_list=[]
        traj_xu=[]

        J=0

        #Xk = cd.MX.sym('X_0', self.x_dim) #Xk stands for the state in kth time step
        Xk = init_state
        traj_xu.append(Xk)

        for k in range(horizon): #at step k, construct u_k and x_{k+1}
            Uk=cd.SX.sym('U_' + str(k), self.u_dim) 
            #add u to var list
            u_mx_list.append(Uk)
            traj_xu.append(Uk)
            
            #dynamics
            Xk_1=self.dyn_f(Xk,Uk)
            traj_xu.append(Xk_1)
            #J
            Ck=self.step_cost(Xk,Uk)
            J=J+Ck 
            #update xk
            Xk=Xk_1
        #define trajectory
        traj_xu_flat=cd.vcat(traj_xu)
        traj_u_flat=cd.vcat(u_mx_list)
        #finish J
        J=J+self.terminal_cost(Xk_1)
        #Barrier
        B=J
        if hasattr(self,'g_flag'):
            expand_w=cd.vertcat(cd.DM([1]),weights)
            phi=self.features(traj_xu_flat)
            g_theta=cd.dot(expand_w,phi)
            B=B-self.gamma*cd.log(-g_theta)

        self.g_func=cd.Function('g_func',[weights,init_state,traj_u_flat],[g_theta])


        marginal= g_theta 
        m_grad=cd.gradient(marginal,traj_u_flat)
        self.m_grad_func = cd.Function('u_grad',[weights,init_state,traj_u_flat],[m_grad])

        a=cd.SX.sym('human_corr',horizon*self.u_dim) # type: ignore
        #object: aligning the negative gradient of the margin to the correction
        obj = cd.sumsqr(a + m_grad)

        obj_grad = cd.gradient(obj,weights)
        self.obj_grad_func = cd.Function('weight_grad',[weights,init_state,traj_u_flat,a],[obj_grad])