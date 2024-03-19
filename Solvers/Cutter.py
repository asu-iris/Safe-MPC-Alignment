import numpy as np
import casadi as cd

class cutter(object):
    def __init__(self,name:str) -> None:
        self.name=name

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

    def set_g(self,Features:cd.Function,weights,gamma=0.001):
        self.features = Features
        self.weights=cd.SX(weights)
        self.gamma=gamma
        self.g_flag=True

    def construct_graph(self,horizon):
        assert hasattr(self,'x_dim'), "missing x_dim"
        assert hasattr(self,'u_dim'), "missing u_dim"
        assert hasattr(self,'dyn_f'), "missing dyn_f"
        assert hasattr(self,'step_cost'), "missing step_cost"
        assert hasattr(self,'terminal_cost'), "terminal_cost"
        
        self.init_state=cd.SX.sym('init_x',self.x_dim)
        self.horizon=horizon
        #pre-define lists for casadi solver, joint optimization of x,u
        self.u_mx_list=[]
        self.traj_xu=[]

        self.J=0

        #Xk = cd.MX.sym('X_0', self.x_dim) #Xk stands for the state in kth time step
        Xk = self.init_state
        self.traj_xu.append(Xk)

        for k in range(horizon): #at step k, construct u_k and x_{k+1}
            Uk=cd.SX.sym('U_' + str(k), self.u_dim)
            #add u to var list
            self.u_mx_list.append(Uk)
            self.traj_xu.append(Uk)
            
            #dynamics
            Xk_1=self.dyn_f(Xk,Uk)
            self.traj_xu.append(Xk_1)
            #J
            Ck=self.step_cost(Xk,Uk)
            self.J=self.J+Ck
            #update xk
            Xk=Xk_1
        #define trajectory
        self.traj_xu_flat=cd.vertcat(*self.traj_xu)
        self.traj_u_flat=cd.vertcat(*self.u_mx_list)
        #finish J
        self.J=self.J+self.terminal_cost(Xk_1)
        #Barrier
        self.B=self.J
        if hasattr(self,'g_flag'):
            expand_w=cd.vertcat(cd.DM([1]),self.weights)
            self.phi=self.features(self.traj_xu_flat)
            self.g_theta=cd.dot(expand_w,self.phi)
            self.B=self.B-self.gamma*cd.log(-self.g_theta)

        self.g_func=cd.Function('g_func',[self.init_state, self.traj_u_flat],[self.g_theta])

        self.B_grad=cd.gradient(self.B,self.traj_u_flat)
        self.B_grad_func=cd.Function('B_grad',[self.init_state, self.traj_u_flat],[self.B_grad])

        self.J_grad=cd.gradient(self.J,self.traj_u_flat)
        self.J_grad_func=cd.Function('J_grad',[self.init_state, self.traj_u_flat],[self.J_grad])

        self.marginal=-self.gamma*cd.log(-self.g_theta)
        self.m_grad=cd.gradient(self.marginal,self.traj_u_flat)
        self.m_grad_func=cd.Function('m_grad',[self.init_state, self.traj_u_flat],[self.m_grad])

        self.phi_Jacobi=cd.jacobian(self.phi,self.traj_u_flat)
        self.phi_Jacobi_func=cd.Function('phi_jacobi',[self.init_state, self.traj_u_flat],[self.phi_Jacobi])

    def calc_planes(self,init_state,traj_xu,human_corr):
        self.traj_u=[]
        for i in range(self.horizon):
            start_idx=i*(self.x_dim+self.u_dim)+self.x_dim
            end_idx=start_idx+self.u_dim
            self.traj_u+=list(traj_xu[start_idx:end_idx])
        self.traj_u=np.array(self.traj_u)

        phi=self.features(traj_xu)
        #print('J grad', self.J_grad_func(init_state,self.traj_u))
        #print('M grad', self.m_grad_func(init_state,self.traj_u))
        # theta^T h <= b
        h=-np.dot(human_corr.T ,self.J_grad_func(init_state,self.traj_u))*phi[1:]
        phi_jacobi=self.phi_Jacobi_func(init_state,self.traj_u)
        phi_jacobi_1_=phi_jacobi[1:,:]
        h=h+self.gamma*(phi_jacobi_1_ @ human_corr)

        b=np.dot(human_corr.T ,self.J_grad_func(init_state,self.traj_u))*phi[0]
        phi_grad_0=phi_jacobi[0,:]
        b=b-self.gamma*(phi_grad_0 @ human_corr)

        # theta^T h_phi <= b_phi
        h_phi=phi[1:]
        b_phi=-phi[0]

        return h,b,h_phi,b_phi
    
    def calc_m_grad(self,init_state,traj_xu):
        self.traj_u=[]
        for i in range(self.horizon):
            start_idx=i*(self.x_dim+self.u_dim)+self.x_dim
            end_idx=start_idx+self.u_dim
            self.traj_u+=list(traj_xu[start_idx:end_idx])
        self.traj_u=np.array(self.traj_u)
        return self.m_grad_func(init_state,self.traj_u)

class cutter_v2(object):
    def __init__(self,name:str) -> None:
        self.name=name

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

        B_grad=cd.gradient(B,traj_u_flat)
        self.B_grad_func=cd.Function('B_grad',[weights,init_state,traj_u_flat],[B_grad])

        J_grad=cd.gradient(J,traj_u_flat)
        self.J_grad_func=cd.Function('J_grad',[weights,init_state, traj_u_flat],[J_grad])

        marginal=-self.gamma*cd.log(-g_theta)
        m_grad=cd.gradient(marginal,traj_u_flat)
        self.m_grad_func=cd.Function('m_grad',[weights,init_state, traj_u_flat],[m_grad])

        phi_Jacobi=cd.jacobian(phi,traj_u_flat)
        self.phi_Jacobi_func=cd.Function('phi_jacobi',[init_state,traj_u_flat],[phi_Jacobi])

    def calc_planes(self,weights,init_state,traj_xu,human_corr):
        traj_u=[]
        for i in range(self.horizon):
            start_idx=i*(self.x_dim+self.u_dim)+self.x_dim
            end_idx=start_idx+self.u_dim
            traj_u+=list(traj_xu[start_idx:end_idx])
        traj_u=np.array(traj_u)

        phi=self.features(traj_xu)
        #print('J grad', self.J_grad_func(init_state,self.traj_u))
        #print('M grad', self.m_grad_func(init_state,self.traj_u))
        # theta^T h <= b
        h=-np.dot(human_corr.T ,self.J_grad_func(weights,init_state,traj_u))*phi[1:]
        phi_jacobi=self.phi_Jacobi_func(init_state,traj_u)
        phi_jacobi_1_=phi_jacobi[1:,:]
        h=h+self.gamma*(phi_jacobi_1_ @ human_corr)

        b=np.dot(human_corr.T ,self.J_grad_func(weights,init_state,traj_u))*phi[0]
        phi_grad_0=phi_jacobi[0,:]
        b=b-self.gamma*(phi_grad_0 @ human_corr)

        # theta^T h_phi <= b_phi
        h_phi=phi[1:]
        b_phi=-phi[0]

        return h,b,h_phi,b_phi
    
