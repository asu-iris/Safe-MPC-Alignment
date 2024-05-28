import casadi as cd
import numpy as np
import sys,os
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
sys.path.append(os.path.abspath(os.getcwd()))
from Envs.UAV import UAV_env,Quat_Rot
class Correction_Agent(object):
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
    def set_p(self,p=1):
        self.p=p
    def set_threshold(self,thresh=-0.15):
        self.corr_thresh=thresh

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
            phi=self.features(self.traj_xu_flat)
            self.g_theta=cd.dot(expand_w,phi)
            self.B=self.B-self.gamma*cd.log(-self.g_theta)

        self.g_func=cd.Function('g_func',[self.init_state, self.traj_u_flat],[self.g_theta])

        self.B_grad=cd.gradient(self.B,self.traj_u_flat)
        self.B_grad_func=cd.Function('B_grad',[self.init_state, self.traj_u_flat],[self.B_grad])

        self.J_grad=cd.gradient(self.J,self.traj_u_flat)
        self.J_grad_func=cd.Function('J_grad',[self.init_state, self.traj_u_flat],[self.J_grad])

        self.marginal=-self.gamma*cd.log(-self.g_theta)
        self.m_grad=cd.gradient(self.marginal,self.traj_u_flat)
        self.m_grad_func=cd.Function('m_grad',[self.init_state, self.traj_u_flat],[self.m_grad])

    def correct(self,traj_xu):
        init_state=np.array(traj_xu[0:self.x_dim])
        #print('grad B', self.B_grad_func(init_state,self.traj_u))
        #print('grad J', self.J_grad_func(init_state,self.traj_u))
        #print('grad m', self.m_grad_func(init_state,self.traj_u))
        return -self.B_grad_func(init_state,self.traj_u)
    
    def act(self,traj_xu):
        init_state=np.array(traj_xu[0:self.x_dim])
        self.traj_u=[]
        for i in range(self.horizon):
            start_idx=i*(self.x_dim+self.u_dim)+self.x_dim
            end_idx=start_idx+self.u_dim
            self.traj_u+=list(traj_xu[start_idx:end_idx])
        self.traj_u=np.array(self.traj_u)
        #print('g human',self.g_func(init_state,self.traj_u))
        #print('ln -g human',np.log(-self.g_func(init_state,self.traj_u)))
        if self.g_func(init_state,self.traj_u) >=0:
            return None
        elif self.g_func(init_state,self.traj_u) <= self.corr_thresh:
            return True
        else:
            if np.random.uniform(0,1)<self.p:
                #print('agent correction')
                return self.correct(traj_xu)
            else:
                #print('near boundary')
                return True
            
def uav_trans(world_corr,env:UAV_env):#world corr can be [+-1,0,0] [0,+-1,0], output is 4D correction in u
    R_B_I=np.array(Quat_Rot(env.curr_x[6:10])).T #from world to body
    x_corr_b = R_B_I @ world_corr.reshape(-1,1)
    x_corr_b=x_corr_b.flatten()
    x_dir_thrust=np.array([-1,0,1,0]).reshape(-1,1)
    y_dir_thrust=np.array([0,1,0,-1]).reshape(-1,1)
    z_dir_thrust=np.array([1,1,1,1]).reshape(-1,1)
    u_x = x_corr_b[0] * x_dir_thrust + x_corr_b[1] * y_dir_thrust +x_corr_b[2] * z_dir_thrust
    return 0.1*u_x

        