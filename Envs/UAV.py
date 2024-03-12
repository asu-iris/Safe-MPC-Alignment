import numpy as np
from quaternion import as_rotation_matrix,from_rotation_matrix,as_float_array
import casadi as cd

class UAV_env(object):
    def __init__(self,gravity,m,J_B,l_w,dt,c) -> None:
        self.g=gravity
        self.g_I=np.reshape(np.array([0,0,-self.g]),(-1,1))
        self.m=m
        self.J_B=J_B
        self.l_w=l_w
        self.dt=dt

        self.K_tau = np.array([[0,-self.l_w/2,0,-self.l_w/2],
                               [-self.l_w/2,0,self.l_w/2,0],
                               [c,-c,c,-c]])

    def set_init_state(self,x:np.ndarray):
        self.x_0=np.array(x)

        self.clear_traj()
        self.x_traj.append(self.x_0)

        self.curr_x=self.x_0

    def get_curr_state(self):
        return self.curr_x
    
    def clear_traj(self):
        self.x_traj=[]
        self.u_traj=[]

    def step(self,u):# u: T_1, T_2, T_3, T_4
        self.r_I=self.curr_x[0:3]
        self.v_I=self.curr_x[3:6]
        self.q_BI=self.curr_x[6:10]
        self.w_B=self.curr_x[10:]

        self.R_B_I = Quat_Rot(self.q_BI.flatten())#rotation matrix: inertial to body
        self.R_I_B = self.R_B_I.T #rotation matrix: body to inertial

        thrust= u.T @ np.ones((4,1))
        print(thrust)
        f_I = self.R_I_B @ np.reshape(np.concatenate([np.zeros((2,1)),thrust]),(-1,1))
        d_r_I = self.v_I
        d_v_I = self.g_I + f_I/self.m
        d_q = 0.5 * self.Omega(self.w_B.flatten()) @ self.q_BI
        
        d_w_B = np.linalg.inv(self.J_B) @ (self.K_tau @ u - np.reshape(np.cross(self.w_B.flatten(), (self.J_B @ self.w_B).flatten()),(-1,1)))

        self.r_I += self.dt * d_r_I
        self.v_I += self.dt * d_v_I
        self.q_BI += self.dt * d_q
        self.w_B += self.dt * d_w_B

        self.curr_x=np.concatenate([self.r_I,self.v_I,self.q_BI,self.w_B],axis=0)
        self.x_traj.append(self.curr_x)

    def get_state(self):
        return self.curr_x
    
    def get_pos(self):
        return self.curr_x[0:3]

    def Omega(self, w):
        Omeg = np.array([[0, -w[0], -w[1], -w[2]],
                         [w[0], 0, w[2], -w[1]],
                         [w[1], -w[2], 0, w[0]],
                         [w[2], w[1], -w[0], 0]])
        return Omeg

def Quat_Rot(q):
    Rot = cd.vertcat(
            cd.horzcat(1 - 2 * (q[2] ** 2 + q[3] ** 2), 2 * (q[1] * q[2] + q[0] * q[3]), 2 * (q[1] * q[3] - q[0] * q[2])),
            cd.horzcat(2 * (q[1] * q[2] - q[0] * q[3]), 1 - 2 * (q[1] ** 2 + q[3] ** 2), 2 * (q[2] * q[3] + q[0] * q[1])),
            cd.horzcat(2 * (q[1] * q[3] + q[0] * q[2]), 2 * (q[2] * q[3] - q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2))
        )
    return Rot

if __name__ == '__main__':
    init_r = np.zeros((3,1))
    init_v = np.zeros((3,1))
    init_q = np.reshape(as_float_array(from_rotation_matrix(np.eye(3))),(-1,1))
    #print(Quat_Rot(init_q))
    init_w_B = np.zeros((3,1))
    init_x=np.concatenate([init_r,init_v,init_q,init_w_B],axis=0)
    #print(init_x)
    uav_env=UAV_env(10,1,np.eye(3),1,0.05,1)
    uav_env.set_init_state(init_x)
    u=0*np.ones((4,1))

    for i in range(10):
        uav_env.step(u)
        print(uav_env.get_state())
