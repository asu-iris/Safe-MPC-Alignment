import numpy as np
import casadi as cd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R

class UAV_env(object):
    def __init__(self,gravity,m,J_B,l_w,dt,c) -> None:
        self.g=gravity
        self.g_I=np.reshape(np.array([0,0,-self.g]),(-1,1))
        self.m=m
        self.J_B=J_B
        self.l_w=l_w
        self.dt=dt

        self.K_tau = np.array([[0,-self.l_w/2,0,self.l_w/2],
                               [-self.l_w/2,0,self.l_w/2,0],
                               [c,-c,c,-c]])

    def set_init_state(self,x:np.ndarray):
        self.x_0=np.array(x)

        self.clear_traj()
        self.x_traj.append(self.x_0.flatten())

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
        print(f_I)
        d_r_I = self.v_I
        d_v_I = self.g_I + f_I/self.m
        d_q = 0.5 * Omega(self.w_B.flatten()) @ self.q_BI
        
        d_w_B = np.linalg.inv(self.J_B) @ (self.K_tau @ u - np.reshape(np.cross(self.w_B.flatten(), (self.J_B @ self.w_B).flatten()),(-1,1)))
        self.r_I += self.dt * d_r_I
        self.v_I += self.dt * d_v_I
        self.q_BI += self.dt * d_q
        self.w_B += self.dt * d_w_B

        self.curr_x=np.concatenate([self.r_I,self.v_I,self.q_BI,self.w_B],axis=0)
        self.x_traj.append(self.curr_x.flatten())

    def get_state(self):
        return self.curr_x
    
    def get_pos(self):
        return self.curr_x[0:3]
    
    def show_animation(self,flag_2d=False):
        def draw_quadrotor(ax_3d, ax_2d, pos, quat, wing_length):
            # Extracting position and attitude information
            x, y, z = pos
            q0, q1, q2, q3 = quat
            
            # Defining quadrotor wings tips (IN BODY FRAME)
            #wing1_tip = np.array([x-wing_length, y, z])
            #wing2_tip = np.array([x+wing_length, y, z])
            #wing3_tip = np.array([x, y-wing_length, z])
            #wing4_tip = np.array([x, y+wing_length, z])

            wing1_tip = np.array([+wing_length, 0, 0])
            wing2_tip = np.array([0, +wing_length, 0])
            wing3_tip = np.array([-wing_length, 0, 0])
            wing4_tip = np.array([0, -wing_length, 0])
            
            # Rotate wing tips based on quaternion
            rot_I_B = R.from_quat([q1, q2, q3, q0]).as_matrix().T
            wing1_tip = rot_I_B @ wing1_tip +pos
            wing2_tip = rot_I_B @ wing2_tip +pos
            wing3_tip = rot_I_B @ wing3_tip +pos
            wing4_tip = rot_I_B @ wing4_tip +pos
            
            # Plotting quadrotor wings
            ax_3d.scatter(x, y, z, color='black', marker='o')
            ax_3d.scatter(wing1_tip[0], wing1_tip[1], wing1_tip[2], color='r', marker='o')
            ax_3d.plot((x,wing1_tip[0]),(y, wing1_tip[1]), (z,wing1_tip[2]), color='r')

            ax_3d.scatter(wing2_tip[0], wing2_tip[1], wing2_tip[2], color='b', marker='o')
            ax_3d.plot((x,wing2_tip[0]),(y, wing2_tip[1]), (z,wing2_tip[2]), color='b')

            ax_3d.scatter(wing3_tip[0], wing3_tip[1], wing3_tip[2], color='r', marker='o')
            ax_3d.plot((x,wing3_tip[0]),(y, wing3_tip[1]), (z,wing3_tip[2]), color='r')

            ax_3d.scatter(wing4_tip[0], wing4_tip[1], wing4_tip[2], color='b', marker='o')
            ax_3d.plot((x,wing4_tip[0]),(y, wing4_tip[1]), (z,wing4_tip[2]), color='b')
            if flag_2d:
            # 2d projection
                ax_2d.scatter(x, y, color='black', marker='o')
                ax_2d.scatter(wing1_tip[0], wing1_tip[1], color='r', marker='o')
                ax_2d.plot((x,wing1_tip[0]),(y, wing1_tip[1]), color='r')

                ax_2d.scatter(wing2_tip[0], wing2_tip[1], color='r', marker='o')
                ax_2d.plot((x,wing2_tip[0]),(y, wing2_tip[1]), color='r')

                ax_2d.scatter(wing3_tip[0], wing3_tip[1], color='b', marker='o')
                ax_2d.plot((x,wing3_tip[0]),(y, wing3_tip[1]), color='b')

                ax_2d.scatter(wing4_tip[0], wing4_tip[1], color='b', marker='o')
                ax_2d.plot((x,wing4_tip[0]),(y, wing4_tip[1]), color='b')

            return
        
        fig = plt.figure()
        if flag_2d:
            ax_3d = fig.add_subplot(121, projection='3d')
            ax_2d = fig.add_subplot(122)

        else:
            ax_3d = fig.add_subplot(111, projection='3d')
            ax_2d = None
        positions=np.array(self.x_traj)[:,0:3]
        quaternions=np.array(self.x_traj)[:,6:10]

        def update(frame):
            ax_3d.clear()
            if flag_2d:
                ax_2d.clear()
            draw_quadrotor(ax_3d, ax_2d, positions[frame], quaternions[frame], self.l_w)
            ax_3d.set_xlim([-1, 8])
            ax_3d.set_ylim([-1, 8])
            ax_3d.set_zlim([-1, 8])
            ax_3d.set_xlabel('X')
            ax_3d.set_ylabel('Y')
            ax_3d.set_zlabel('Z')
            ax_3d.set_title('Quadrotor Trajectory')
            if flag_2d:
                ax_2d.set_xlim([-1, 8])
                ax_2d.set_ylim([-1, 8])
                ax_2d.set_xlabel('X')
                ax_2d.set_ylabel('Y')
                ax_2d.set_title('Quadrotor Trajectory (Top View)')
            return

        # Animate the quadrotor trajectory
        ani = FuncAnimation(fig, update, frames=len(positions), interval=1000*self.dt)
        plt.show()

def Quat_Rot(q):
    Rot = cd.vertcat(
            cd.horzcat(1 - 2 * (q[2] ** 2 + q[3] ** 2), 2 * (q[1] * q[2] + q[0] * q[3]), 2 * (q[1] * q[3] - q[0] * q[2])),
            cd.horzcat(2 * (q[1] * q[2] - q[0] * q[3]), 1 - 2 * (q[1] ** 2 + q[3] ** 2), 2 * (q[2] * q[3] + q[0] * q[1])),
            cd.horzcat(2 * (q[1] * q[3] + q[0] * q[2]), 2 * (q[2] * q[3] - q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2))
        )
    return Rot

def Omega(w):
    Omeg = cd.vertcat(
        cd.horzcat(0, -w[0], -w[1], -w[2]),
        cd.horzcat(w[0], 0, w[2], -w[1]),
        cd.horzcat(w[1], -w[2], 0, w[0]),
        cd.horzcat(w[2], w[1], -w[0], 0),
    )
    return Omeg

if __name__ == '__main__':
    init_r = np.zeros((3,1))
    init_v = np.zeros((3,1))
    init_q = np.reshape(np.array([1,0,0,0]),(-1,1))
    #print(Quat_Rot(init_q))
    init_w_B = np.zeros((3,1))
    init_x=np.concatenate([init_r,init_v,init_q,init_w_B],axis=0)
    #print(init_x)
    uav_env=UAV_env(10,1,np.eye(3),0.5,0.05,1)
    uav_env.set_init_state(init_x)

    u=2.6*np.ones((4,1))
    u[1]+=0.1
    u[3]-=0.1

    for i in range(15):
        uav_env.step(u)
        print(uav_env.get_state().T)

    #uav_env.show_animation(flag_2d=False)
