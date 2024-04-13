import numpy as np
import casadi as cd

import mujoco
import mujoco.viewer

import os
import time

class Robot_Arm_model(object):
    def __init__(self,dt) -> None:
        self.dt=dt

    def get_dyn_f(self):
        x_t=cd.SX.sym('x_t',7)
        u_t=cd.SX.sym('u_t',7)
        x_t_1= self.dt*u_t + x_t
        return cd.Function('arm_dynamics', [x_t, u_t], [x_t_1])
    
    def get_step_cost_param(self, param_vec: np.ndarray): #param:[kx,ky,kz,ku]
        q=cd.SX.sym('q',7) # theta angles
        u=cd.SX.sym('u',7) #speed control
        target_end_pos=cd.SX.sym('target',3)

        #print(current_DHForm.shape)
        current_end_pos=(DH_to_Mat(q) @ cd.DM(np.array([0,0,0,1])))[0:3]
        l_vec=cd.vertcat((current_end_pos-target_end_pos)**2,cd.sumsqr(u))
        p_vec = cd.DM(param_vec)
        cost=p_vec.T @ l_vec
        #print(cost)
        return cd.Function('step_cost', [q, u, target_end_pos], [cost])
    
    def get_terminal_cost_param(self, param_vec: np.ndarray): #param:[kx,ky,kz]
        q=cd.SX.sym('q',7) # theta angles
        target_end_pos=cd.SX.sym('target',3)

        current_end_pos=(DH_to_Mat(q) @ cd.DM(np.array([0,0,0,1])))[0:3]
        l_vec=(current_end_pos-target_end_pos)**2
        p_vec = cd.DM(param_vec)
        cost=p_vec.T @ l_vec

        return cd.Function('step_cost', [q, target_end_pos], [cost])
    
    def calc_end_pos(self,q):
        q=cd.DM(q)
        current_end_pos=(DH_to_Mat(q) @ cd.DM(np.array([0,0,0,1])))[0:3]
        return current_end_pos.full()

    
class ARM_env_mj(object):
    def __init__(self, xml_path) -> None:
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        ini_joint=np.zeros(8)
        ini_joint[0]=0
        ini_joint[3]=-1.5
        ini_joint[5]=1.5
        self.set_init_state_v(ini_joint)


    def set_init_state(self, x: np.ndarray):
        mujoco.mj_resetData(self.model, self.data)
        x = x.flatten()
        self.data.qpos[0:7] = x[0:7]
        self.data.ctrl = x
        for i in range(100):
            mujoco.mj_step(self.model, self.data)

    def set_init_state_v(self, x: np.ndarray):
        mujoco.mj_resetData(self.model, self.data)
        x = x.flatten()
        self.data.qpos[0:7] = x[0:7]
        self.data.ctrl = np.zeros(8)
        for i in range(100):
            mujoco.mj_step(self.model, self.data)

    def step(self, u, dt):  # u:speed of joints
        loop_num=int(dt/0.002)
        inner_ctrl = self.data.qpos[0:7] + u*dt
        print('ctrl',self.data.ctrl)
        self.data.ctrl[0:7] = inner_ctrl.flatten()
        self.data.ctrl[7] = 0
        for i in range(loop_num):
            mujoco.mj_step(self.model, self.data)
            #print(self.data.qpos[1])

    def step_vel(self, u, dt):  # u:speed of joints
        loop_num=int(dt/0.002)
        #print('ctrl',self.data.ctrl)
        self.data.ctrl[0:7] = u.flatten()
        self.data.ctrl[7] = 0
        for i in range(loop_num):
            mujoco.mj_step(self.model, self.data)
            #print(self.data.qpos[1])

    def get_curr_state(self):
        return self.data.qpos[0:7].copy()

def Rot_x(alpha):
    return cd.vertcat(
        cd.horzcat(1,0,0,0),
        cd.horzcat(0,cd.cos(alpha),-cd.sin(alpha),0),
        cd.horzcat(0,cd.sin(alpha),cd.cos(alpha),0),
        cd.horzcat(0,0,0,1),
    )

def Rot_y(beta):
    return cd.vertcat(
        cd.horzcat(cd.cos(beta),0,cd.sin(beta),0),
        cd.horzcat(0,1,0,0),
        cd.horzcat(-cd.sin(beta),0,cd.cos(beta),0),
        cd.horzcat(0,0,0,1),
    )

def Rot_z(theta):
    return cd.vertcat(
        cd.horzcat(cd.cos(theta),-cd.sin(theta),0,0),
        cd.horzcat(cd.sin(theta),cd.cos(theta),0,0),
        cd.horzcat(0,0,1,0),
        cd.horzcat(0,0,0,1),
    )

def Trans_x(a):
    return cd.vertcat(
        cd.horzcat(1,0,0,a),
        cd.horzcat(0,1,0,0),
        cd.horzcat(0,0,1,0),
        cd.horzcat(0,0,0,1),
    )

def Trans_y(b):
    return cd.vertcat(
        cd.horzcat(1,0,0,0),
        cd.horzcat(0,1,0,b),
        cd.horzcat(0,0,1,0),
        cd.horzcat(0,0,0,1),
    )

def Trans_z(d):
    return cd.vertcat(
        cd.horzcat(1,0,0,0),
        cd.horzcat(0,1,0,0),
        cd.horzcat(0,0,1,d),
        cd.horzcat(0,0,0,1),
    )

def DH_to_Mat(q):
    Sub_Mats=[]
    E_1=Trans_z(0.333)
    Sub_Mats.append(E_1)
    E_2=Rot_z(q[0])
    Sub_Mats.append(E_2)
    E_3=Rot_y(q[1])
    Sub_Mats.append(E_3)
    E_4=Trans_z(0.316)
    Sub_Mats.append(E_4)
    E_5=Rot_z(q[2])
    Sub_Mats.append(E_5)
    E_6=Trans_x(0.0825)
    Sub_Mats.append(E_6)
    E_7=Rot_y(-q[3])
    Sub_Mats.append(E_7)
    E_8=Trans_x(-0.0825)
    Sub_Mats.append(E_8)
    E_9=Trans_z(0.384)
    Sub_Mats.append(E_9)
    E_10=Rot_z(q[4])
    Sub_Mats.append(E_10)
    E_11=Rot_y(-q[5])
    Sub_Mats.append(E_11)
    E_12=Trans_x(0.088)
    Sub_Mats.append(E_12)
    E_13=Rot_x(cd.pi)
    Sub_Mats.append(E_13)
    E_14=Trans_z(0.107)
    Sub_Mats.append(E_14)
    E_15=Rot_z(q[6])
    Sub_Mats.append(E_15)

    T=cd.DM(np.eye(4))
    for mat in Sub_Mats:
        T=T@mat
    return T

if __name__=='__main__':
    test_q=np.zeros(7)
    #viewer=mujoco.viewer.launch()
    x=np.array([0,0,0,1]).reshape(-1,1)
    print(DH_to_Mat(test_q) @ x)
    #q=DHForm_to_Mat(test_DHForm) @ np.array([0,0,0,1]).reshape(-1,1)
    #print(q)
    filepath = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                        'mujoco_arm', 'franka_emika_panda',
                        'scene.xml')
    env=ARM_env_mj(filepath)
    #viewer=mujoco.viewer.launch(env.model,env.data)
    #exit()
    #env=ARM_env_mj(filepath)
    viewer=mujoco.viewer.launch_passive(env.model,env.data)
    print(env.get_curr_state())
    for i in range(100):
        env.step_vel(0.0*np.ones(7),0.1)
        print(env.get_curr_state())
        viewer.sync()
        time.sleep(0.1)