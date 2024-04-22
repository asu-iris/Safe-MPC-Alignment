import numpy as np
import casadi as cd

import mujoco
import mujoco.viewer

import os
import time
from scipy.spatial.transform import Rotation as R

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

class End_Effector_model(object):
    def __init__(self,dt) -> None:
        self.dt=dt

    def get_dyn_f(self):
        x_t=cd.SX.sym('x_t',7)
        r_t=x_t[0:3]
        q_t=x_t[3:7] # from end to the world!
        u_t=cd.SX.sym('u_t',6) #[v,w]
        v_t=u_t[0:3]
        w_t=u_t[3:6] #rotation in world frame

        r_t_1= self.dt*v_t + r_t

        d_q = 0.5 * Omega(Quat_Rot(q_t).T @ w_t) @ q_t
        q_t_1 = q_t + d_q * self.dt

        x_t_1=cd.vertcat(r_t_1,q_t_1)
        return cd.Function('arm_dynamics', [x_t, u_t], [x_t_1])
    
    def get_step_cost_param(self, param_vec: np.ndarray): #param:[kr,kq,kv,kw]
        x=cd.SX.sym('x',7) #[r,q]
        u=cd.SX.sym('u',6) #speed control

        target_x=cd.SX.sym('target_x',7)
        target_r=target_x[0:3]
        target_q=target_x[3:7]

        l_vec=cd.vertcat(cd.sumsqr(x[0:3]-target_r),q_dist(x[3:7],target_q),cd.sumsqr(u[0:3]),cd.sumsqr(u[3:]))

        p_vec = cd.DM(param_vec)
        cost=p_vec.T @ l_vec
        #print(cost)
        return cd.Function('step_cost', [x, u, target_x], [cost])
    
    def get_terminal_cost_param(self, param_vec: np.ndarray): #param:[kr,kq]
        x=cd.SX.sym('x',7) #[r,q]

        target_x=cd.SX.sym('target_x',7)
        target_r=target_x[0:3]
        target_q=target_x[3:7]

        l_vec=cd.vertcat(cd.sumsqr(x[0:3]-target_r),q_dist(x[3:7],target_q))

        p_vec = cd.DM(param_vec)
        cost=p_vec.T @ l_vec

        return cd.Function('term_cost', [x, target_x], [cost])
        
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

class EFFECTOR_env_mj(object):
    def __init__(self, xml_path, dt) -> None:
        self.dt=dt
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.last_ang_vel=np.zeros((7,1))

        # ini_joint=np.zeros(8)
        # ini_joint[0]=-1.8
        # ini_joint[1]= 0.405
        # ini_joint[2]=-0.348
        # ini_joint[3]=-1.39
        # ini_joint[4]=-1.65
        # ini_joint[5]=2.15
        # ini_joint[6]=-0.55
        # ini_joint=np.array([-1.58753662,-0.31941105,-0.81050407,-2.28855788,-2.17938154,2.16288644,-0.20836714,0])
        self.ini_joint=np.array([-1.76443251,0.52896963,-0.76707975,-1.50648771,-1.63760893 ,2.51745144,-0.26245633,0])
        #ini_joint=np.array([-1.40675402,0.08373927,-1.24319759,-1.97988368,-2.18859521,2.52699744,0.01613408,0])
        self.set_init_state_v(self.ini_joint)

    def reset_env(self):
        self.set_init_state_v(self.ini_joint)

    def set_init_state_v(self, x: np.ndarray):
        mujoco.mj_resetData(self.model, self.data)
        x = x.flatten()
        self.data.qpos[0:7] = x[0:7]
        self.data.ctrl = np.zeros(8)
        for i in range(100):
            mujoco.mj_step(self.model, self.data)

    def step(self, u):  # u:speed in end effector
        u=np.array(u)
        loop_num=int(self.dt/0.002)

        site_id=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_SITE,'flange')
        jac_p=np.zeros((3,9))
        jac_r=np.zeros((3,9))
        mujoco.mj_jacSite(self.model,self.data,jac_p,jac_r,site_id)
        Jac=np.concatenate([jac_p[:,0:7],jac_r[:,0:7]])
        #print('shape',Jac.shape)
        J_Inv=np.linalg.pinv(Jac)
        inner_ctrl = J_Inv @ u.reshape(-1,1) #+ (np.eye(7)-J_Inv @ Jac) @ self.last_ang_vel
        self.last_ang_vel = inner_ctrl 
        #print('ctrl',inner_ctrl)
        self.data.ctrl[0:7] = inner_ctrl.flatten()
        self.data.ctrl[7] = 0
        for i in range(loop_num):
            mujoco.mj_step(self.model, self.data)
            #print(self.data.qpos[1])

    def get_curr_joints(self):
        return self.data.qpos[0:7].copy()
    
    def get_site_pos(self):
        site_id=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_SITE,'flange')
        return self.data.site_xpos[site_id]
    
    def get_site_vel(self):
        site_id=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_SITE,'flange')
        jac_p=np.zeros((3,9))
        jac_r=np.zeros((3,9))
        mujoco.mj_jacSite(self.model,self.data,jac_p,jac_r,site_id)
        Jac=np.concatenate([jac_p[:,0:7],jac_r[:,0:7]])
        return Jac @ self.data.qvel[0:7]
    
    def get_site_quat(self):
        site_id=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_SITE,'flange')
        site_quat=np.zeros(4)
        mujoco.mju_mat2Quat(site_quat,self.data.site_xmat[site_id])
        return site_quat
    
    def get_hand_quat(self):
        hand_id=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,'hand')
        #print(hand_id)
        #input()
        hand_quat=self.data.xquat[hand_id]
        return hand_quat
    
    def get_curr_state(self):
        pos=self.get_site_pos().reshape(-1,1)
        quat=self.get_site_quat().reshape(-1,1)
        x=np.concatenate((pos,quat),axis=0)
        return x
    
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

def Quat_Rot(q):
    Rot = cd.vertcat(
        cd.horzcat(1 - 2 * (q[2] ** 2 + q[3] ** 2), 2 * (q[1] * q[2] - q[0] * q[3]), 2 * (q[1] * q[3] + q[0] * q[2])),
        cd.horzcat(2 * (q[1] * q[2] + q[0] * q[3]), 1 - 2 * (q[1] ** 2 + q[3] ** 2), 2 * (q[2] * q[3] - q[0] * q[1])),
        cd.horzcat(2 * (q[1] * q[3] - q[0] * q[2]), 2 * (q[2] * q[3] + q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2))
    )
    return Rot

def Rot_Quat(R):
    eta=0.5*cd.sqrt(cd.trace(R)+1)
    eps_0=0.5*cd.sign(R[2,1]-R[1,2])*cd.sqrt(R[0,0]-R[1,1]-R[2,2]+1)
    eps_1=0.5*cd.sign(R[0,2]-R[2,0])*cd.sqrt(R[1,1]-R[2,2]-R[0,0]+1)
    eps_2=0.5*cd.sign(R[1,0]-R[0,1])*cd.sqrt(R[2,2]-R[0,0]-R[1,1]+1)
    return cd.vertcat(eta,eps_0,eps_1,eps_2)

def Quat_mul(q_1,q_2):
    eta_1=q_1[0]
    eta_2=q_2[0]
    eps_1=q_1[1:]
    eps_2=q_2[1:]

    return cd.vertcat(eta_1*eta_2-eps_1.T @ eps_2,
                      eta_1*eps_2+eta_2*eps_1 + cd.cross(eps_1,eps_2))

def Angle_Axis_Rot(theta,axis):
    r_x=axis[0]
    r_y=axis[1]
    r_z=axis[2]
    Rot = cd.vertcat(
        cd.horzcat(r_x**2*(1-cd.cos(theta))+cd.cos(theta), r_x*r_y*(1-cd.cos(theta))-r_z*cd.sin(theta), r_x*r_z*(1-cd.cos(theta))+r_y*cd.sin(theta)),
        cd.horzcat(r_x*r_y*(1-cd.cos(theta))+r_z*cd.sin(theta), r_y**2*(1-cd.cos(theta))+cd.cos(theta), r_y*r_z*(1-cd.cos(theta))-r_x*cd.sin(theta)),
        cd.horzcat(r_x*r_z*(1-cd.cos(theta))-r_y*cd.sin(theta), r_y*r_z*(1-cd.cos(theta))+r_x*cd.sin(theta), r_z**2*(1-cd.cos(theta))+cd.cos(theta))
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

def q_dist(q_1, q_2):
    I = cd.DM(np.eye(3))
    return 0.5 * cd.trace(I - Quat_Rot(q_2).T @ Quat_Rot(q_1))
    return 1 - (q_1.T @ q_2)**2

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
    #env=ARM_env_mj(filepath)
    env=EFFECTOR_env_mj(filepath)
    #viewer=mujoco.viewer.launch(env.model,env.data)
    #exit()
    #env=ARM_env_mj(filepath)
    viewer=mujoco.viewer.launch_passive(env.model,env.data)
    print(env.get_curr_joints())
    id=mujoco.mj_name2id(env.model,mujoco.mjtObj.mjOBJ_SITE,'flange')
    jac_p=np.zeros((3,9))
    jac_r=np.zeros((3,9))
    mujoco.mj_jacSite(env.model,env.data,jac_p,jac_r,id)
    print(jac_p)
    print(jac_r)
    for i in range(100):
        ctrl=np.zeros(6)
        ctrl[2]=-0.01
        ctrl[5]=-0.4
        env.step(ctrl,0.1)
        #print(env.get_curr_joints())
        quat=np.zeros(4)
        mujoco.mju_mat2Quat(quat,env.data.site_xmat[0])
        print(quat)
        viewer.sync()
        time.sleep(0.1)

    viewer.close()