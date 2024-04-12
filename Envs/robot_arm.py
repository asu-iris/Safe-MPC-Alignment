import numpy as np
import casadi as cd

import mujoco
import mujoco.viewer

import os

class Robot_Arm_model(object):
    def __init__(self,dt) -> None:
        self.dt=dt
        self.DHForm_p1 = cd.vertcat(
            cd.horzcat(0,0.333,0),
            cd.horzcat(0,0,-cd.pi/2),
            cd.horzcat(0,0.316,cd.pi/2),
            cd.horzcat(0.0825,0,cd.pi/2),
            cd.horzcat(-0.0825,0.384,-cd.pi/2),
            cd.horzcat(0,0,cd.pi/2),
            cd.horzcat(0.088,0,cd.pi/2),
            cd.horzcat(0,0.107,0), #flange
        )

    def get_dyn_f(self):
        x_t=cd.SX.sym('x_t',7)
        u_t=cd.SX.sym('u_t',7)
        x_t_1= self.dt*u_t + x_t
        return cd.Function('arm_dynamics', [x_t, u_t], [x_t_1])
    
    def get_step_cost_param(self, param_vec: np.ndarray): #param:[kx,ky,kz,ku]
        q=cd.SX.sym('q',7) # theta angles
        u=cd.SX.sym('u',7) #speed control
        target_end_pos=cd.SX.sym('target',3)

        DHForm_P2=cd.vertcat(q,0)
        current_DHForm=cd.horzcat(self.DHForm_p1,DHForm_P2)
        #print(current_DHForm.shape)
        current_end_pos=(DHForm_to_Mat(current_DHForm) @ cd.DM(np.array([0,0,0,1])))[0:3]
        l_vec=cd.vertcat((current_end_pos-target_end_pos)**2,cd.sumsqr(u))
        p_vec = cd.DM(param_vec)
        cost=p_vec.T @ l_vec
        #print(cost)
        return cd.Function('step_cost', [q, u, target_end_pos], [cost])
    
    def get_terminal_cost_param(self, param_vec: np.ndarray): #param:[kx,ky,kz]
        q=cd.SX.sym('q',7) # theta angles
        target_end_pos=cd.SX.sym('target',3)

        DHForm_P2=cd.vertcat(q,0)
        current_DHForm=cd.horzcat(self.DHForm_p1,DHForm_P2)
        current_end_pos=(DHForm_to_Mat(current_DHForm) @ cd.DM(np.array([0,0,0,1])))[0:3]
        l_vec=(current_end_pos-target_end_pos)**2
        p_vec = cd.DM(param_vec)
        cost=p_vec.T @ l_vec

        return cd.Function('step_cost', [q, target_end_pos], [cost])
    
    def calc_end_pos(self,q):
        q=cd.DM(q)
        DHForm_P2=cd.vertcat(q,0)
        current_DHForm=cd.horzcat(self.DHForm_p1,DHForm_P2)
        current_end_pos=(DHForm_to_Mat(current_DHForm) @ cd.DM(np.array([0,0,0,1])))[0:3]
        return current_end_pos.full()

    
class ARM_env_mj(object):
    def __init__(self, xml_path) -> None:
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)


    def set_init_state(self, x: np.ndarray):
        mujoco.mj_resetData(self.model, self.data)
        x = x.flatten()
        self.data.qpos = x

    def step(self, u, dt):  # u:speed of joints
        loop_num=int(dt/0.002)
        inner_ctrl = self.data.qpos[0:7] + u*dt
        #print('inner',inner_ctrl)
        for i in range(loop_num):
            self.data.ctrl[0:7] = inner_ctrl.flatten()
            self.data.ctrl[7] = 0
            mujoco.mj_step(self.model, self.data)


    def get_curr_state(self):
        return self.data.qpos[0:7]
       

def DHLine_to_Mat(a,d,alpha,theta):
    A_2=cd.vertcat(
        cd.horzcat(cd.cos(theta),-cd.sin(theta),0,0),
        cd.horzcat(cd.sin(theta),cd.cos(theta),0,0),
        cd.horzcat(0,0,1,d),
        cd.horzcat(0,0,0,1),
    )

    A_1=cd.vertcat(
        cd.horzcat(1,0,0,a),
        cd.horzcat(0,cd.cos(alpha),-cd.sin(alpha),0),
        cd.horzcat(0,cd.sin(alpha),cd.cos(alpha),0),
        cd.horzcat(0,0,0,1),
    )

    return A_2 @ A_1
    #return A_1 @ A_2

def DHForm_to_Mat(DHForm):
    Trans_Mat=cd.DM(np.eye(4)) #from end effector to world
    num_joint=DHForm.shape[0]
    for i in np.arange(num_joint-1,-1,-1):
        #print(i)
        Trans_Mat = DHLine_to_Mat(DHForm[i,0],DHForm[i,1],DHForm[i,2],DHForm[i,3]) @ Trans_Mat
    return Trans_Mat

if __name__=='__main__':
    test_DHForm=cd.vertcat(
            cd.horzcat(0,0.333,0,0),
            cd.horzcat(0,0,-cd.pi/2,0),
            cd.horzcat(0,0.316,cd.pi/2,0),
            cd.horzcat(0.0825,0,cd.pi/2,0),
            cd.horzcat(-0.0825,0.384,-cd.pi/2,0),
            cd.horzcat(0,0,cd.pi/2,0.28),
            cd.horzcat(0.088,0,cd.pi/2,-0.11),
            cd.horzcat(0,0.107,0,0), #flange
        )
    #viewer=mujoco.viewer.launch()
    x=np.array([0,0,0,1]).reshape(-1,1)
    for i in np.arange(7,-1,-1):
        x=DHLine_to_Mat(test_DHForm[i,0],test_DHForm[i,1],test_DHForm[i,2],test_DHForm[i,3]) @ x
        print(x)
    #q=DHForm_to_Mat(test_DHForm) @ np.array([0,0,0,1]).reshape(-1,1)
    #print(q)
    exit()
    filepath = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                        'mujoco_arm', 'franka_emika_panda',
                        'scene.xml')
    print('path', filepath)
    env=ARM_env_mj(filepath)
    #viewer=mujoco.viewer.launch(env.model,env.data)
    for i in range(20):
        env.step(0.05*np.ones(7),0.1)
        print(env.get_curr_state())