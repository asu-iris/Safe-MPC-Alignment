import sys
import os
import time
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
sys.path.append(os.path.abspath(os.getcwd()))

import casadi as cd
from Envs.robot_arm import EFFECTOR_env_mj, End_Effector_model
from Solvers.OCsolver import  ocsolver_v4
import numpy as np
from matplotlib import pyplot as plt

from utils.Visualize_mj import arm_visualizer_mj_v1
from scipy.spatial.transform import Rotation as R
import mujoco

def rot_to_quat(M):
    r=R.from_matrix(M)
    return r.as_quat().reshape(-1,1)

filepath = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                    'mujoco_arm', 'franka_emika_panda',
                    'scene.xml')
print('path', filepath)

dt=0.1
Horizon=20
target_end_pos=[-0.3,0.4,0.5]
target_quat=[0,0,0,1]
target_x=target_end_pos+target_quat

env=EFFECTOR_env_mj(filepath)
arm_model=End_Effector_model(dt=dt)
dyn_f = arm_model.get_dyn_f()

step_cost_vec = np.array([0.0,0.0,1.2]) * 1e-1
step_cost_f = arm_model.get_step_cost_param(step_cost_vec)
term_cost_vec = np.array([0.5,0]) * 1e0
term_cost_f = arm_model.get_terminal_cost_param(term_cost_vec)

controller = ocsolver_v4('arm control')
controller.set_state_param(7, None, None)
controller.set_ctrl_param(6, 6*[-1e10], 6*[1e10])
controller.set_dyn(dyn_f)
controller.set_step_cost(step_cost_f)
controller.set_term_cost(term_cost_f)
controller.construct_prob(horizon=Horizon)

visualizer = arm_visualizer_mj_v1(env, controller=controller)
visualizer.render_init()

u_list=[]
for i in range(200):
    pos=env.data.site_xpos.reshape(-1,1)
    quat=np.zeros(4)
    mujoco.mju_mat2Quat(quat,env.data.site_xmat[0])

    quat=quat.reshape(-1,1)
    
    x=np.concatenate((pos,quat),axis=0)
    u=controller.control(x,target_x=target_x)
    #print(u)
    #break
    u_list.append(np.linalg.norm(u))
    env.step(u,dt)
    x_pred=controller.opt_traj[-7:]
    print('---------------------')
    #print('calculated',arm_model.calc_end_pos(x))
    print('site',env.data.site_xpos)
    site_quat=np.zeros(4)
    mujoco.mju_mat2Quat(site_quat,env.data.site_xmat[0])
    print('site quat', site_quat)
    print('pred',x_pred)
    print('---------------------')
    #break
    visualizer.render_update()
    time.sleep(0.05)

visualizer.close_window()