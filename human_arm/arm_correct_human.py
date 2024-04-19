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
Horizon=10
target_end_pos=[0.35,-0.5,0.5]
target_quat=[0.0,-0.707,0.0,0.707]
#target_quat=[-0.36,0.6,0.36,-0.6]
target_x=target_end_pos+target_quat

env=EFFECTOR_env_mj(filepath)
arm_model=End_Effector_model(dt=dt)
dyn_f = arm_model.get_dyn_f()

step_cost_vec = np.array([0.0,0.0,8.0,2.0]) * 1e-1
step_cost_f = arm_model.get_step_cost_param(step_cost_vec)
term_cost_vec = np.array([1.5,2.5]) * 1e0
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

site_x_list=[]
site_y_list=[]
site_v_list=[]
for i in range(300):
    pos=env.get_site_pos().reshape(-1,1)
    quat=env.get_site_quat()
    quat=quat.reshape(-1,1)
    
    x=np.concatenate((pos,quat),axis=0)
    theta=min(np.pi*i/300,np.pi/2)
    track_target_pos=[0.55 * np.cos(theta), 0.55 * np.sin(theta), 0.55]
    track_target_quat=[0,1,0,0]
    track_target= track_target_pos + track_target_quat
    #if i > 100:
        #target_x = [0.4,-0.5,0.5,0.0,-0.707,0.0,0.707]
    u=controller.control(x,target_x=target_x)
    #print(u)
    #break
    env.step(u,dt)
    x_pred=controller.opt_traj[-7:]
    print('---------------------')
    #print('calculated',arm_model.calc_end_pos(x))
    site_pos=env.get_site_pos()
    print('site',site_pos)
    site_x_list.append(site_pos[0])
    site_y_list.append(site_pos[1])
    site_v_list.append(np.linalg.norm(env.get_site_vel()))
    site_quat=env.get_site_quat()
    hand_quat=env.get_hand_quat()
    print('site quat', site_quat)
    #print('hand quat', hand_quat)
    #print('pred',x_pred)
    print('---------------------')
    #break
    visualizer.render_update()
    time.sleep(0.05)

print(env.get_curr_joints())
visualizer.close_window()
plt.figure(0)
plt.title('xy')
plt.scatter(site_x_list,site_y_list)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

plt.figure(1)
plt.title('v')
plt.plot(site_v_list)
plt.xlabel('t')
plt.show()