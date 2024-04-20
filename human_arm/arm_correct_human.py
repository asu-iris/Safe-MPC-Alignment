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
from utils.RBF import generate_rbf_quat
from scipy.spatial.transform import Rotation as R
import mujoco

from utils.recorder import Recorder_Arm

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

step_cost_vec = np.array([0.0,0.0,12.0,1.3]) * 1e-0
step_cost_f = arm_model.get_step_cost_param(step_cost_vec)
term_cost_vec = np.array([3,3]) * 1e1
term_cost_f = arm_model.get_terminal_cost_param(term_cost_vec)

#phi_func =  generate_rbf_quat(Horizon,-0.2,0.2,np.array([1,0,0]),num=10,bias=-0.1,epsilon=2.0,mode='default')
phi_func =  generate_rbf_quat(Horizon,-0.2,0.15,np.array([1,0,0]),num=10,bias=-0.25,epsilon=2.2,mode='cumulative')
test_weight = 2.0*np.ones(10)
test_weight[3]= 0.0
test_weight[4]=-0.5
test_weight[5]=-1.0
test_weight[6]=-1.0
test_weight[7]=-2.0
Gamma=1.0

controller = ocsolver_v4('arm control')
controller.set_state_param(7, None, None)
controller.set_ctrl_param(6, 6*[-1e10], 6*[1e10])
controller.set_dyn(dyn_f)
controller.set_step_cost(step_cost_f)
controller.set_term_cost(term_cost_f)
controller.set_g(phi_func, gamma=Gamma)
controller.construct_prob(horizon=Horizon)

visualizer = arm_visualizer_mj_v1(env, controller=controller)
visualizer.render_init()

#rec=Recorder_Arm(env)
#rec.record_mj()

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
    u=controller.control(x,target_x=target_x,weights=test_weight)
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
    phi=phi_func(controller.opt_traj)
    print('phi', phi)
    print('g',phi.T @ cd.vertcat(1,test_weight))
    print('---------------------')
    #break
    visualizer.render_update()
    #rec.record_mj()
    time.sleep(0.1)

#rec.write()
print(env.get_curr_joints())
visualizer.close_window()
exit()
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