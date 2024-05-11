import sys
import os
import time

sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
sys.path.append(os.path.abspath(os.getcwd()))

import casadi as cd
from Envs.robot_arm import EFFECTOR_env_mj, End_Effector_model, DH_to_Mat
from Solvers.OCsolver import  ocsolver_v4
from Solvers.Cutter import  cutter_v4
from Solvers.MVEsolver import mvesolver
import numpy as np
from matplotlib import pyplot as plt

from utils.Visualize_mj import arm_visualizer_mj_v1


import mujoco

from utils.recorder import Recorder_Arm

print(os.path.abspath(os.path.dirname(os.getcwd())))
filepath = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                    'mujoco_arm', 'franka_emika_panda',
                    'scene.xml')
print('path', filepath)

dt=0.1
Horizon=10

env=EFFECTOR_env_mj(filepath,dt)
arm_model=End_Effector_model(dt=dt)
dyn_f = arm_model.get_dyn_f()

step_cost_vec = np.array([8,0.0,1.0,1.5]) * 1e0
step_cost_f = arm_model.get_step_cost_param(step_cost_vec)
term_cost_vec = np.array([6,6]) * 1e1
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

target_end_pos=[-0.25,-0.5,0.5]
target_quat=[0.0,-0.707,0.0,0.707]
target_x=target_end_pos+target_quat
for i in range(400):
    x = env.get_curr_state()
    u = controller.control(x, target_x=target_x)
    visualizer.render_update()
    env.step(u)
    time.sleep(0.1)
    q=env.get_curr_joints()
    calc_pos=(DH_to_Mat(q) @ np.array([0,0,0,1]))[0:3]
    print('---------------------')
    print('site',env.get_site_pos())
    print('calc',calc_pos)
print(env.get_curr_joints())
