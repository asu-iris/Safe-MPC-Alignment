import sys
import os
import time
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
sys.path.append(os.path.abspath(os.getcwd()))

import casadi as cd
from Envs.robot_arm import ARM_env_mj,Robot_Arm_model
from Solvers.OCsolver import  ocsolver_v3
import numpy as np

from utils.Visualize_mj import arm_visualizer_mj_v1

filepath = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                    'mujoco_arm', 'franka_emika_panda',
                    'scene.xml')
print('path', filepath)

dt=0.1
Horizon=20
target_end_pos=(0.0,0.5,0.5)

env=ARM_env_mj(filepath)
arm_model=Robot_Arm_model(dt=dt)
dyn_f = arm_model.get_dyn_f()

step_cost_vec = np.array([0.1,0.1,0.1,4]) * 1e-1
step_cost_f = arm_model.get_step_cost_param(step_cost_vec)
term_cost_vec = np.array([5, 5, 5]) * 1e1
term_cost_f = arm_model.get_terminal_cost_param(term_cost_vec)

controller = ocsolver_v3('arm control')
controller.set_state_param(7, None, None)
controller.set_ctrl_param(7, 7*[-1e10], 7*[1e10])
controller.set_dyn(dyn_f)
controller.set_step_cost(step_cost_f)
controller.set_term_cost(term_cost_f)
controller.construct_prob(horizon=Horizon)

visualizer = arm_visualizer_mj_v1(env, controller=controller)
visualizer.render_init()

for i in range(400):
    x=env.get_curr_state()
    #print(x)
    #print(arm_model.calc_end_pos(x))
    #break
    u=controller.control(x)
    #print(u)
    env.step(0,dt)
    x=env.get_curr_state()
    print(x)
    print(arm_model.calc_end_pos(x))
    #break
    visualizer.render_update()
    time.sleep(10)

visualizer.close_window()