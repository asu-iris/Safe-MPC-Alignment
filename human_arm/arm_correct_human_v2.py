import sys
import os
import time
from pynput import keyboard
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
sys.path.append(os.path.abspath(os.getcwd()))

import casadi as cd
from Envs.robot_arm import EFFECTOR_env_mj, End_Effector_model
from Solvers.OCsolver import  ocsolver_v4
from Solvers.Cutter import  cutter_v4
from Solvers.MVEsolver import mvesolver
import numpy as np
from matplotlib import pyplot as plt

from utils.Visualize_mj import arm_visualizer_mj_v1
from utils.RBF import generate_rbf_quat_z
from utils.Keyboard import arm_key_handler_v2,arm_key_interface_v2
from utils.filter import LowPassFilter_Vel

import mujoco

from utils.recorder import Recorder_Arm
from utils.user_study_logger import UserLogger

#Configuration of log directory
USER_ID=0
TRAIL_ID=0


def mainloop(learned_theta, arm_env, controller, hb_calculator, mve_calc, visualizer, logger=None, recorder=None, filter=None):
    global PAUSE, MSG
    num_corr = 0
    #target_end_pos=[0.35,-0.5,0.5] #[-0.6,-0.5,0.5]
    target_pos_list=[[-0.6,-0.5,0.4],[-0.6,-0.5,0.5],[-0.6,-0.5,0.6]]
    target_quat=[0.0,-0.707,0.0,0.707]
    #target_quat=[-0.36,0.6,0.36,-0.6]
    
    target_idx=0

    while True:
        target_x=target_pos_list[target_idx]+target_quat
        target_idx=(target_idx+1)%3

        print('current theta:', learned_theta)
        arm_env.reset_env()
        controller.reset_warmstart()
        #print(env.get_site_pos())

        human_corr_str = None

        for i in range(400):
            if not PAUSE[0]:
                if MSG[0]:
                    # correction
                    # print('message ',MSG[0])
                    if MSG[0] == 'quit':
                        MSG[0] = None
                        visualizer.close_window()
                        return True, num_corr ,learned_theta

                    if MSG[0] == 'fail':
                        MSG[0] = None
                        visualizer.close_window()
                        return False, num_corr ,learned_theta

                    if MSG[0] == 'reset':
                        MSG[0] = None
                        #recorder.record(True, 'reset')
                        break
                    human_corr = arm_key_interface_v2(MSG)
                    human_corr_str = MSG[0]
                    MSG[0] = None

                    print('correction', human_corr)
                    human_corr_e = np.concatenate([human_corr.reshape(-1, 1), np.zeros((6 * (Horizon - 1), 1))])
                    h, b, h_phi, b_phi = hb_calculator.calc_planes(learned_theta, x, controller.opt_traj,
                                                                   human_corr=human_corr_e,
                                                                   target_x=target_x)

                    mve_calc.add_constraint(h, b[0])
                    mve_calc.add_constraint(h_phi, b_phi[0])
                    try:
                        learned_theta, C = mve_calc.solve()
                    except:
                        return False, num_corr ,learned_theta
                    
                    print('theta', learned_theta)
                    print('vol', np.log(np.linalg.det(C)))

                    num_corr += 1
                    logger.log_correction(human_corr_str)
                    time.sleep(0.1)
                
                # simulation
                x = arm_env.get_curr_state()
                # print(x.flatten())
                try:
                    u = controller.control(x, weights=learned_theta, target_x=target_x)
                except:
                    return False, num_corr ,learned_theta
                
                # visualization
                visualizer.render_update()

                arm_env.step(u)
                time.sleep(0.05)

            else:
                while PAUSE[0]:
                    time.sleep(0.2)

        #print(env.get_curr_joints())
        print(env.get_site_pos())


# list for msg passing
PAUSE = [False]
MSG = [None]

# listener for keyboard ops
listener = keyboard.Listener(
    on_press=lambda key: arm_key_handler_v2(key, PAUSE, MSG),
    on_release=None)
listener.start()

filepath = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                    'mujoco_arm', 'franka_emika_panda',
                    'scene.xml')
print('path', filepath)

dt=0.1
Horizon=20 #10

env=EFFECTOR_env_mj(filepath,dt)
arm_model=End_Effector_model(dt=dt)
dyn_f = arm_model.get_dyn_f()

# step_cost_vec = np.array([0.4,0.0,28.0,1.0]) * 1e0
# step_cost_f = arm_model.get_step_cost_param(step_cost_vec)
step_cost_vec = np.array([2.0,1.0,30.0,30.0,1.0,0.85]) * 1e0 #param:[kr,kq,kvx,kvy,kvz,kw] [0.4,0.0,30.0,30.0,1.0,0.85]
step_cost_f = arm_model.get_step_cost_param_sep(step_cost_vec)
term_cost_vec = np.array([8,6]) * 1e1 #[6,6]
term_cost_f = arm_model.get_terminal_cost_param(term_cost_vec)

theta_dim = 20
hypo_lbs = -3 * np.ones(theta_dim)
hypo_ubs = 5 * np.ones(theta_dim)
init_theta = learned_theta = (hypo_lbs + hypo_ubs) / 2

phi_func =  generate_rbf_quat_z(Horizon,x_center=-0.15,x_half=0.15,ref_axis=np.array([1,0,0]),num_q=10, #half:0.15
                                z_min=0.2,z_max=0.9, num_z=10, bias=-0.8, epsilon_z=12, epsilon_q=1.8,z_factor=0.05,mode='cumulative')

Gamma=1.5 #1.0

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

hb_calculator = cutter_v4('arm cut')
hb_calculator.from_controller(controller)
hb_calculator.construct_graph(horizon=Horizon)

mve_calc = mvesolver('uav_mve', theta_dim)
mve_calc.set_init_constraint(hypo_lbs, hypo_ubs)
#rec=Recorder_Arm(env)
#rec.record_mj()

# logger
logger_path=os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),'Data','user_study_arm_mj'
                         ,"user_"+str(USER_ID),'trial_'+str(TRAIL_ID))
logger = UserLogger(user=USER_ID,trail=TRAIL_ID,dir=logger_path)

flag, cnt, weights = mainloop(learned_theta=learned_theta,
         arm_env=env,
         controller=controller,
         hb_calculator=hb_calculator,
         mve_calc=mve_calc,
         visualizer=visualizer,
         logger=logger)
print(flag, cnt)
logger.log_termination(flag, cnt,weights)
#recorder.write()
sys.exit()