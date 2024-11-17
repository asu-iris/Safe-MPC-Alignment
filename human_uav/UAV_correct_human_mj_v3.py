"""
The code for simulated drone navigation game
to run: python UAV_correct_human_mj_v3.py
"""

import time
from pynput import keyboard
import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
sys.path.append(os.path.abspath(os.getcwd()))
from utils.Keyboard import uav_key_handler
import matplotlib.pyplot as plt
import casadi as cd
from Envs.UAV import UAV_env_mj, UAV_model
from Solvers.OCsolver import  ocsolver_v3
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from Solvers.Cutter import  cutter_v3
from Solvers.MVEsolver import mvesolver
from utils.RBF import generate_phi_rbf, gen_eval_rbf

from utils.Visualize_mj import uav_visualizer_mj_v4
from utils.Keyboard import uav_key_handler, key_interface, remove_conflict
from utils.user_study_logger import UserLogger
from utils.recorder import Recorder_sync

#from data_process.heatmap import heatmap_weight_uav
import mujoco

#Configuration of log directory
import argparse

parser = argparse.ArgumentParser(description='Parser for User and Trial IDs')
parser.add_argument('-u','--user_id',help='User ID',default=0)
parser.add_argument('-t','--trial_id',help='Trial ID',default=0)
args = parser.parse_args()

USER_ID=args.user_id
TRIAL_ID=args.trial_id

def mainloop(learned_theta, uav_env, controller, hb_calculator, mve_calc, visualizer, logger=None, recorder=None):
    global PAUSE, MSG
    num_corr = 0

    target_idx=0
    traj_idx=0

    #heatmap_weight_uav(learned_theta,name='heatmap_0.png')
    while True:

        print('current theta:', learned_theta)

        init_r = np.array([0, 0, 0]).reshape(-1, 1)
        init_v = np.zeros((3, 1))
        init_q = np.reshape(np.array([1, 0, 0, 0]), (-1, 1))
        # print(Quat_Rot(init_q))
        init_w_B = np.zeros((3, 1))
        init_x = np.concatenate([init_r, init_v, init_q, init_w_B], axis=0)
        # init_x[0] = np.random.uniform(1.0, 7.0)
        init_x[0] = 0.1
        init_x[1] = np.random.uniform(4.0, 6.0)
        # init_x[2] = np.random.uniform(0.3, 2.0)
        #init_x[1] = 5
        init_x[2] = 0.5
        # print('init state', init_x.T)

        # Three Different Target Positions
        target_r_set = [np.array([19, 1, 9]), np.array([19, 5, 9]), np.array([19, 9, 9])]
        target_r=target_r_set[target_idx]

        print('target_r', target_r)
        visualizer.set_target_pos(target_r)
        if recorder is not None:
            recorder.set_target_pos(target_r)

        uav_env.set_init_state(init_x)
        controller.reset_warmstart()

        correction_flag = False
        human_corr_str = None
        for i in range(800):
            if not PAUSE[0]:
                if MSG[0]:
                    # correction
                    # print('message ',MSG[0])
                    if MSG[0] == 'quit':
                        MSG[0] = None
                        logger.log_trajectory(uav_env.get_traj_arr(),str(traj_idx)+'_target_'+str(target_idx)+'_cnum_'+str(num_corr))
                        visualizer.close_window()
                        return True, num_corr ,learned_theta

                    if MSG[0] == 'fail':
                        MSG[0] = None
                        logger.log_trajectory(uav_env.get_traj_arr(),str(traj_idx)+'_target_'+str(target_idx)+'_cnum_'+str(num_corr))
                        visualizer.close_window()
                        return False, num_corr ,learned_theta

                    if MSG[0] == 'reset':
                        MSG[0] = None
                        if recorder is not None:
                            recorder.record(True, 'reset')
                        break
                    human_corr = key_interface(MSG)
                    human_corr_str = MSG[0]
                    MSG[0] = None

                    print('correction', human_corr)
                    human_corr_e = np.concatenate([human_corr.reshape(-1, 1), np.zeros((4 * (Horizon - 1), 1))])
                    # Core Part: Use Human Corrections to Update the params
                    st = time.time()
                    h, b, h_phi, b_phi = hb_calculator.calc_planes(learned_theta, x, controller.opt_traj,
                                                                   human_corr=human_corr_e,
                                                                   target_r=target_r)

                    mve_calc.add_constraint(h, b[0])
                    mve_calc.add_constraint(h_phi, b_phi[0])
                    try:
                        learned_theta, C = mve_calc.solve()
                    except:
                        return False, num_corr ,learned_theta
                    print('calc time',time.time() - st)
                    print('vol', np.log(np.linalg.det(C)))

                    # mve_calc.savefig(C,learned_theta,np.array([-5,-5]),dir='D:\\ASU_Work\\Research\\learn safe mpc\\experiment\\results\\cut_figs\\' +str(num_corr)+'.png')
                    num_corr += 1
                    correction_flag = True
                    logger.log_correction(human_corr_str)
                    #heatmap_weight_uav(learned_theta,name='heatmap_'+str(num_corr)+'.png')
                    #time.sleep(0.05)

                # simulation
                x = uav_env.get_curr_state()
                # print(x.flatten())
                try:
                    u = controller.control(x, weights=learned_theta, target_r=target_r)
                except:
                    return False, num_corr ,learned_theta

                # recording
                #recorder.record(correction_flag, human_corr_str)
                # visualization
                visualizer.render_update()
                if recorder is not None:
                    recorder.record(correction_flag, human_corr_str)

                correction_flag = False
                human_corr_str = None
                

                uav_env.step(u)
                time.sleep(0.03)

            else:
                while PAUSE[0]:
                    time.sleep(0.2)
        
        logger.log_trajectory(uav_env.get_traj_arr(),str(traj_idx)+'_target_'+str(target_idx)+'_cnum_'+str(num_corr))
        traj_idx+=1
        target_idx=(target_idx+1)%3


# list for msg passing
PAUSE = [False]
MSG = [None]

# listener for keyboard ops
listener = keyboard.Listener(
    on_press=lambda key: uav_key_handler(key, PAUSE, MSG),
    on_release=None)
listener.start()

# set up safety features
Horizon = 15  # 25
Gamma = 50  # 10

# set up rbf function
rbf_mode = 'gau_rbf_sep_cum'
rbf_X_c = np.array([9.8])
rbf_Y_c = np.linspace(0, 10, 10)  # 5
rbf_Z_c = np.linspace(0, 10, 10)  # 5
phi_func = generate_phi_rbf(Horizon, X_c=rbf_X_c, Y_c=rbf_Y_c, Z_c=rbf_Z_c, epsilon=0.45, bias=-1, mode=rbf_mode)

theta_dim = 20
hypo_lbs = -80 * np.ones(theta_dim)
hypo_ubs = 100 * np.ones(theta_dim)
init_theta = learned_theta = (hypo_lbs + hypo_ubs) / 2
# init_theta = learned_theta = np.zeros(24)
# get dynamics, set up step cost and terminal cost
uav_params = {'gravity': 9.8, 'm': 0.1, 'J_B': 0.01 * np.eye(3), 'l_w': 1.2, 'dt': 0.1, 'c': 1}
filepath = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                        'mujoco_uav', 'bitcraze_crazyflie_2',
                        'scene_rand.xml')
print('path', filepath)
uav_env = UAV_env_mj(filepath, lock_flag=True)
uav_model = UAV_model(**uav_params)
dyn_f = uav_model.get_dyn_f()
######################################################################################

# r,v,q,w,u
step_cost_vec = np.array([0.05, 200, 1, 5, 0.01]) * 1e-2
step_cost_f = uav_model.get_step_cost_param(step_cost_vec)
term_cost_vec = np.array([10, 6, 1, 5]) * 1e0
term_cost_f = uav_model.get_terminal_cost_param(term_cost_vec)

#########################################################################################
controller = ocsolver_v3('uav control')
controller.set_state_param(13, None, None)
controller.set_ctrl_param(4, [-1e10, -1e10, -1e10, -1e10], [1e10, 1e10, 1e10, 1e10])
controller.set_dyn(dyn_f)
controller.set_step_cost(step_cost_f)
controller.set_term_cost(term_cost_f)
controller.set_g(phi_func, gamma=Gamma)
controller.construct_prob(horizon=Horizon)
# init_theta = learned_theta = (hypo_lbs + hypo_ubs) / 2
######################################################################################

######################################################################################
visualizer = uav_visualizer_mj_v4(uav_env, controller=controller)
visualizer.render_init()
######################################################################################

#########################################################################################
#  cutter
hb_calculator = cutter_v3('uav cut')
hb_calculator.from_controller(controller)
hb_calculator.construct_graph(horizon=Horizon)
#########################################################################################


#########################################################################################
# MVESolver
mve_calc = mvesolver('uav_mve', theta_dim)
mve_calc.set_init_constraint(hypo_lbs, hypo_ubs)  # Theta_0
#########################################################################################

#########################################################################################
# logger
logger_path=os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),'Data','user_study_uav'
                         ,"user_"+str(USER_ID),'trial_'+str(TRIAL_ID))
logger = UserLogger(user=USER_ID,trail=TRIAL_ID,dir=logger_path)
#########################################################################################

#########################################################################################
# recorder
#recorder = Recorder_sync(env=uav_env, controller=controller,visualizer=visualizer,cam_flag=True)
recorder = None
#########################################################################################
flag, cnt, weights = mainloop(learned_theta=learned_theta,
                     uav_env=uav_env,
                     controller=controller,
                     hb_calculator=hb_calculator,
                     mve_calc=mve_calc,
                     visualizer=visualizer,
                     logger=logger,
                     recorder=recorder)
print(flag, cnt)
logger.log_termination(flag, cnt,weights)
#recorder.write()
sys.exit()
