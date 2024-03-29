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
from Solvers.OCsolver import ocsolver_v2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from Solvers.Cutter import cutter_v2
from Solvers.MVEsolver import mvesolver
from utils.RBF import generate_phi_rbf,gen_eval_rbf

from utils.Visualize_mj import uav_visualizer_mj_v2
from utils.Keyboard import uav_key_handler,key_interface,remove_conflict
import mujoco

def mainloop(learned_theta,uav_env,controller,hb_calculator,mve_calc,visualizer):
    global PAUSE,MSG
    num_corr = 0
    while True:

        print('current theta:', learned_theta)

        init_r = np.array([0, 0, 0]).reshape(-1, 1)
        init_v = np.zeros((3, 1))
        init_q = np.reshape(np.array([1, 0, 0, 0]), (-1, 1))
        # print(Quat_Rot(init_q))
        init_w_B = np.zeros((3, 1))
        init_x = np.concatenate([init_r, init_v, init_q, init_w_B], axis=0)
        #init_x[0] = np.random.uniform(1.0, 7.0)
        # init_x[0]=1
        init_x[1] = np.random.uniform(2.0, 4.0)
        init_x[2] = np.random.uniform(0.3, 2.0)
        # init_x[1]=1
        # print('init state', init_x.T)

        uav_env.set_init_state(init_x)
        controller.reset_warmstart()
        for i in range(400):
            if not PAUSE[0]:
                if MSG[0]:
                    # correction
                    # print('message ',MSG[0])
                    if MSG[0] == 'quit':
                        MSG[0] = None
                        visualizer.close_window()
                        return True,num_corr
                    
                    if MSG[0] == 'fail':
                        MSG[0] = None
                        visualizer.close_window()
                        return False,num_corr
                    
                    if MSG[0] == 'reset':
                        MSG[0] = None
                        break
                    human_corr=key_interface(MSG)
                    MSG[0] = None

                    print('correction', human_corr)
                    human_corr_e = np.concatenate([human_corr.reshape(-1, 1), np.zeros((4 * (Horizon - 1), 1))])
                    h, b, h_phi, b_phi = hb_calculator.calc_planes(learned_theta, x, controller.opt_traj, human_corr=human_corr_e)

                    mve_calc.add_constraint(h, b[0])
                    mve_calc.add_constraint(h_phi, b_phi[0])
                    try:
                        learned_theta, C = mve_calc.solve()
                    except:
                        return False,num_corr
                    
                    print('vol', np.log(np.linalg.det(C)))

                    # mve_calc.savefig(C,learned_theta,np.array([-5,-5]),dir='D:\\ASU_Work\\Research\\learn safe mpc\\experiment\\results\\cut_figs\\' +str(num_corr)+'.png')
                    num_corr += 1
                    time.sleep(0.05)

                # simulation
                x = uav_env.get_curr_state()
                #print(x.flatten())
                u = controller.control(x, weights=learned_theta)
                visualizer.render_update()

                uav_env.step(u)
                time.sleep(0.05)

            else:
                while PAUSE[0]:
                    time.sleep(0.2)

#list for msg passing
PAUSE = [False]
MSG = [None]

# listener for keyboard ops
listener = keyboard.Listener(
    on_press=lambda key: uav_key_handler(key, PAUSE, MSG),
    on_release=None)
listener.start()

# set up safety features
Horizon = 15  # 25
Gamma = 5  #10

#set up rbf function
rbf_mode='gau_rbf_sep_cum'
rbf_X_c=np.array([9.8])
rbf_Y_c=np.linspace(0,10,12)#5
rbf_Z_c=np.linspace(0,10,12)#5
phi_func = generate_phi_rbf(Horizon,X_c=rbf_X_c,Y_c=rbf_Y_c,Z_c=rbf_Z_c,epsilon=0.45,bias=-1,mode=rbf_mode)

theta_dim = 24
hypo_lbs = -80 * np.ones(theta_dim)
hypo_ubs = 100 * np.ones(theta_dim)
init_theta = learned_theta = (hypo_lbs + hypo_ubs) / 2
# get dynamics, set up step cost and terminal cost
uav_params = {'gravity': 9.8, 'm': 0.1, 'J_B': 0.01 * np.eye(3), 'l_w': 1.2, 'dt': 0.1, 'c': 1}
filepath=os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),'mujoco_uav','bitcraze_crazyflie_2','scene.xml')
print('path',filepath)
uav_env = UAV_env_mj(filepath)
uav_model = UAV_model(**uav_params)
dyn_f = uav_model.get_dyn_f()
######################################################################################

# r,v,q,w,u
step_cost_vec = np.array([0.1, 200, 1, 5, 0.01]) * 1e-2
step_cost_f = uav_model.get_step_cost(step_cost_vec, target_pos=np.array([19, 9, 9]))
term_cost_vec = np.array([10, 6, 1, 5]) * 1e0
term_cost_f = uav_model.get_terminal_cost(term_cost_vec, target_pos=np.array([19, 9, 9]))

#########################################################################################
controller = ocsolver_v2('uav control')
controller.set_state_param(13, None, None)
controller.set_ctrl_param(4, [-1e10, -1e10, -1e10, -1e10], [1e10, 1e10, 1e10, 1e10])
controller.set_dyn(dyn_f)
controller.set_step_cost(step_cost_f)
controller.set_term_cost(term_cost_f)
controller.set_g(phi_func, gamma=Gamma)
controller.construct_prob(horizon=Horizon)
#init_theta = learned_theta = (hypo_lbs + hypo_ubs) / 2
######################################################################################

######################################################################################
visualizer = uav_visualizer_mj_v2(uav_env,controller=controller)
visualizer.render_init()
######################################################################################

#########################################################################################
#  cutter
hb_calculator = cutter_v2('uav cut')
hb_calculator.from_controller(controller)
hb_calculator.construct_graph(horizon=Horizon)
#########################################################################################


#########################################################################################
# MVESolver
mve_calc = mvesolver('uav_mve', theta_dim)
mve_calc.set_init_constraint(hypo_lbs, hypo_ubs)  # Theta_0
#########################################################################################
flag,cnt=mainloop(learned_theta=learned_theta,
         uav_env=uav_env,
         controller=controller,
         hb_calculator=hb_calculator,
         mve_calc=mve_calc,
         visualizer=visualizer)
print(flag,cnt)
sys.exit()
init_r = np.array([0, 0, 0]).reshape(-1, 1)
init_v = np.zeros((3, 1))
init_q = np.reshape(np.array([1, 0, 0, 0]), (-1, 1))
# print(Quat_Rot(init_q))
init_w_B = np.zeros((3, 1))
init_x = np.concatenate([init_r, init_v, init_q, init_w_B], axis=0)
uav_env.set_init_state(init_x)
for i in range(400):
    x = uav_env.get_curr_state()
    u = controller.control(x,weights=learned_theta)
    #print(x[0:3])
    #print(u)
    visualizer.render_update()
    uav_env.step(u)
    time.sleep(0.1)