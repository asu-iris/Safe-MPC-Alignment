import time
from pynput import keyboard
import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
sys.path.append(os.path.abspath(os.getcwd()))
from utils.Keyboard import uav_key_handler
import matplotlib.pyplot as plt
import casadi as cd
from Envs.UAV import UAV_env, UAV_model
from Solvers.OCsolver import ocsolver_v2
import numpy as np
from matplotlib import pyplot as plt
from Solvers.Cutter import cutter_v2
from Solvers.MVEsolver import mvesolver
from utils.Correction import Correction_Agent, uav_trans
from utils.RBF import rbf

from utils.Visualize import uav_visualizer
from utils.Keyboard import uav_key_handler

# list for msg passing
PAUSE = [False]
MSG = [None]

# listener for keyboard ops
listener = keyboard.Listener(
    on_press=lambda key: uav_key_handler(key, PAUSE, MSG),
    on_release=None)
listener.start()

center = (5, 5, 3)
radius = 2

# set up safety features
Horizon = 20  # 25
Gamma = 10



def generate_phi_x_single():
    x_dim = 13
    u_dim = 4
    traj = cd.SX.sym('xi', (x_dim + u_dim) * Horizon + x_dim)
    x_pos_1 = traj[5 * (x_dim + u_dim)]
    y_pos_1 = traj[5 * (x_dim + u_dim) + 1]
    phi = cd.vertcat(cd.DM(1 * radius ** 2), (x_pos_1 - center[0]) * (x_pos_1 - center[0]),
                     (y_pos_1 - center[1]) * (y_pos_1 - center[1]))  # to make theta_H [-5,-5]
    return cd.Function('phi', [traj], [phi])


def generate_phi_x_cum():
    x_dim = 13
    u_dim = 4
    traj = cd.SX.sym('xi', (x_dim + u_dim) * Horizon + x_dim)
    sum_phi = np.zeros(3)
    for t in range(Horizon):
        x_pos_t = traj[t * (x_dim + u_dim)]
        y_pos_t = traj[t * (x_dim + u_dim) + 1]
        phi = cd.vertcat(cd.DM(radius ** 2), (x_pos_t - center[0]) * (x_pos_t - center[0]),
                         (y_pos_t - center[1]) * (y_pos_t - center[1]))  # to make theta_H [-5,-5]
        sum_phi += phi
    return cd.Function('phi', [traj], [sum_phi])


def generate_phi_rbf():
    x_dim = 13
    u_dim = 4
    traj = cd.SX.sym('xi', (x_dim + u_dim) * Horizon + x_dim)
    x_pos = traj[5 * (x_dim + u_dim)]
    y_pos = traj[5 * (x_dim + u_dim) + 1]
    z_pos_1 = traj[2 * (x_dim + u_dim) + 2]

    sum_phi = np.zeros(3)
    for t in range(Horizon):
        phi_list = []
        phi_list.append(-2)  # -4:16
        X_c = np.linspace(4, 6, 3)
        Y_c = np.linspace(4, 6, 3)
        grid_x, grid_y = np.meshgrid(X_c, Y_c)
        grid_x = grid_x.reshape(-1, 1)
        grid_y = grid_y.reshape(-1, 1)
        centers = np.concatenate([grid_x, grid_y], axis=1)
        for center in centers:
            print(center)
            phi_i = rbf(x_pos, y_pos, center[0], center[1], 1.5)
            phi_list.append(-phi_i)

        phi = cd.vertcat(*phi_list)
    return cd.Function('phi', [traj], [phi])


phi_func = generate_phi_rbf()
theta_dim = 9
hypo_lbs = -5 * np.ones(theta_dim)
hypo_ubs = 0 * np.ones(theta_dim)



######################################################################################
# get dynamics, set up step cost and terminal cost
uav_params = {'gravity': 10, 'm': 1, 'J_B': 0.1 * np.eye(3), 'l_w': 0.5, 'dt': 0.1, 'c': 1}
uav_env = UAV_env(**uav_params)
uav_model = UAV_model(**uav_params)
dyn_f = uav_model.get_dyn_f()
######################################################################################


######################################################################################
visualizer = uav_visualizer(uav_env, [0, 0, 0], [10, 10, 10])
visualizer.render_init()
######################################################################################


# r,v,q,w,u
step_cost_vec = np.array([5, 50, 1, 1, 0.01]) * 1e-1
step_cost_f = uav_model.get_step_cost(step_cost_vec, target_pos=np.array([9, 9, 5]))
term_cost_vec = np.array([100, 6, 1, 50]) * 1e-1
term_cost_f = uav_model.get_terminal_cost(term_cost_vec, target_pos=np.array([9, 9, 5]))

#########################################################################################
controller = ocsolver_v2('uav control')
controller.set_state_param(13, None, None)
controller.set_ctrl_param(4, [-1e10, -1e10, -1e10, -1e10], [1e10, 1e10, 1e10, 1e10])
controller.set_dyn(dyn_f)
controller.set_step_cost(step_cost_f)
controller.set_term_cost(term_cost_f)
controller.set_g(phi_func, gamma=Gamma)
controller.construct_prob(horizon=Horizon)
learned_theta = (hypo_lbs + hypo_ubs) / 2
######################################################################################


#########################################################################################
#  cutter
hb_calculator = cutter_v2('uav cut')
hb_calculator.set_state_dim(13)
hb_calculator.set_ctrl_dim(4)
hb_calculator.set_dyn(dyn_f)
hb_calculator.set_step_cost(step_cost_f)
hb_calculator.set_term_cost(term_cost_f)
hb_calculator.set_g(phi_func, gamma=Gamma)
hb_calculator.construct_graph(horizon=Horizon)
#########################################################################################


#########################################################################################
# MVESolver
mve_calc = mvesolver('uav_mve', theta_dim)
mve_calc.set_init_constraint(hypo_lbs, hypo_ubs)  # Theta_0
#########################################################################################


num_corr = 0
while True:

    print('current theta:', learned_theta)

    init_r = np.array([0, 0, 0]).reshape(-1, 1)
    init_v = np.zeros((3, 1))
    init_q = np.reshape(np.array([1, 0, 0, 0]), (-1, 1))
    # print(Quat_Rot(init_q))
    init_w_B = np.zeros((3, 1))
    init_x = np.concatenate([init_r, init_v, init_q, init_w_B], axis=0)
    # init_x[0] = np.random.uniform(0.0, 2.5)
    # init_x[0]=1
    # init_x[1] = np.random.uniform(0.5, 8.5)
    # init_x[1]=1
    # print('init state', init_x.T)

    uav_env.set_init_state(init_x)
    for i in range(120):
        if not PAUSE[0]:
            if MSG[0]:
                # correction
                # print('message ',MSG[0])
                if MSG[0] == 'up':  # y+
                    # print(uav_trans(np.array([0,1,0]),uav_env))
                    # human_corr=uav_trans(np.array([0,1,0]),uav_env)
                    human_corr = np.array([-1, 0, 1, 0])
                    print('current key:', MSG[0])
                if MSG[0] == 'down':  # y-
                    # print(uav_trans(np.array([0,-1,0]),uav_env))
                    # human_corr=uav_trans(np.array([0,-1,0]),uav_env)
                    human_corr = np.array([1, 0, -1, 0])
                    print('current key:', MSG[0])

                if MSG[0] == 'right':  # x+
                    # print(uav_trans(np.array([1,0,0]),uav_env))
                    # human_corr=uav_trans(np.array([1,0,0]),uav_env)
                    human_corr = np.array([0, -1, 0, 1])
                    print('current key:', MSG[0])

                if MSG[0] == 'left':  # x-
                    # print(uav_trans(np.array([-1,0,0]),uav_env))
                    # human_corr=uav_trans(np.array([-1,0,0]),uav_env)
                    human_corr = np.array([0, 1, 0, -1])
                    print('current key:', MSG[0])

                if MSG[0] == 'quit' or MSG[0] == 'reset':
                    break
                MSG[0] = None

                print('correction', human_corr)
                human_corr_e = np.concatenate([human_corr.reshape(-1, 1), np.zeros((4 * (Horizon - 1), 1))])
                h, b, h_phi, b_phi = hb_calculator.calc_planes(learned_theta, x, controller.opt_traj, human_corr=human_corr_e)

                mve_calc.add_constraint(h, b[0])
                mve_calc.add_constraint(h_phi, b_phi[0])
                learned_theta, C = mve_calc.solve()
                print('vol', np.log(np.linalg.det(C)))

                # mve_calc.savefig(C,learned_theta,np.array([-5,-5]),dir='D:\\ASU_Work\\Research\\learn safe mpc\\experiment\\results\\cut_figs\\' +str(num_corr)+'.png')
                num_corr += 1
                time.sleep(0.05)

            # simulation
            x = uav_env.get_curr_state()
            u = controller.control(x, weights=learned_theta)
            uav_env.step(u)
            visualizer.render_update(scale_ratio=2.5)
            time.sleep(0.05)

        else:
            while PAUSE[0]:
                time.sleep(0.05)

    if MSG[0]=='reset':
        MSG[0]=None

    if MSG[0]=='quit':
        break

