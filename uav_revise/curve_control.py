import time
from pynput import keyboard
import sys
import os
import mujoco.viewer

sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
sys.path.append(os.path.abspath(os.getcwd()))
from utils.Keyboard import uav_key_handler
import matplotlib.pyplot as plt
import casadi as cd
from Envs.UAV import UAV_env_mj, UAV_model

from Solvers.OCsolver import  ocsolver_v3
from Solvers.Cutter import  cutter_v3
from Solvers.MVEsolver import mvesolver
import numpy as np

from utils.RBF import generate_phi_rbf, gen_eval_rbf

from utils.Visualize_mj import uav_visualizer_mj_v4
from poly_feature import gen_poly_feats_single,gen_poly_feats

Horizon = 15  # 25
uav_params = {'gravity': 9.8, 'm': 0.1, 'J_B': 0.01 * np.eye(3), 'l_w': 1.2, 'dt': 0.1, 'c': 1}
filepath = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                        'mujoco_uav', 'bitcraze_crazyflie_2',
                        'scene_revise_geom.xml')
print('path', filepath)
uav_env = UAV_env_mj(filepath, lock_flag=True)
uav_model = UAV_model(**uav_params)
dyn_f = uav_model.get_dyn_f()
######################################################################################

# r,v,q,w,u
step_cost_vec = np.array([0.05, 400, 1, 5, 0.01]) * 1e-2
step_cost_f = uav_model.get_step_cost_param(step_cost_vec)
term_cost_vec = np.array([10, 6, 1, 5]) * 1e0
term_cost_f = uav_model.get_terminal_cost_param(term_cost_vec)
#####################################################################################
#constraint vector
phi_func = gen_poly_feats(Horizon=Horizon, con_idx=4, bias = -1)
phi_func_single = gen_poly_feats_single(bias = -1)

#########################################################################################
controller = ocsolver_v3('uav control')
controller.set_state_param(13, None, None)
controller.set_ctrl_param(4, [-1e10, -1e10, -1e10, -1e10], [1e10, 1e10, 1e10, 1e10])
controller.set_dyn(dyn_f)
controller.set_step_cost(step_cost_f)
controller.set_term_cost(term_cost_f)
controller.set_g(phi_func, gamma=60.0)
controller.construct_prob(horizon=Horizon)
# init_theta = learned_theta = (hypo_lbs + hypo_ubs) / 2
######################################################################################

hypo_lbs = -80 * np.ones(12)
hypo_ubs = 200 * np.ones(12)

hb_calculator = cutter_v3('uav cut')
hb_calculator.from_controller(controller)
hb_calculator.construct_graph(horizon=Horizon)

mve_calc = mvesolver('uav_mve', 12)
mve_calc.set_init_constraint(hypo_lbs, hypo_ubs)  # Theta_0
#########################################################################################

#correction interface
y_thrust = np.array([0, 1, 0, -1])
z_thrust = np.array([1, 1, 1, 1])

viewer = mujoco.viewer.launch_passive(uav_env.model,uav_env.data)

init_r = np.array([0.0,0.0,5.0]).reshape(-1, 1)
init_v = np.zeros((3, 1))
init_q = np.reshape(np.array([1, 0, 0, 0]), (-1, 1))
# print(Quat_Rot(init_q))
init_w_B = np.zeros((3, 1))
init_x = np.concatenate([init_r, init_v, init_q, init_w_B], axis=0)

target_r = np.array([15.0,0.0,10.0])
target_x = np.concatenate([target_r.reshape(-1,1), init_v, init_q, init_w_B], axis=0)

#init cut
phi_init = phi_func_single(init_x)
phi_target = phi_func_single(target_x)
mve_calc.add_constraint(phi_init[1:], - phi_init[0])
mve_calc.add_constraint(phi_target[1:], - phi_target[0])
weights,C = mve_calc.solve()
print(weights)
input()

curve_points = np.load("../Data/uav_revise/curve_points.npy")
circle_points = np.load("../Data/uav_revise/circle_points.npy")

traj_cnt = 0
while True:
    rand_init = init_x.copy()
    # phi_init = phi_func_single(rand_init)
    # mve_calc.add_constraint(phi_init[1:], - phi_init[0])
    rand_init[1:3]+= np.random.randn(2,1)*0.05
    uav_env.set_init_state(init_x)
    traj = []
    for i in range(500):
        x = uav_env.get_curr_state()
        
        print("current constraint vec", phi_func_single(x))

        uav_p = x[0:3].flatten()
        dists = np.linalg.norm(circle_points - uav_p,axis = 2)
        point_id = np.unravel_index(np.argmin(dists), dists.shape)
        dist = dists[point_id]
        # direction_con = uav_p - circle_points[point_id]
        # direction_con /= np.linalg.norm(direction_con)

        # direction_target = target_r - uav_p
        # direction_target /= np.linalg.norm(direction_target)

        direction = curve_points[point_id[0]]-uav_p

        u = controller.control(x, weights=weights, target_r=target_r)
        print(u)
        

        if dist<=0.7:
            corr = y_thrust * direction[1] + z_thrust*direction[2]
            # corr = y_thrust * 1
            print("direction",direction)
            corr_e = np.concatenate([corr.reshape(-1, 1), np.zeros((4 * (Horizon - 1), 1))])
            h, b, h_phi, b_phi = hb_calculator.calc_planes(weights, x, controller.opt_traj,
                                                                    human_corr=corr_e,
                                                                    target_r=target_r)
            print(h, b)
            print(h_phi, b_phi)
            mve_calc.add_constraint(h, b[0])
            mve_calc.add_constraint(h_phi, b_phi[0])

            weights, C = mve_calc.solve()
            print("learned", weights)
            break

        uav_env.step(u)
        traj.append(uav_env.get_curr_state())
        viewer.sync()
        # time.sleep(0.05)

    np.save(os.path.join('../Data/uav_revise/traj_poly', 'traj_{}.npy'.format(traj_cnt)),np.array(traj))
    traj_cnt+=1
    input()