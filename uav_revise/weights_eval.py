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
from poly_feature import gen_poly_feats,gen_poly_feats_single_v,gen_poly_feats_v_full
from scipy.spatial.transform import Rotation as R
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
step_cost_vec = np.array([5.0, 400, 1, 5, 1]) * 1e-2
step_cost_f = uav_model.get_step_cost_param(step_cost_vec)
term_cost_vec = np.array([10, 6, 1, 5]) * 1e0
term_cost_f = uav_model.get_terminal_cost_param(term_cost_vec)
#####################################################################################
#additional cost
phi_func_poly = gen_poly_feats(Horizon=Horizon, con_idx=4, bias = -1)
constraint_weights = np.load("../Data/uav_revise/theta_poly_3.npy")

def generate_add_func(weights,Horizon=Horizon):
    x_dim = 13
    u_dim = 4
    traj = cd.SX.sym('xi', (x_dim + u_dim) * Horizon + x_dim)
    phi = phi_func_poly(traj)
    feature_weights = cd.vertcat(1, weights)
    g = cd.dot(phi, feature_weights)
    ln_g =  - 60 * cd.log(-g)
    add_func = cd.Function('mtimes', [traj], [ln_g])   
    return add_func

add_func = generate_add_func(constraint_weights, Horizon=Horizon)
#########################################################################################
#constraint vector
phi_func = gen_poly_feats_v_full(Horizon=Horizon, bias=-1, start_idx=1, end_idx=10)
phi_func_single = gen_poly_feats_single_v(bias=-1)

#########################################################################################
controller = ocsolver_v3('uav control')
controller.set_state_param(13, None, None)
controller.set_ctrl_param(4, [-1e10, -1e10, -1e10, -1e10], [1e10, 1e10, 1e10, 1e10])
controller.set_dyn(dyn_f)
controller.set_step_cost(step_cost_f)
controller.set_term_cost(term_cost_f)
controller.set_additional_cost(add_func)
controller.set_g(phi_func, gamma=100.0)
controller.construct_prob(horizon=Horizon)
# init_theta = learned_theta = (hypo_lbs + hypo_ubs) / 2
######################################################################################

hypo_lbs = -0.0 * np.ones(8)
hypo_ubs = 0.6 * np.ones(8)

hb_calculator = cutter_v3('uav cut')
hb_calculator.from_controller(controller)
hb_calculator.construct_graph(horizon=Horizon)

mve_calc = mvesolver('uav_mve', 8)
mve_calc.set_init_constraint(hypo_lbs, hypo_ubs)  # Theta_0


init_r = np.array([0.0,0.0,5.0]).reshape(-1, 1)
init_v = np.zeros((3, 1))
init_q = np.reshape(np.array([1, 0, 0, 0]), (-1, 1))
# print(Quat_Rot(init_q))
init_w_B = np.zeros((3, 1))
init_x = np.concatenate([init_r, init_v, init_q, init_w_B], axis=0)

target_r = np.array([15.0,0.0,10.0])
target_x = np.concatenate([target_r.reshape(-1,1), init_v, init_q, init_w_B], axis=0)

def eval_weights(weights, horizon=Horizon):
    """
    Evaluate the weights using the MVE solver.
    """
    uav_env.set_init_state(init_x)
    controller.reset_warmstart()
    traj = [uav_env.get_curr_state()]
    for i in range(100):
        x = uav_env.get_curr_state()
        
        print("current constraint vec", phi_func_single(x))
   
        uav_v = x[3:6].flatten()
        print("uav_v", uav_v, np.linalg.norm(uav_v))

        u = controller.control(x, weights= weights, target_r=target_r)
        print("predicted vel",controller.opt_traj[2* 17 + 3:2*17+6])
        # controller.warm_start_sol[7:10] = 0
        print(u)

        uav_env.step(u)
        traj.append(uav_env.get_curr_state())
    
    return np.array(traj)

def eval_weights_long(weights, horizon=Horizon):
    """
    Evaluate the weights using the MVE solver.
    """
    viewer = mujoco.viewer.launch_passive(uav_env.model,uav_env.data)
    uav_env.set_init_state(init_x)
    controller.reset_warmstart()
    traj = [uav_env.get_curr_state()]
    for i in range(10000):
        x = uav_env.get_curr_state()
        
        print("current constraint vec", phi_func_single(x))
   
        uav_v = x[3:6].flatten()
        print("uav_v", uav_v, np.linalg.norm(uav_v))

        u = controller.control(x, weights= weights, target_r=target_r)
        print("predicted vel",controller.opt_traj[2* 17 + 3:2*17+6])
        # controller.warm_start_sol[7:10] = 0
        print(u)

        uav_env.step(u)
        traj.append(uav_env.get_curr_state())
        viewer.sync()
    
    


weights = np.load("../Data/uav_revise/weights_vel_long/weights_init.npy")
traj = eval_weights(weights, horizon=Horizon)
traj_list = [np.linalg.norm(traj[:,3:6],axis=1)]

for i in range(1,18):
    weights = np.load("../Data/uav_revise/weights_vel_long/weights_{}.npy".format(i))
    traj = eval_weights(weights, horizon=Horizon)
    traj_list.append(np.linalg.norm(traj[:,3:6],axis=1))

# Visualize the trajectories
from matplotlib import pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(traj_list[0], label='Initial Weights', linewidth=2.5)
for i in [6,9,15,17]:
    plt.plot(traj_list[i], label=f'Weights {i}', linewidth=2.5)
plt.title('Velocity Norms for Different Weights')
plt.xlabel('Time Step')
plt.ylabel('Velocity Norm')
#horizontal line at 0.5
plt.axhline(y=0.4, color='black', linestyle='dotted',linewidth=3, label='Correction Threshold (0.4)')
plt.axhline(y=0.45, color='black', linestyle='dashed',linewidth=3, label='Stop Threshold (0.45)')

plt.legend()
plt.grid()
plt.show()

eval_weights_long(weights, horizon=Horizon)