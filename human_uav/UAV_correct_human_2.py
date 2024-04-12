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
from matplotlib import cm
from Solvers.Cutter import cutter_v2
from Solvers.MVEsolver import mvesolver
from utils.Correction import Correction_Agent, uav_trans
from utils.RBF import generate_phi_rbf,gen_eval_rbf

from utils.Visualize import uav_visualizer
from utils.Keyboard import uav_key_handler,key_interface,remove_conflict

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
Horizon = 10  # 25
Gamma = 1 #10

rbf_mode='gau_rbf_xy'
phi_func = generate_phi_rbf(Horizon,mode=rbf_mode)
theta_dim = 25
hypo_lbs = -5 * np.ones(theta_dim)
hypo_ubs = 0 * np.ones(theta_dim)

######################################################################################
# get dynamics, set up step cost and terminal cost
uav_params = {'gravity': 10, 'm': 0.05, 'J_B': 0.01 * np.eye(3), 'l_w': 0.5, 'dt': 0.1, 'c': 1}
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
init_theta = learned_theta = (hypo_lbs + hypo_ubs) / 2
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
remove_conflict(plt.rcParams)
#print(plt.rcParams)
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
                human_corr=key_interface(MSG)

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

plt.ioff()    
print('finish')
eval_func_1=gen_eval_rbf(init_theta,mode=rbf_mode)
eval_func_2=gen_eval_rbf(learned_theta,mode=rbf_mode)
X_eval = np.linspace(0, 10, 50)
Y_eval = np.linspace(0, 10, 50)
grid_x_raw, grid_y_raw = np.meshgrid(X_eval, Y_eval)
grid_x = grid_x_raw.reshape(-1, 1)
grid_y = grid_y_raw.reshape(-1, 1)
points = np.concatenate([grid_x, grid_y], axis=1)
z_1=eval_func_1(points.T)
z_2=eval_func_2(points.T)

fig=plt.figure()
ax=plt.axes(projection='3d')
ax.plot_trisurf(points[:,0].flatten(),points[:,1].flatten(),z_1.full().flatten(),cmap=cm.coolwarm)
ax.set_zlim(-5, 1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('original')
#ax.plot_surface(grid_x,grid_y,z.full().reshape(-1,1),cmap=cm.coolwarm)
fig=plt.figure()
ax=plt.axes(projection='3d')
ax.set_zlim(-5, 1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.plot_trisurf(points[:,0].flatten(),points[:,1].flatten(),z_2.full().flatten(),cmap=cm.coolwarm)
ax.set_title('learned')

fig=plt.figure()
ax=plt.axes()
levels=np.arange(-6,3,0.02)
plt.contourf(grid_x_raw,grid_y_raw,z_1.full().reshape(50,50),levels,cmap=plt.get_cmap('Spectral'))
plt.colorbar(label='Function value')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('original')


fig=plt.figure()
ax=plt.axes()
levels=np.arange(-6,3,0.02)
plt.contourf(grid_x_raw,grid_y_raw,z_2.full().reshape(50,50),levels,cmap=plt.get_cmap('Spectral'))
plt.colorbar(label='Function value')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('learned')
#plt.colorbar(label='Function value')

plt.show()
