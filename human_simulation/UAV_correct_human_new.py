import time
from pynput import keyboard
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
sys.path.append(os.path.abspath(os.getcwd()))
import matplotlib.pyplot as plt
import casadi as cd
from Envs.UAV import UAV_env,UAV_model
from Solvers.OCsolver import ocsolver,ocsolver_fast,ocsolver_inner_Barrier
import numpy as np
from matplotlib import pyplot as plt
from Solvers.Cutter import cutter
from Solvers.MVEsolver import mvesolver
from utils.Correction import Correction_Agent, uav_trans
from utils.RBF import rbf
from utils.Visualize import uav_visualizer
from utils.Keyboard import uav_key_handler
#list for msg passing
PAUSE=[False]
MSG=[None]

#listener for keyboard ops
listener = keyboard.Listener(
    on_press=lambda key: uav_key_handler(key, PAUSE, MSG),
    on_release=None )
listener.start()

# get dynamics, set up step cost and terminal cost
uav_params={'gravity':10,'m':1,'J_B':np.eye(3),'l_w':0.5,'dt':0.1,'c':1}
uav_env=UAV_env(**uav_params)

visualizer=uav_visualizer(uav_env,[0,0,0],[10,10,10])
visualizer.render_init()

uav_model=UAV_model(**uav_params)
dyn_f=uav_model.get_dyn_f()

#r,v,q,w,u
#step_cost_vec=np.array([6,8,100,1,10])*1e-2
#step_cost_vec=np.array([40,60,20,1,10])*1e-3
step_cost_vec=np.array([40,6,40,100,10])*1e-3
step_cost_vec=np.array([50,10,5,10,15])*1e-3
step_cost_f=uav_model.get_step_cost(step_cost_vec,target_pos=np.array([9,9,5]))
#term_cost_vec=np.array([2,6,100,0.1])*1e-1
#term_cost_vec=np.array([30,30,15,2])*1e-2
term_cost_vec=np.array([20,5,15,100])*1e-2
term_cost_vec=np.array([50,6,40,50])*1e-1
term_cost_f=uav_model.get_terminal_cost(term_cost_vec,target_pos=np.array([9,9,5]))

# set up safety features
Horizon=5 #25
Gamma=0.1

#phi_func = generate_phi_x_2()
#weights_init = (hypo_lbs_2d+hypo_ubs_2d)/2
#ctrl
controller=ocsolver_fast('uav control')
controller.set_state_param(13,None,None)
controller.set_ctrl_param(4,[-1e10,-1e10,-1e10,-1e10],[1e10,1e10,1e10,1e10])
controller.set_dyn(dyn_f)
controller.set_step_cost(step_cost_f)
controller.set_term_cost(term_cost_f)
#controller.construct_graph(horizon=Horizon)
#controller.set_g(phi_func,weights=weights_init,gamma=Gamma)
controller.construct_prob(horizon=Horizon)


init_r = np.array([1,8,1]).reshape(-1,1)
init_v = np.zeros((3,1))
init_q = np.reshape(np.array([1,0,0,0]),(-1,1))
#print(Quat_Rot(init_q))
init_w_B = np.zeros((3,1))
init_x=np.concatenate([init_r,init_v,init_q,init_w_B],axis=0)
init_x[0]=np.random.uniform(0.5,2.5)
#init_x[0]=1
init_x[1]=np.random.uniform(0.5,8.5)
#init_x[1]=1
print('init',init_x.T)
while True:
    uav_env.set_init_state(init_x) 
    for i in range(200):
        if not PAUSE[0]:
            # simulation
            x=uav_env.get_curr_state()
            u=controller.control(x)
            uav_env.step(u)
            visualizer.render_update(scale_ratio=1.5)
            time.sleep(0.05) 

        else:
            while PAUSE[0]:
                time.sleep(0.05)