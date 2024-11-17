import os
import sys
import casadi as cd
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
sys.path.append(os.path.abspath(os.getcwd()))
from Envs.reacher import Reacher_Env,ReacherModel
from Solvers.OCsolver import ocsolver_v2
from Solvers.Cutter import cutter_v2
from Solvers.MVEsolver import mvesolver
import numpy as np
from matplotlib import pyplot as plt
from utils.Correction import Correction_Agent

# get dynamics, set up step cost and terminal cost
model=ReacherModel()

P_matrix=np.diag([1.0,1.0,2.0,2.0])
Q_matrix=np.diag([0.1,0.1])
T_matrix=5 * P_matrix

dyn_func=model.initDyn(l1=1.0,m1=1.0,l2=1.0,m2=1.0)
step_func = model.get_step_cost(P_matrix,Q_matrix)
term_func = model.get_terminal_cost(T_matrix)

Horizon=30

#construct controller
controller=ocsolver_v2('reacher control')
controller.set_state_param(4,[-2*np.pi,-2*np.pi,-100,-100],[2*np.pi,2*np.pi,100,100])
controller.set_ctrl_param(2,[-1e10]*2,[1e10]*2)
controller.set_dyn(dyn_func)
controller.set_step_cost(step_func)
controller.set_term_cost(term_func)
#controller.construct_graph(horizon=Horizon)
controller.construct_prob(horizon=Horizon)



env = Reacher_Env(1.0,1.0,1.0,1.0)
env.set_init_state(np.array([-np.pi/2,0,0,0]))

ee_traj=[]
for i in range(200):
    x=env.get_curr_state()
    ee_pos = env.get_arm_position(x)
    ee_traj.append(ee_pos)
    u=controller.control(x,weights=0)
    env.step(u)

env.show_animation()
np.save('../Data/Reacher/traj_raw.npy',np.array(ee_traj))