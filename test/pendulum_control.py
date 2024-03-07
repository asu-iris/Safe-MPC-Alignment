import os
import sys
from casadi import *
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
sys.path.append(os.path.abspath(os.getcwd()))
from Envs.pendulum import Pendulum_Env,Pendulum_Model
from OCsolver.OCsolver import ocsolver,ocsolver_fast
import numpy as np
from matplotlib import pyplot as plt

# get dynamics, set up step cost and terminal cost
p_model=Pendulum_Model(10,1,1,0.4,0.05)

P_matrix=np.array([[10,0],
                   [0,1]])
T_matrix=np.array([[20,0],
                   [0,1]])

dyn_func=p_model.get_dyn_f()
step_func=p_model.get_step_cost(P_matrix,0.05)
terminal_func=p_model.get_terminal_cost(T_matrix)

# set up safety features
Horizon=20
def generate_phi():
        traj=SX.sym('xi',3*Horizon + 2)
        phi=vertcat(SX(-4.2),traj[3:5])
        return Function('phi',[traj],[phi])

phi_func=generate_phi() #traj: [x,u,x,u,..,x] phi:[phi0, phi1, phi2]
weights=np.array([1,1])

#construct environment
p_env=Pendulum_Env(10,1,1,0.4,0.05)
p_env.set_init_state(np.array([0,0]))
#p_env.set_noise(False)
#construct controller
controller=ocsolver_fast('pendulum control')
controller.set_state_param(2,[-2*np.pi,-100],[2*np.pi,100])
controller.set_ctrl_param(1,[-10],[10])
controller.set_dyn(dyn_func)
controller.set_step_cost(step_func)
controller.set_term_cost(terminal_func)
controller.set_g(phi_func,weights=weights,gamma=0.01)
controller.construct_graph(horizon=Horizon)
for i in range(80):
    x=p_env.get_curr_state()
    u=controller.control(x)
    #print(u)
    #print(type(u))
    p_env.step(u)
p_env.show_motion_scatter()
plt.figure()
plt.scatter(np.array(p_env.x_traj)[:,0],np.array(p_env.x_traj)[:,1])
plt.plot(np.linspace(0,3.2,100), -(np.linspace(0,3.2,100)-4.2))
plt.show()
p_env.show_animation()