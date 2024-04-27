import os
import sys
import casadi as cd
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
sys.path.append(os.path.abspath(os.getcwd()))
from Envs.pendulum import Pendulum_Env,Pendulum_Model
from Solvers.OCsolver import ocsolver,ocsolver_fast
import numpy as np
from matplotlib import pyplot as plt

# get dynamics, set up step cost and terminal cost
p_model=Pendulum_Model(10,1,1,0.4,0.02)

P_matrix=np.array([[0.5,0],
                   [0,0.05]])
T_matrix=np.array([[1,0],
                   [0,0.025]])

dyn_func=p_model.get_dyn_f()
step_func=p_model.get_step_cost(P_matrix,0.1)
terminal_func=p_model.get_terminal_cost(T_matrix)

# set up safety features
Horizon=40
def generate_phi():
        traj=cd.SX.sym('xi',3*Horizon + 2)
        phi=cd.vertcat(cd.DM(-3.5),traj[3:5])
        return cd.Function('phi',[traj],[phi])

phi_func=generate_phi() #traj: [x,u,x,u,..,x] phi:[phi0, phi1, phi2]
weights=np.array([1,1])

#construct environment
p_env=Pendulum_Env(10,1,1,0.4,0.02)
p_env.set_init_state(np.array([0,0]))
#p_env.set_noise(False)
#construct controller
controller=ocsolver_fast('pendulum control')
controller.set_state_param(2,[-2*np.pi,-100],[2*np.pi,100])
controller.set_ctrl_param(1,[-1e10],[1e10])
controller.set_dyn(dyn_func)
controller.set_step_cost(step_func)
controller.set_term_cost(terminal_func)
#controller.construct_graph(horizon=Horizon)
#controller.set_g(phi_func,weights=weights,gamma=0.05)
controller.construct_prob(horizon=Horizon)
for i in range(200):
    #print(i)
    x=p_env.get_curr_state()
    #test for parameter switching
    u=controller.control(x)
    #print(u)
    #print(type(u))
    if np.sqrt(np.sum((x-np.array([np.pi,0]))**2)) <=0.05:
        print('reached desired position')
        break
    p_env.step(u)
#p_env.show_motion_scatter()
p_env.save_traj(name='free.npy')
plt.figure()
plt.xlabel('alpha')
plt.ylabel('dalpha')
plt.scatter(np.array(p_env.x_traj)[:,0],np.array(p_env.x_traj)[:,1],s=10,label='trajectory under learned params')
plt.plot(np.linspace(0,3.2,100), -0.6*(np.linspace(0,3.2,100)-5),color='r',label='ground truth constraint')
plt.legend()
plt.show()
p_env.show_animation()