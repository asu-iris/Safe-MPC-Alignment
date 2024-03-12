import os
import sys
import casadi as cd
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
sys.path.append(os.path.abspath(os.getcwd()))
from Envs.pendulum import Pendulum_Env,Pendulum_Model
from Solvers.OCsolver import ocsolver,ocsolver_fast,ocsolver_inner_Barrier
import numpy as np
from matplotlib import pyplot as plt

# get dynamics, set up step cost and terminal cost
#0.05
p_model=Pendulum_Model(10,1,1,0.4,0.01)

P_matrix=np.array([[0.5,0],
                   [0,0.05]])
T_matrix=np.array([[1,0],
                   [0,0.025]])



dyn_func=p_model.get_dyn_f()
step_func=p_model.get_step_cost(P_matrix,0.1)
terminal_func=p_model.get_terminal_cost(T_matrix)

# set up safety features
Horizon=80
Gamma=0.01
def generate_phi():
        traj=cd.SX.sym('xi',3*Horizon + 2)
        phi=cd.vertcat(cd.DM(-3),traj[3:5])
        return cd.Function('phi',[traj],[phi])

phi_func=generate_phi() #traj: [x,u,x,u,..,x] phi:[phi0, phi1, phi2]
weights_init=np.array([0.6,1])
#weights_H=np.array([0.6,1])

#construct environment
p_env=Pendulum_Env(10,1,1,0.4,0.01)
p_env.set_init_state(np.array([0,0]))
#p_env.set_noise(False)

#construct controller
controller_1=ocsolver_fast('pendulum control')
controller_1.set_state_param(2,[-2*np.pi,-100],[2*np.pi,100])
controller_1.set_ctrl_param(1,[-6],[6])
controller_1.set_dyn(dyn_func)
controller_1.set_step_cost(step_func)
controller_1.set_term_cost(terminal_func)
#controller.construct_graph(horizon=Horizon)
controller_1.set_g(phi_func,weights=weights_init,gamma=Gamma)
controller_1.construct_prob(horizon=Horizon)

#construct controller
controller_2=ocsolver_inner_Barrier('pendulum control (inner)')
controller_2.set_state_param(2,[-2*np.pi,-100],[2*np.pi,100])
controller_2.set_ctrl_param(1,[-6],[6])
controller_2.set_dyn(dyn_func)
controller_2.set_step_cost(step_func)
controller_2.set_term_cost(terminal_func)
#controller.construct_graph(horizon=Horizon)
controller_2.set_g(phi_func,weights=weights_init,gamma=Gamma)
controller_2.construct_prob(horizon=Horizon)

p_env.set_init_state(np.array([0,0]))
for i in range(300):
    x=p_env.get_curr_state()
    if np.sqrt(np.sum((x-np.array([np.pi,0]))**2)) <=0.04:
        print('reached desired position')
        break
    print('demo step',i)
    u_1=controller_1.control(x)
    u_2=controller_2.control(x)
    controller_1.reset_warmstart()
    controller_2.reset_warmstart()
    print(u_1-u_2,u_1,(u_1-u_2)/u_1)
    p_env.step(u_2)

p_env.show_motion_scatter()
plt.figure()
plt.xlabel('alpha')
plt.ylabel('dalpha')
plt.scatter(np.array(p_env.x_traj)[:,0],np.array(p_env.x_traj)[:,1],s=10,label='trajectory under learned params')
plt.plot(np.linspace(0,3.2,100), -0.6*(np.linspace(0,3.2,100)-5),color='r',label='ground truth constraint')
plt.legend()
plt.show()
p_env.show_animation()