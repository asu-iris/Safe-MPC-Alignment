from Envs.pendulum import Pendulum_Env,Pendulum_Model
from OCsolver.OCsolver import ocsolver

import numpy as np

p_model=Pendulum_Model(10,1,1,0.4,0.05)

P_matrix=np.array([[10,0],
                   [0,1]])
T_matrix=np.array([[20,0],
                   [0,1]])

dyn_func=p_model.get_dyn_f()
step_func=p_model.get_step_cost(P_matrix,0.05)
terminal_func=p_model.get_terminal_cost(T_matrix)

p_env=Pendulum_Env(10,1,1,0.4,0.05)
p_env.set_init_state(np.array([0,0]))
controller=ocsolver('pendulum control')
controller.set_state_param(2,[-2*np.pi,-100],[2*np.pi,100])
controller.set_ctrl_param(1,[-10],[10])
controller.set_dyn(dyn_func)
controller.set_step_cost(step_func)
controller.set_term_cost(terminal_func)
for i in range(80):
    x=p_env.get_curr_state()
    u=controller.control(x,20)
    #print(u)
    #print(type(u))
    p_env.step(u)
p_env.show_motion_scatter()
p_env.show_animation()