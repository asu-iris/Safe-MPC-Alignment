import os
import sys
import casadi as cd
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
sys.path.append(os.path.abspath(os.getcwd()))
from Envs.pendulum import Pendulum_Env,Pendulum_Model
from Solvers.OCsolver import ocsolver,ocsolver_fast
from Solvers.Cutter import cutter
from Solvers.MVEsolver import mvesolver
from utils.Correction import Correction_Agent
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
weights_init=np.array([0,0])
weights_H=np.array([0.6,1])

#construct environment
p_env=Pendulum_Env(10,1,1,0.4,0.01)
p_env.set_init_state(np.array([0,0]))
#p_env.set_noise(False)
#construct correction agent
agent=Correction_Agent('dummy')
agent.set_state_dim(2)
agent.set_ctrl_dim(1)
agent.set_dyn(dyn_func)
agent.set_step_cost(step_func)
agent.set_term_cost(terminal_func)
agent.set_g(phi_func,weights=weights_H,gamma=Gamma)
agent.set_threshold(-0.2) #-0.5
agent.set_p(0.5)
agent.construct_graph(horizon=Horizon)

#construct controller
controller=ocsolver_fast('pendulum control')
controller.set_state_param(2,[-2*np.pi,-100],[2*np.pi,100])
controller.set_ctrl_param(1,[-6],[6])
controller.set_dyn(dyn_func)
controller.set_step_cost(step_func)
controller.set_term_cost(terminal_func)
#controller.construct_graph(horizon=Horizon)
controller.set_g(phi_func,weights=weights_init,gamma=Gamma)
controller.construct_prob(horizon=Horizon)

#construct cutter
hb_calculator=cutter('pendulum cut')
hb_calculator.set_state_dim(2)
hb_calculator.set_ctrl_dim(1)
hb_calculator.set_dyn(dyn_func)
hb_calculator.set_step_cost(step_func)
hb_calculator.set_term_cost(terminal_func)
hb_calculator.set_g(phi_func,weights=weights_init,gamma=Gamma)
hb_calculator.construct_graph(horizon=Horizon)

#construct MVESolver
mve_calc=mvesolver('pendulum_mve',2)
lbs=np.array([-10,-10]) #-6
ubs=np.array([10,10])
mve_calc.set_init_constraint(lbs, ubs) #Theta_0

learned_theta=np.array(weights_init)
#learning logs
theta_log=[np.array(weights_init)]
error_log=[np.linalg.norm(weights_init-weights_H)]

d_0,C_0=mve_calc.solve()
v_0=np.log(np.linalg.det(C_0))
volume_log=[v_0]

EPISODE=0
corr_num=0
termination_flag=False
while not termination_flag:
    print('episode',EPISODE)
    p_env.set_init_state(np.array([0,0]))
    for i in range(300):
        x=p_env.get_curr_state()
        if np.sqrt(np.sum((x-np.array([np.pi,0]))**2)) <=0.15:
            print('reached desired position')
            break
        #print(i)
        u=controller.control(x)
        agent_output=agent.act(controller.opt_traj_t)
        if agent_output==None:
            print('emergency stop')
            break
        elif type(agent_output)==bool:
            pass
        else:
            h,b,h_phi,b_phi=hb_calculator.calc_planes(x,controller.opt_traj_t,np.sign(agent_output))
            print('cutting plane calculated')
            print('h',h)
            print('b',b)
            print('diff', h.T @ learned_theta - b)
            print('h_phi',h_phi)
            print('b_phi',b_phi)

            mve_calc.add_constraint(h,b[0])
            mve_calc.add_constraint(h_phi,b_phi[0])
            learned_theta,C=mve_calc.solve()
            difference=np.linalg.norm(learned_theta-weights_H)
            vol=np.log(np.linalg.det(C))
            print('leanred safety param',learned_theta)
            theta_log.append(learned_theta)
            print('difference', difference)
            error_log.append(difference)
            print('volume', vol)
            volume_log.append(vol)
            #mve_calc.draw(C,learned_theta,weights_H)
            if difference<0.08:
                print("converged! Final Result: ",learned_theta)
                termination_flag=True
                break
            #set new constraints
            controller.set_g(phi_func,weights=learned_theta,gamma=Gamma)
            controller.construct_prob(horizon=Horizon)

            hb_calculator.set_g(phi_func,weights=learned_theta,gamma=Gamma)
            hb_calculator.construct_graph(horizon=Horizon)
            corr_num+=1
        p_env.step(u)
    EPISODE+=1
#plot learning process
plt.figure()
plt.title("learning error")
plt.plot(error_log,label='HSC error')
plt.show()
plt.figure()
plt.title("MVE volume")
plt.plot(volume_log,label='MVE volume')
plt.show()
#perform one round with converged params
p_env.set_init_state(np.array([0,0]))
controller.set_g(phi_func,weights=learned_theta,gamma=Gamma)
controller.construct_prob(horizon=Horizon)
controller.reset_warmstart()
print('demo with learned params')
for i in range(300):
    x=p_env.get_curr_state()
    if np.sqrt(np.sum((x-np.array([np.pi,0]))**2)) <=0.04:
        print('reached desired position')
        break
    print('demo step',i)
    u=controller.control(x)
    p_env.step(u)
p_env.show_motion_scatter()
plt.figure()
plt.xlabel('alpha')
plt.ylabel('dalpha')
plt.scatter(np.array(p_env.x_traj)[:,0],np.array(p_env.x_traj)[:,1],s=10,label='trajectory under learned params')
plt.plot(np.linspace(0,3.2,100), -0.6*(np.linspace(0,3.2,100)-5),color='r',label='ground truth constraint')
plt.legend()
plt.show()
p_env.show_animation()

