"""
The code for watching the performance of the algorithm in simulated pendulum experiment.
python pendulum_correct_v2.py
"""

import os
import sys
import casadi as cd
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
sys.path.append(os.path.abspath(os.getcwd()))
from Envs.pendulum import Pendulum_Env,Pendulum_Model
from Solvers.OCsolver import ocsolver_v2
from Solvers.Cutter import cutter_v2
from Solvers.MVEsolver import mvesolver
from utils.Correction import Correction_Agent
import numpy as np
from matplotlib import pyplot as plt

# get dynamics, set up step cost and terminal cost
#0.05
p_model=Pendulum_Model(10,1,1,0.4,0.02)

P_matrix=np.array([[0.0,0],
                   [0,0.0]])
T_matrix=np.array([[25,0],
                   [0,10]])



dyn_func=p_model.get_dyn_f()
step_func=p_model.get_step_cost(P_matrix,0.1)
terminal_func=p_model.get_terminal_cost(T_matrix)

# set up safety features
Horizon=20
Gamma=0.1
def generate_phi():
        traj=cd.SX.sym('xi',3*Horizon + 2)
        phi=cd.vertcat(cd.DM(-3),traj[3:5])
        return cd.Function('phi',[traj],[phi])

phi_func=generate_phi() #traj: [x,u,x,u,..,x] phi:[phi0, phi1, phi2]

lbs=np.array([-6,-6]) #-6
ubs=np.array([2,2])

weights_init=(lbs+ubs)/2
weights_H=np.array([0.6,1])

#construct environment
p_env=Pendulum_Env(10,1,1,0.4,0.02)
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
#agent.set_threshold(-0.1) #-0.5
agent.set_threshold(-0.25) #-0.5
agent.set_p(0.3)
agent.construct_graph(horizon=Horizon)

#construct controller
controller=ocsolver_v2('pendulum control')
controller.set_state_param(2,[-2*np.pi,-100],[2*np.pi,100])
controller.set_ctrl_param(1,[-1e10],[1e10])
controller.set_dyn(dyn_func)
controller.set_step_cost(step_func)
controller.set_term_cost(terminal_func)
#controller.construct_graph(horizon=Horizon)
controller.set_g(phi_func,gamma=Gamma)
controller.construct_prob(horizon=Horizon)

#construct cutter
hb_calculator=cutter_v2('pendulum cut')
hb_calculator.from_controller(controller)
hb_calculator.construct_graph(horizon=Horizon)

#construct MVESolver
mve_calc=mvesolver('pendulum_mve',2)

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
    # random init
    init_state=np.array([0,0])
    init_state[0] += np.random.uniform(0,2*np.pi/3)
    init_state[1] += np.random.uniform(0,3)
    p_env.set_init_state(init_state)
    for i in range(200):
        x=p_env.get_curr_state()
        if np.sqrt(np.sum((x-np.array([np.pi,0]))**2)) <=0.15:
            print('reached desired position')
            break
        #print(i)
        u=controller.control(x,weights=learned_theta)
        agent_output=agent.act(controller.opt_traj)
        #print(agent_output)
        if agent_output is None:
            print('emergency stop')
            break
        elif type(agent_output)==bool:
            pass
        else:
            h,b,h_phi,b_phi=hb_calculator.calc_planes(learned_theta,x,controller.opt_traj,np.sign(agent_output))
            print('cutting plane calculated')
            print('h',h)
            print('b',b)
            print('diff', h.T @ learned_theta - b)
            print('h_phi',h_phi)
            print('b_phi',b_phi)

            mve_calc.add_constraint(h,b[0])
            mve_calc.add_constraint(h_phi,b_phi[0])
            learned_theta,C=mve_calc.solve()
            #difference=np.linalg.norm(learned_theta-weights_H)
            difference=learned_theta-weights_H
            vol=np.log(np.linalg.det(C))
            print('leanred safety param',learned_theta)
            theta_log.append(learned_theta)
            print('difference', difference)
            error_log.append(np.linalg.norm(difference))
            print('volume', vol)
            volume_log.append(vol)
            #mve_calc.draw(C,learned_theta,weights_H)
            #if np.max(np.abs(difference))<0.04:
            if np.linalg.norm(difference) < 0.02:
                print("converged! Final Result: ",learned_theta)
                termination_flag=True
                break
            
            corr_num+=1
        p_env.step(u)
    #p_env.show_animation()
    EPISODE+=1

print(theta_log)
#plot learning process
plt.figure()
plt.title("learning error")
plt.plot(error_log,label='HSC error')
plt.show()
plt.figure()
plt.title("MVE volume")
plt.plot(volume_log,label='MVE volume')
plt.show()

# plt.rcParams.update({
#     "text.usetex": True
# })

def eval_theta(theta,controller,id=0):
    #perform one round with converged params
    p_env.set_init_state(np.array([0,0]))
    controller.reset_warmstart()
    print('demo with learned params')
    for i in range(200):
        x=p_env.get_curr_state()
        if np.sqrt(np.sum((x-np.array([np.pi,0]))**2)) <=0.05:
            print('reached desired position')
            break
        #print('demo step',i)
        u=controller.control(x,weights=theta)
        p_env.step(u)

    p_env.save_traj(name='constraint.npy')

    #p_env.show_motion_scatter()
    plt.figure()
    #plt.title("Trajectory Solved at Correction "+str(id),fontsize=20)
    plt.xlabel('Angle',fontsize=28)
    plt.ylabel('Vel',fontsize=28)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.xlim(-0.5,3.5)
    plt.ylim(-0.5,5)
    plt.scatter(np.array(p_env.x_traj)[:,0],np.array(p_env.x_traj)[:,1],s=30,label='Trajectory')
    plt.plot(np.linspace(0,3.2,100), -0.6*(np.linspace(0,3.2,100)-5),linewidth=5.0,color='r',label='Ground Truth')
    if id > 0:
        plt.plot(np.linspace(0,3.2,100), (3-theta[0]*np.linspace(0,3.2,100))/theta[1],linewidth=5.0,color='orange',label='Learned Constraint')
    plt.legend(fontsize=18)
    plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.grid()
    plt.savefig('../Data/pendulum/traj_'+str(id)+'.png')
    plt.show(block=False)

for i in range(0,len(theta_log),3):
    print(i)
    eval_theta(theta_log[i],controller,id=i)
eval_theta(theta_log[-1],controller,id=len(theta_log)-1)

def plot_theta_route(theta_log,lbs,ubs):
    plt.figure()
    plt.xlabel('dim_0')
    plt.ylabel('dim_1')
    plt.xlim(lbs[0]-1,ubs[0]+1)
    plt.ylim(lbs[1]-1,ubs[1]+1)

    theta_arr=np.array(theta_log)
    plt.scatter(theta_arr[:,0],theta_arr[:,1])
    for i in range(len(theta_log) - 1):
        plt.quiver(theta_arr[i,0], theta_arr[i,1], theta_arr[i+1,0] - theta_arr[i,0], theta_arr[i+1,1] - theta_arr[i,1],
                    angles='xy', scale_units='xy', scale=1,width=0.003, headwidth=3)

    plt.show(block=False)

#plot_theta_route(theta_log,lbs,ubs)
input()
#p_env.show_animation()

