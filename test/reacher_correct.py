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
import time
# get dynamics, set up step cost and terminal cost
model=ReacherModel()

P_matrix=np.diag([1.0,1.0,2.0,2.0])
Q_matrix=np.diag([0.1,0.1])
T_matrix=5 * P_matrix

dyn_func=model.initDyn(l1=1.0,m1=1.0,l2=1.0,m2=1.0)
step_func = model.get_step_cost(P_matrix,Q_matrix)
term_func = model.get_terminal_cost(T_matrix)

Horizon=30
Gamma=2.0 #0.1

def generate_phi():
    traj=cd.SX.sym('xi',6*Horizon + 4)
    theta_1 = traj[18]
    theta_2 = traj[19]

    x1 = 1 * cd.cos(theta_1)
    y1 = 1 * cd.sin(theta_1)
    x2 = 1 * cd.cos(theta_1 + theta_2) + x1
    y2 = 1 * cd.sin(theta_1 + theta_2) + y1 

    phi_1 = cd.tanh(x2*y2)
    phi_2 = cd.tanh(x2**2)
    phi_3 = cd.tanh(y2**2)
    phi_4 = cd.tanh(x2)
    phi_5 = cd.tanh(y2)
    phi=cd.vertcat(cd.DM(-1),phi_1,phi_2,phi_3,phi_4,phi_5)
    return cd.Function('phi',[traj],[phi])

def generate_phi_xy():
    x2 = cd.SX.sym('x2')
    y2 = cd.SX.sym('y2')

    phi_1 = cd.tanh(x2*y2)
    phi_2 = cd.tanh(x2**2)
    phi_3 = cd.tanh(y2**2)
    phi_4 = cd.tanh(x2)
    phi_5 = cd.tanh(y2)
    phi=cd.vertcat(cd.DM(-1),phi_1,phi_2,phi_3,phi_4,phi_5)
    return  cd.Function('phi_xy',[x2,y2],[phi])

phi_func=generate_phi()
phi_func_xy=generate_phi_xy()
lbs=np.array([-3,-3,-3,-3,-3]) #-6
ubs=np.array([3,3,3,3,3])

weights_init=(lbs+ubs)/2
weights_H=np.array([0.6, 0.7,-0.4,0.4,-0.8])*1.3

agent=Correction_Agent('dummy')
agent.set_state_dim(4)
agent.set_ctrl_dim(2)
agent.set_dyn(dyn_func)
agent.set_step_cost(step_func)
agent.set_term_cost(term_func)
agent.set_g(phi_func,weights=weights_H,gamma=Gamma)
#agent.set_threshold(-0.1) #-0.5
agent.set_threshold(-100) #-0.5
agent.set_p(0.01)
agent.construct_graph(horizon=Horizon)

#construct controller
controller=ocsolver_v2('reacher control')
controller.set_state_param(4,[-2*np.pi,-2*np.pi,-100,-100],[2*np.pi,2*np.pi,100,100])
controller.set_ctrl_param(2,[-1e10]*2,[1e10]*2)
controller.set_dyn(dyn_func)
controller.set_step_cost(step_func)
controller.set_term_cost(term_func)
controller.set_g(phi_func,gamma=Gamma)
#controller.construct_graph(horizon=Horizon)
controller.construct_prob(horizon=Horizon)

#construct cutter
hb_calculator=cutter_v2('pendulum cut')
hb_calculator.from_controller(controller)
hb_calculator.construct_graph(horizon=Horizon)

#construct MVESolver
mve_calc=mvesolver('mve',5)

mve_calc.set_init_constraint(lbs, ubs) #Theta_0

learned_theta=np.array(weights_init)

env = Reacher_Env(1.0,1.0,1.0,1.0)
env.set_init_state(np.array([-np.pi/2,0,0,0]))

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
    init_state=np.array([-np.pi/2,0,0,0])
    # init_state[0] += np.random.uniform(-np.pi/4,np.pi/4)
    # init_state[1] += np.random.uniform(-np.pi/4,np.pi/4)
    # init_state[2] += np.random.uniform(-0.03,0.03)
    # init_state[3] += np.random.uniform(-0.03,0.03)

    env.set_init_state(init_state)
    for i in range(50):
        x=env.get_curr_state()
        #print(x)
        #input()
        try:
            u=controller.control(x,weights=learned_theta)
        except:
            print(phi_func(controller.opt_traj))
            input()
        agent_output=agent.act(controller.opt_traj)
        if agent_output is None:
            #print('emergency stop')
            break
        elif type(agent_output)==bool:
            pass
        else:
            st=time.time()
            #print(agent_output.flatten())
            h,b,h_phi,b_phi=hb_calculator.calc_planes(learned_theta,x,controller.opt_traj,np.sign(agent_output))
            #print('cutting plane calculated')
            #print('h',h)
            #print('b',b)
            #print('diff', h.T @ learned_theta - b)
            #print('h_phi',h_phi)
            #print('b_phi',b_phi)

            mve_calc.add_constraint(h,b[0])
            mve_calc.add_constraint(h_phi,b_phi[0])
            try:
                learned_theta,C=mve_calc.solve()
                print(learned_theta)
            except:
                termination_flag=True
                break
            print('calculation time',time.time()-st)
            #difference=np.linalg.norm(learned_theta-weights_H)
            difference=learned_theta-weights_H
            vol=np.log(np.linalg.det(C))
            #print('leanred safety param',learned_theta)
            theta_log.append(learned_theta)
            print('difference', difference)
            error_log.append(np.linalg.norm(difference))
            #print('volume', vol)
            volume_log.append(vol)
            #mve_calc.draw(C,learned_theta,weights_H)
            #if np.max(np.abs(difference))<0.04:
            if np.linalg.norm(difference) < 0.2:
                print("converged! Final Result: ",learned_theta)
                print(difference)
                termination_flag=True
                break
            
            corr_num+=1
        
        #print(controller.opt_traj)
        #input()
        
        env.step(u)
    #env.show_animation()
    EPISODE+=1

env.set_init_state(np.array([-np.pi/2,0,0,0]))
ee_traj = []
for i in range(200):
    x=env.get_curr_state()
    ee_pos = env.get_arm_position(x)
    ee_traj.append(ee_pos)
    u=controller.control(x,weights=learned_theta)
    env.step(u)

env.show_animation()
# np.save('../Data/Reacher/weights_2.npy',learned_theta)
# np.save('../Data/Reacher/traj_2.npy',np.array(ee_traj))

plt.figure()
plt.plot(error_log)
plt.show()

plt.figure()
plt.plot(volume_log)
plt.show()