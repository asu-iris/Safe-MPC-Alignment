import os
import sys
import casadi as cd
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
sys.path.append(os.path.abspath(os.getcwd()))
from Envs.UAV import UAV_env,UAV_model
from Solvers.OCsolver import ocsolver,ocsolver_fast,ocsolver_inner_Barrier
import numpy as np
from matplotlib import pyplot as plt
from Solvers.Cutter import cutter,cutter_v2
from Solvers.MVEsolver import mvesolver
from utils.Correction import Correction_Agent
# get dynamics, set up step cost and terminal cost
uav_params={'gravity':10,'m':1,'J_B':np.eye(3),'l_w':0.5,'dt':0.1,'c':1}
uav_env=UAV_env(**uav_params)


uav_model=UAV_model(**uav_params)
dyn_f=uav_model.get_dyn_f()

#r,v,q,w,u
target_pos=np.array([7,7,5])
#step_cost_vec=np.array([6,8,100,1,10])*1e-2
#step_cost_vec=np.array([40,6,20,1,10])*1e-3
step_cost_vec=np.array([50,5,5,1,15])*1e-3
#step_cost_vec=np.array([10,20,0.0,0.0,40])*1e-3
step_cost_f=uav_model.get_step_cost(step_cost_vec,target_pos=target_pos)
#term_cost_vec=np.array([2,6,100,0.1])*1e-1
#term_cost_vec=np.array([20,5,15,2])*1e-2
term_cost_vec=np.array([20,6,5,5])*1e-2
#term_cost_vec=np.array([20,5,0,0.0])*1e-2
term_cost_f=uav_model.get_terminal_cost(term_cost_vec,target_pos=target_pos)

# set up safety features
Horizon=25
Gamma=0.1
#Hypothe.sis space
hypo_lbs=np.array([-20,-20]) #-6
hypo_ubs=np.array([-2,-2])
#simple phi, avoid the circle c:(3,3), r=2 (Severe Gradient Issue, Need to Address)
Center=(3,3.1,3)
Radius=2.25

def generate_phi_x_2():
    x_dim=13
    u_dim=4
    traj=cd.SX.sym('xi',(x_dim+u_dim)*Horizon + x_dim)
    x_pos_1=traj[5*(x_dim+u_dim)]
    y_pos_1=traj[5*(x_dim+u_dim)+1]
    z_pos_1=traj[3*(x_dim+u_dim)+2]
    phi=cd.vertcat(cd.DM(5*Radius**2),(x_pos_1-Center[0])*(x_pos_1-Center[0]),(y_pos_1-Center[1])*(y_pos_1-Center[1])) # to make theta_H [-5,-5]
    return cd.Function('phi',[traj],[phi])

#phi_func=generate_phi_x() #traj: [x,u,x,u,..,x] phi:[phi0, phi1, phi2]
#weights=np.array([-1,-1]) # 4 - (x-4)^2 - (y-4)^2 <=0

phi_func=generate_phi_x_2() #traj: [x,u,x,u,..,x] phi:[phi0, phi1, phi2]
weights_init = (hypo_lbs+hypo_ubs)/2
weights_H=np.array([-5,-5,])
#weights_H=np.array([-15,-5,])
agent=Correction_Agent('dummy')
agent.set_state_dim(13)
agent.set_ctrl_dim(4)
agent.set_dyn(dyn_f)
agent.set_step_cost(step_cost_f)
agent.set_term_cost(term_cost_f)
agent.set_g(phi_func,weights=weights_H,gamma=Gamma)
agent.set_threshold(-10) #-0.5
agent.set_p(0.5)
agent.construct_graph(horizon=Horizon)

def sim_human(agent_corr):
    idx_max=np.argmax(agent_corr)
    idx_min=np.argmin(agent_corr)
    output=np.zeros(agent_corr.shape)
    output[idx_max]=1
    output[idx_min]=-1
    return output

controller=ocsolver_fast('uav control')
controller.set_state_param(13,None,None)
controller.set_ctrl_param(4,[-1e10,-1e10,-1e10,-1e10],[1e10,1e10,1e10,1e10])
controller.set_dyn(dyn_f)
controller.set_step_cost(step_cost_f)
controller.set_term_cost(term_cost_f)
#controller.construct_graph(horizon=Horizon)
controller.set_g(phi_func,weights=weights_init,gamma=Gamma)
controller.construct_prob(horizon=Horizon)

#construct cutter
hb_calculator=cutter('uav cut')
hb_calculator.set_state_dim(13)
hb_calculator.set_ctrl_dim(4)
hb_calculator.set_dyn(dyn_f)
hb_calculator.set_step_cost(step_cost_f)
hb_calculator.set_term_cost(term_cost_f)
hb_calculator.set_g(phi_func,weights=weights_init,gamma=Gamma)
hb_calculator.construct_graph(horizon=Horizon)

cut_2=cutter_v2('uav cut')
cut_2.set_state_dim(13)
cut_2.set_ctrl_dim(4)
cut_2.set_dyn(dyn_f)
cut_2.set_step_cost(step_cost_f)
cut_2.set_term_cost(term_cost_f)
cut_2.set_g(phi_func,gamma=Gamma)
cut_2.construct_graph(horizon=Horizon)

#construct MVESolver
mve_calc=mvesolver('uav_mve',2)

mve_calc.set_init_constraint(hypo_lbs, hypo_ubs) #Theta_0

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
    # random init on a arc
    init_r = np.zeros((3,1))
    start_ang=np.random.uniform(-np.pi,-np.pi/2)
    start_r=np.sqrt(Center[0]**2+Center[1]**2)
    init_r[0]=Center[0] + np.cos(start_ang) * start_r
    init_r[1]=Center[1] + np.sin(start_ang) * start_r
    init_v = np.zeros((3,1))
    init_q = np.reshape(np.array([1,0,0,0]),(-1,1))
    #init_q = np.reshape(np.array([np.sqrt(2)/2,np.sqrt(2)/2,0,0]),(-1,1))
    #print(Quat_Rot(init_q))
    init_w_B = np.zeros((3,1))
    init_x=np.concatenate([init_r,init_v,init_q,init_w_B],axis=0)

    uav_env.set_init_state(init_x)
    for i in range(100):
        x=uav_env.get_curr_state()
        if np.sqrt(np.sum((x[0:3,0]-np.array([7,7,7]))**2)) <=0.15:
            print('reached desired position')
            break
        #print(i)
        u=controller.control(x)
        #print('u',u)
        agent_output=agent.act(controller.opt_traj_t)
        if agent_output==None:
            print('emergency stop')
            break
        elif type(agent_output)==bool:
            pass
        else:
            print('agent corr',agent_output)
            print('sim human', sim_human(agent_output[0:4]))
            interface_corr=np.zeros(agent_output.shape)
            interface_corr[0:4]=sim_human(agent_output[0:4])
            agent_output[4:]=0
            #agent_output=np.sign(agent_output)
            #agent_output[0]*=0.3
            #h,b,h_phi,b_phi=hb_calculator.calc_planes(x,controller.opt_traj_t,interface_corr)
            h,b,h_phi,b_phi=cut_2.calc_planes(learned_theta,x,controller.opt_traj_t,interface_corr)
            print('cutting plane calculated')
            #print('h diff',cd.norm_2(h-h_2))
            #print('b diff',cd.norm_2(b-b_2))
            #print('diff', h.T @ learned_theta - b)
            #print('h_phi diff',cd.norm_2(h_phi-h_phi_2))
            #print('b_phi diff',cd.norm_2(b_phi-b_phi_2))

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
            mve_calc.draw(C,learned_theta,weights_H)
            if np.max(np.abs(difference))<0.05:
                print("converged! Final Result: ",learned_theta)
                termination_flag=True
                break
            #set new constraints
            controller.set_g(phi_func,weights=learned_theta,gamma=Gamma)
            controller.construct_prob(horizon=Horizon)

            #hb_calculator.set_g(phi_func,weights=learned_theta,gamma=Gamma)
            #hb_calculator.construct_graph(horizon=Horizon)
            corr_num+=1
        uav_env.step(u)
    #uav_env.show_animation(center=Center,radius=Radius,mode='cylinder')
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

init_r = np.zeros((3,1))
init_v = np.zeros((3,1))
init_q = np.reshape(np.array([1,0,0,0]),(-1,1))
#print(Quat_Rot(init_q))
init_w_B = np.zeros((3,1))
init_x=np.concatenate([init_r,init_v,init_q,init_w_B],axis=0)
uav_env.set_init_state(init_x)
controller.set_g(phi_func,weights=learned_theta,gamma=Gamma)
controller.construct_prob(horizon=Horizon)
controller.reset_warmstart()
for i in range(120):
    print(i,'--------------------------------------------')
    x=uav_env.get_curr_state()
    u=controller.control(x)
    print(u)
    print(x.flatten())
    uav_env.step(u)

uav_env.show_animation(center=Center,radius=Radius,mode='cylinder')