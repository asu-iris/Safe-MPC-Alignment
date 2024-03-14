import os
import sys
import casadi as cd
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
sys.path.append(os.path.abspath(os.getcwd()))
from Envs.UAV import UAV_env,UAV_model
from Solvers.OCsolver import ocsolver,ocsolver_fast,ocsolver_inner_Barrier
import numpy as np
from matplotlib import pyplot as plt
from Solvers.Cutter import cutter
# get dynamics, set up step cost and terminal cost
uav_params={'gravity':10,'m':1,'J_B':np.eye(3),'l_w':0.5,'dt':0.1,'c':1}
uav_env=UAV_env(**uav_params)

init_r = np.zeros((3,1))
init_v = np.zeros((3,1))
init_q = np.reshape(np.array([1,0,0,0]),(-1,1))
#print(Quat_Rot(init_q))
init_w_B = np.zeros((3,1))
init_x=np.concatenate([init_r,init_v,init_q,init_w_B],axis=0)

uav_env.set_init_state(init_x)

uav_model=UAV_model(**uav_params)
dyn_f=uav_model.get_dyn_f()

#r,v,q,w,u
#step_cost_vec=np.array([6,8,100,1,10])*1e-2
step_cost_vec=np.array([40,5,20,1,10])*1e-2
step_cost_f=uav_model.get_step_cost(step_cost_vec)
#term_cost_vec=np.array([2,6,100,0.1])*1e-1
term_cost_vec=np.array([20,6,10,2])*1e-1
term_cost_f=uav_model.get_terminal_cost(term_cost_vec)

# set up safety features
Horizon=25
#simple phi, avoid the circle c:(4,4), r=2 (Severe Gradient Issue, Need to Address)
Center=(3,3,3)
Radius=2.25
def generate_phi_x_1():
    x_dim=13
    u_dim=4
    traj=cd.SX.sym('xi',(x_dim+u_dim)*Horizon + x_dim)
    x_pos_1=traj[2*(x_dim+u_dim)]
    y_pos_1=traj[2*(x_dim+u_dim)+1]
    z_pos_1=traj[2*(x_dim+u_dim)+2]
    phi=cd.vertcat(cd.DM(Radius**2),(x_pos_1-Center[0])*(x_pos_1-Center[0]),(y_pos_1-Center[1])*(y_pos_1-Center[1]),(z_pos_1-Center[2])*(z_pos_1-Center[2]))
    return cd.Function('phi',[traj],[phi])

def generate_phi_x_2():
    x_dim=13
    u_dim=4
    traj=cd.SX.sym('xi',(x_dim+u_dim)*Horizon + x_dim)
    x_pos_1=traj[2*(x_dim+u_dim)]
    y_pos_1=traj[2*(x_dim+u_dim)+1]
    z_pos_1=traj[2*(x_dim+u_dim)+2]
    phi=cd.vertcat(cd.DM(Radius**2),(x_pos_1-Center[0])*(x_pos_1-Center[0]),(y_pos_1-Center[1])*(y_pos_1-Center[1]))
    return cd.Function('phi',[traj],[phi])

#cumulative version of constraint
def generate_phi_x():
    x_dim=13
    u_dim=4
    traj=cd.SX.sym('xi',(x_dim+u_dim)*Horizon + x_dim)
    x_feat=y_feat=0
    disc=1
    thresh=0
    for i in range(1,Horizon):
        x_pos=traj[i*(x_dim+u_dim)+0]
        y_pos=traj[i*(x_dim+u_dim)+1]
        x_feat=x_feat+disc*(x_pos-4)*(x_pos-4)
        y_feat=y_feat+disc*(y_pos-4)*(y_pos-4)
        thresh+=disc*5
        disc*=0.8
    phi=cd.vertcat(cd.DM(thresh),x_feat,y_feat)
    return cd.Function('phi',[traj],[phi])

#simple phi, speed limit v < 1
def generate_phi_v(): 
    x_dim=13
    u_dim=4
    traj=cd.SX.sym('xi',(x_dim+u_dim)*Horizon + x_dim)
    v_x=v_y=v_z=0
    disc=1
    thresh=0
    for i in range(1,Horizon+1):
        v_x=v_x+disc*traj[i*(x_dim+u_dim) + 3]**2
        v_y=v_y+disc*traj[i*(x_dim+u_dim) + 4]**2
        v_z=v_z+disc*traj[i*(x_dim+u_dim) + 5]**2
        thresh+=disc*4
        disc*=0.8
    phi=cd.vertcat(cd.DM(-thresh),v_x,v_y,v_z)
    return cd.Function('phi',[traj],[phi])

phi_func=generate_phi_v() #traj: [x,u,x,u,..,x] phi:[phi0, phi1, phi2]
weights=np.array([1,1,1]) # 1 - (x-4)^2 - (y-4)^2 <=0

#phi_func=generate_phi_x() #traj: [x,u,x,u,..,x] phi:[phi0, phi1, phi2]
#weights=np.array([-1,-1]) # 4 - (x-4)^2 - (y-4)^2 <=0

phi_func_hard=generate_phi_x_1() #traj: [x,u,x,u,..,x] phi:[phi0, phi1, phi2]
weights_hard=np.array([-1,-1,-1])

phi_func_hard=generate_phi_x_2() #traj: [x,u,x,u,..,x] phi:[phi0, phi1, phi2]
weights_hard=np.array([-1,-1,])

controller=ocsolver_fast('uav control')
controller.set_state_param(13,None,None)
controller.set_ctrl_param(4,[-1e10,-1e10,-1e10,-1e10],[1e10,1e10,1e10,1e10])
controller.set_dyn(dyn_f)
controller.set_step_cost(step_cost_f)
controller.set_term_cost(term_cost_f)
#controller.construct_graph(horizon=Horizon)
controller.set_g(phi_func_hard,weights=weights_hard,gamma=0.1)
controller.construct_prob(horizon=Horizon)

#construct controller
controller_2=ocsolver_inner_Barrier('uav control (inner)')
controller_2.set_state_param(13,None,None)
controller_2.set_ctrl_param(4,[-1e10,-1e10,-1e10,-1e10],[1e10,1e10,1e10,1e10])
controller_2.set_dyn(dyn_f)
controller_2.set_step_cost(step_cost_f)
controller_2.set_term_cost(term_cost_f)
#controller.construct_graph(horizon=Horizon)
controller_2.set_g(phi_func_hard,weights=weights_hard,gamma=0.1)
controller_2.construct_prob(horizon=Horizon)

#construct cutter
hb_calculator=cutter('uav cut')
hb_calculator.set_state_dim(13)
hb_calculator.set_ctrl_dim(4)
hb_calculator.set_dyn(dyn_f)
hb_calculator.set_step_cost(step_cost_f)
hb_calculator.set_term_cost(term_cost_f)
hb_calculator.set_g(phi_func_hard,weights=weights_hard,gamma=0.1)
hb_calculator.construct_graph(horizon=Horizon)

import time
for i in range(100):
    print(i,'--------------------------------------------')
    x=uav_env.get_curr_state()
    u=controller.control(x)
    #u=controller_2.control(x)
    print(u)
    print(x.flatten())
    #print(controller.opt_traj_t[17:19])
    #print(hb_calculator.calc_m_grad(x,controller.opt_traj_t))
    #print(np.array(u))
    #print(type(u))
    uav_env.step(u)
    #time.sleep(0.5)
    

uav_env.show_animation(center=Center,radius=Radius,mode='cylinder')