import time
from pynput import keyboard
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
sys.path.append(os.path.abspath(os.getcwd()))
from utils.Keyboard import uav_key_handler
import matplotlib.pyplot as plt
import casadi as cd
from Envs.UAV import UAV_env,UAV_model
from Solvers.OCsolver import ocsolver,ocsolver_fast,ocsolver_inner_Barrier
import numpy as np
from matplotlib import pyplot as plt
from Solvers.Cutter import cutter
from Solvers.MVEsolver import mvesolver
from utils.Correction import Correction_Agent, uav_trans
from utils.RBF import rbf

#list for msg passing
PAUSE=[False]
MSG=[None]

#listener for keyboard ops
listener = keyboard.Listener(
    on_press=lambda key: uav_key_handler(key, PAUSE, MSG),
    on_release=None )
listener.start()

#graphics
fig, ax_2d = plt.subplots()

def graph_update(ax,env,theta=None):
    ax.clear()
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 10])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Quadrotor Trajectory (Top View)')
    ax.set_aspect(1)
    env.draw_curr_2d(ax)
    #draw the obstacle
    c_1=plt.Circle(xy=(5.5,7),radius=2)
    rect_1=plt.Rectangle(xy=(2,2),width=5,height=3)
    #rect_2=plt.Rectangle(xy=(0,9),width=4,height=0.5)
    #rect_3=plt.Rectangle(xy=(2,4),width=0.5,height=5)
    ax.add_patch(c_1)
    #ax.add_patch(rect_2)
    #ax.add_patch(rect_3)
    #ax.plot((0,2,8,10),(0,6,6,0))
    #ax.plot((1,3,7,9),(-1,4,4,-1))
    if theta is not None:
        ang_theta=np.linspace(-np.pi,np.pi,30)
        x=np.sqrt(20/-theta[0])*np.cos(ang_theta) + 5.5
        y=np.sqrt(20/-theta[1])*np.sin(ang_theta) + 7
        ax.plot(x,y,color='black')
    ax.scatter(9,9,marker='*',color='m',s=200)
    plt.draw()
    plt.pause(0.01)

#plt.ion()
# get dynamics, set up step cost and terminal cost
uav_params={'gravity':10,'m':1,'J_B':np.eye(3),'l_w':0.5,'dt':0.05,'c':1}
uav_env=UAV_env(**uav_params)


uav_model=UAV_model(**uav_params)
dyn_f=uav_model.get_dyn_f()

#r,v,q,w,u
#step_cost_vec=np.array([6,8,100,1,10])*1e-2
#step_cost_vec=np.array([40,60,20,1,10])*1e-3
step_cost_vec=np.array([40,6,40,100,10])*1e-3
step_cost_vec=np.array([50,10,5,10,15])*1e-3
step_cost_f=uav_model.get_step_cost(step_cost_vec,target_pos=np.array([9,9,5]))
#term_cost_vec=np.array([2,6,100,0.1])*1e-1
#term_cost_vec=np.array([30,30,15,2])*1e-2
term_cost_vec=np.array([20,5,15,100])*1e-2
term_cost_vec=np.array([20,6,40,50])*1e-2
term_cost_f=uav_model.get_terminal_cost(term_cost_vec,target_pos=np.array([9,9,5]))

# set up safety features
Horizon=50 #25
Gamma=3
hypo_lbs=0*np.ones(9)
hypo_ubs=5*np.ones(9)

hypo_lbs_poly=np.array([-5,-5,0,0,-5])
hypo_ubs_poly=np.array([0,0,20,20,10])

hypo_lbs_2d=np.array([-30,-30]) #-6
hypo_ubs_2d=np.array([0,0])
#phi
def generate_phi_rbf():
    x_dim=13
    u_dim=4
    traj=cd.SX.sym('xi',(x_dim+u_dim)*Horizon + x_dim)
    x_pos=traj[4*(x_dim+u_dim)]
    y_pos=traj[4*(x_dim+u_dim)+1]
    z_pos_1=traj[2*(x_dim+u_dim)+2]
    phi_list=[]
    phi_list.append(-2) #-4:16
    X_c=np.linspace(4,8,3)
    Y_c=np.linspace(4.5,9,3)
    grid_x,grid_y=np.meshgrid(X_c,Y_c)
    grid_x=grid_x.reshape(-1,1)
    grid_y=grid_y.reshape(-1,1)
    centers=np.concatenate([grid_x,grid_y],axis=1)
    for center in centers:
        print(center)
        phi_i=rbf(x_pos,y_pos,center[0],center[1],1.5)
        phi_list.append(phi_i)
    
    phi=cd.vertcat(*phi_list)
    return cd.Function('phi',[traj],[phi])

def generate_phi_poly():
    x_dim=13
    u_dim=4
    traj=cd.SX.sym('xi',(x_dim+u_dim)*Horizon + x_dim)
    x_pos=traj[5*(x_dim+u_dim)]
    y_pos=traj[5*(x_dim+u_dim)+1]
    phi_list=[-80]
    phi_list.append(-y_pos**2)
    phi_list.append(-x_pos**2)
    phi_list.append(x_pos)
    phi_list.append(y_pos)
    phi_list.append(-y_pos*x_pos)
    
    phi=cd.vertcat(*phi_list)
    return cd.Function('phi',[traj],[phi])

Center=(5.5,7,3)
Radius=2

def generate_phi_x_2():
    x_dim=13
    u_dim=4
    traj=cd.SX.sym('xi',(x_dim+u_dim)*Horizon + x_dim)
    x_pos_1=traj[5*(x_dim+u_dim)]
    y_pos_1=traj[5*(x_dim+u_dim)+1]
    z_pos_1=traj[4*(x_dim+u_dim)+2]
    phi=cd.vertcat(cd.DM(5*Radius**2),(x_pos_1-Center[0])*(x_pos_1-Center[0]),(y_pos_1-Center[1])*(y_pos_1-Center[1])) # to make theta_H [-5,-5]
    return cd.Function('phi',[traj],[phi])

#phi_func = generate_phi_poly()
#weights_init = (hypo_lbs_poly+hypo_ubs_poly)/2

phi_func = generate_phi_rbf()
weights_init = (hypo_lbs+hypo_ubs)/2

#phi_func = generate_phi_x_2()
#weights_init = (hypo_lbs_2d+hypo_ubs_2d)/2
#ctrl
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

#construct MVESolver
mve_calc=mvesolver('uav_mve',9)

mve_calc.set_init_constraint(hypo_lbs, hypo_ubs) #Theta_0

learned_theta=np.array(weights_init)
#learning logs
#theta_log=[np.array(weights_init)]
#error_log=[np.linalg.norm(weights_init-weights_H)]

d_0,C_0=mve_calc.solve()
v_0=np.log(np.linalg.det(C_0))
volume_log=[v_0]
EPISODE=0
num_corr=0
while True:
    init_r = np.array([1,8,1]).reshape(-1,1)
    init_v = np.zeros((3,1))
    init_q = np.reshape(np.array([1,0,0,0]),(-1,1))
    #print(Quat_Rot(init_q))
    init_w_B = np.zeros((3,1))
    init_x=np.concatenate([init_r,init_v,init_q,init_w_B],axis=0)
    init_x[0]=np.random.uniform(0.5,2.5)
    #init_x[0]=1
    init_x[1]=np.random.uniform(0.5,8.5)
    #init_x[1]=1
    print('init',init_x.T)
    uav_env.set_init_state(init_x) 
    for i in range(200):
        if not PAUSE[0]:
            if MSG[0]:
                # correction
                #print('message ',MSG[0])
                if MSG[0] == 'up': #y+
                    #print(uav_trans(np.array([0,1,0]),uav_env))
                    #human_corr=uav_trans(np.array([0,1,0]),uav_env)
                    human_corr=np.array([-0.5,0,1,0])
                if MSG[0] == 'down': #y-
                    #print(uav_trans(np.array([0,-1,0]),uav_env))
                    #human_corr=uav_trans(np.array([0,-1,0]),uav_env)
                    human_corr=np.array([1,0,-0.5,0])
                if MSG[0] == 'right': #x+
                    #print(uav_trans(np.array([1,0,0]),uav_env))
                    #human_corr=uav_trans(np.array([1,0,0]),uav_env)
                    human_corr=np.array([0,-0.5,0,1])
                if MSG[0] == 'left': #x-
                    #print(uav_trans(np.array([-1,0,0]),uav_env))
                    #human_corr=uav_trans(np.array([-1,0,0]),uav_env)
                    human_corr=np.array([0,1,0,-0.5])
                if MSG[0]=='quit' or MSG[0]=='reset':
                    break
                MSG[0]=None

                print('correction',human_corr)
                human_corr_e=np.concatenate([human_corr.reshape(-1,1),np.zeros((4*(Horizon-1),1))])
                h,b,h_phi,b_phi=hb_calculator.calc_planes(x,controller.opt_traj_t,human_corr=human_corr_e)
                print('cutting plane calculated')
                print('h',h)
                print('b',b)
                print('diff', h.T @ learned_theta - b)
                print('h_phi',h_phi)
                print('b_phi',b_phi)

                mve_calc.add_constraint(h,b[0])
                mve_calc.add_constraint(h_phi,b_phi[0])
                learned_theta,C=mve_calc.solve()
                print('vol',np.log(np.linalg.det(C)))
                #mve_calc.savefig(C,learned_theta,np.array([-5,-5]),dir='D:\\ASU_Work\\Research\\learn safe mpc\\experiment\\results\\cut_figs\\' +str(num_corr)+'.png')
                controller.set_g(phi_func,weights=learned_theta,gamma=Gamma)
                controller.construct_prob(horizon=Horizon)
                hb_calculator.set_g(phi_func,weights=learned_theta,gamma=Gamma)
                hb_calculator.construct_graph(horizon=Horizon)
                print(learned_theta)
                print('switching complete!')
                num_corr+=1
                time.sleep(0.2)  

            # simulation
            x=uav_env.get_curr_state()
            u=controller.control(x)
            uav_env.step(u)
            graph_update(ax_2d,uav_env,None)
            
            #num+=1
            time.sleep(0.1)  
        else:
            while PAUSE[0]:
                time.sleep(0.1)

    if MSG[0]=='reset':
        MSG[0]=None

    if MSG[0]=='quit':
        break
