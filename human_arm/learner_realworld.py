import sys
import os
import time
import threading

sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
sys.path.append(os.path.abspath(os.getcwd()))

import casadi as cd
from Envs.robot_arm import End_Effector_model,Rot_Quat
from Solvers.OCsolver import  ocsolver_v4
from Solvers.Cutter import  cutter_v4
from Solvers.MVEsolver import mvesolver
from utils.RBF import generate_rbf_quat_z
import numpy as np


import rospy
from std_msgs.msg import Float64MultiArray


LATEST_THETA=np.ones((20,1))
LATEST_HOMO_MATRIX=None
LATEST_CORR=None
CORR_FLAG=False

LATEST_TARGET = np.array([-0.65,-0.5,0.5,0.0,-0.707,0.0,0.707])
#lock 1: correction data
#lock 2: target position
ros_lock_1=threading.Lock()
ros_lock_2=threading.Lock()

def listener_correction(data):
    global LATEST_HOMO_MATRIX,LATEST_CORR,CORR_FLAG
    # lock
    #print(data)
    ros_lock_1.acquire()
    LATEST_HOMO_MATRIX=np.array(data.data)[0:16].reshape(4,4).T
    LATEST_CORR=np.array(data.data)[16:22]
    CORR_FLAG=True
    # unlock
    ros_lock_1.release()

    # rospy.loginfo(rospy.get_caller_id() + "I heard %s", data)

def listener_target(data):
    global LATEST_TARGET
    ros_lock_2.acquire()
    LATEST_TARGET=np.array(data.data)
    #print('target',LATEST_TARGET)
    # unlock
    ros_lock_2.release()
    

    
def main():
    global LATEST_THETA,LATEST_HOMO_MATRIX,LATEST_CORR,CORR_FLAG,LATEST_TARGET
    # ros utils
    rospy.init_node('learner', anonymous=True)
    rospy.Subscriber("human_correction", Float64MultiArray, listener_correction, queue_size=100)
    rospy.Subscriber("target_x", Float64MultiArray, listener_target, queue_size=100)

    theta_pub = rospy.Publisher('theta', Float64MultiArray, queue_size=10)
    rate = rospy.Rate(10) #hz

    # Model
    dt=0.1
    Horizon=10
    arm_model=End_Effector_model(dt=dt)
    dyn_f = arm_model.get_dyn_f()
    step_cost_vec = np.array([0.8,0.0,30.0,30.0,1.0,0.85]) * 1e0 #param:[kr,kq,kvx,kvy,kvz,kw]
    step_cost_f = arm_model.get_step_cost_param_sep(step_cost_vec)
    term_cost_vec = np.array([8,6]) * 1e1
    term_cost_f = arm_model.get_terminal_cost_param(term_cost_vec)

    # MPC for trajectory generation
    phi_func =  generate_rbf_quat_z(Horizon,x_center=-0.15,x_half=0.2,ref_axis=np.array([1,0,0]),num_q=10,
                                z_min=0.1,z_max=1.2, num_z=10, bias=-0.8, epsilon_z=9, epsilon_q=1.8,z_factor=0.05,mode='cumulative')

    # phi_func =  generate_rbf_quat_z(Horizon,x_center=-0.15,x_half=0.2,ref_axis=np.array([1,0,0]),num_q=10,
    #                             z_min=0.1,z_max=1.2, num_z=5, bias=-0.8, epsilon_z=7, epsilon_q=1.8,z_factor=0.1,mode='cumulative')
    
    Gamma=1.0 #

    controller = ocsolver_v4('learner controller')
    controller.set_state_param(7, None, None)
    controller.set_ctrl_param(6, 6*[-1e10], 6*[1e10])
    controller.set_dyn(dyn_f)
    controller.set_step_cost(step_cost_f)
    controller.set_term_cost(term_cost_f)
    controller.set_g(phi_func, gamma=Gamma)
    controller.construct_prob(horizon=Horizon)

    # cutter, mvesolveer
    hb_calculator = cutter_v4('arm cut')
    hb_calculator.from_controller(controller)
    hb_calculator.construct_graph(horizon=Horizon)

    theta_dim = 20
    hypo_lbs = -3 * np.ones(theta_dim)
    hypo_ubs = 5 * np.ones(theta_dim)

    mve_calc = mvesolver('arm_mve', theta_dim)
    mve_calc.set_init_constraint(hypo_lbs, hypo_ubs)

    #loop

    while not rospy.is_shutdown():
        corr_pose=None
        corr_mat=None
        correction=None
        process_flag=False
        # lock
        ros_lock_1.acquire()
        if CORR_FLAG:
            #print(LATEST_HOMO_MATRIX)
            corr_pose = LATEST_HOMO_MATRIX[0:3,3].reshape(-1,1)
            corr_mat = np.array(LATEST_HOMO_MATRIX[0:3,0:3])
            correction = np.array(LATEST_CORR)
            CORR_FLAG=False
            process_flag=True
        ros_lock_1.release()

        if process_flag:
            corr_quat=Rot_Quat(corr_mat).full().reshape(-1,1)
            print('pos',corr_pose.flatten())
            print('quat',corr_quat.flatten())
            corr_state=np.concatenate((corr_pose,corr_quat),axis=0)

            # TODO Update Theta
            ros_lock_2.acquire()
            target_x=np.array(LATEST_TARGET)
            ros_lock_2.release()
            
            controller.control(corr_state, weights=LATEST_THETA, target_x=target_x)
            correction[0:2]=0
            correction[2]*=10
            print('correction',correction)
            human_corr_e=np.concatenate([correction.reshape(-1, 1), np.zeros((6 * (Horizon - 1), 1))])
            h, b, h_phi, b_phi = hb_calculator.calc_planes(LATEST_THETA, corr_state, controller.opt_traj,
                                                                   human_corr=human_corr_e,
                                                                   target_x=target_x)
            mve_calc.add_constraint(h, b[0])
            mve_calc.add_constraint(h_phi, b_phi[0])
            try:
                LATEST_THETA, C = mve_calc.solve()
                print('theta',LATEST_THETA.flatten())
            except:
                return -1

        
        theta_msg=Float64MultiArray(data=LATEST_THETA.flatten())
        #cmd_msg.data=vel_command_raw
        theta_pub.publish(theta_msg)
            
        rate.sleep()

if __name__=='__main__':
    main()