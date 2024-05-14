import sys
import os
import time
import threading

sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
sys.path.append(os.path.abspath(os.getcwd()))

import casadi as cd
from Envs.robot_arm import EFFECTOR_env_mj, End_Effector_model, DH_to_Mat, Rot_Quat
from utils.RBF import generate_rbf_quat_z
from Solvers.OCsolver import  ocsolver_v4
import numpy as np


import rospy
from std_msgs.msg import Float64MultiArray


LATEST_HOMO_MATRIX=None
LATEST_THETA_CTRL=np.ones(20)#20

#lock 1: for communication with robot
#lock 2: for communication with learner
ros_lock_1=threading.Lock()
ros_lock_2=threading.Lock()
def listener_robot(data):
    global LATEST_HOMO_MATRIX
    # lock
    #print(data)
    ros_lock_1.acquire()
    LATEST_HOMO_MATRIX = np.array(data.data).reshape(4,4).T #message is column major
    # unlock
    ros_lock_1.release()
    #print(LATEST_HOMO_MATRIX)
    

    # rospy.loginfo(rospy.get_caller_id() + "I heard %s", data)

def listener_learner(data):
    global LATEST_THETA_CTRL
    ros_lock_2.acquire()
    LATEST_THETA_CTRL = np.array(data.data)
    ros_lock_2.release()
    print('theta for ctrl', LATEST_THETA_CTRL.flatten())


def main():
    global LATEST_HOMO_MATRIX, LATEST_THETA_CTRL
    # ros utils
    rospy.init_node('mpc', anonymous=True)
    rospy.Subscriber("robot_otee", Float64MultiArray, listener_robot, queue_size=100)
    rospy.Subscriber("theta", Float64MultiArray, listener_learner, queue_size=100)

    vel_pub = rospy.Publisher('velocity_command', Float64MultiArray, queue_size=10)
    target_pub = rospy.Publisher('target_x', Float64MultiArray, queue_size=10)
    rate = rospy.Rate(50) #hz

    #mpc utils
    dt=0.1
    Horizon=20
    arm_model=End_Effector_model(dt=dt)
    dyn_f = arm_model.get_dyn_f()
    step_cost_vec = np.array([0.8,0.0,30.0,30.0,1.0,0.85]) * 1e0 #param:[kr,kq,kvx,kvy,kvz,kw]
    step_cost_f = arm_model.get_step_cost_param_sep(step_cost_vec)
    term_cost_vec = np.array([8,6]) * 1e1
    term_cost_f = arm_model.get_terminal_cost_param(term_cost_vec)

    phi_func =  generate_rbf_quat_z(Horizon,x_center=-0.15,x_half=0.2,ref_axis=np.array([1,0,0]),num_q=10,
                                z_min=0.1,z_max=1.2, num_z=10, bias=-0.8, epsilon_z=9, epsilon_q=1.8,z_factor=0.05,mode='cumulative')
    
    # phi_func =  generate_rbf_quat_z(Horizon,x_center=-0.15,x_half=0.2,ref_axis=np.array([1,0,0]),num_q=10,
    #                             z_min=0.1,z_max=1.2, num_z=5, bias=-0.8, epsilon_z=7, epsilon_q=1.8,z_factor=0.1,mode='cumulative')


    Gamma=1.5 #1.0

    controller = ocsolver_v4('arm control')
    controller.set_state_param(7, None, None)
    controller.set_ctrl_param(6, 6*[-1e10], 6*[1e10])
    controller.set_dyn(dyn_f)
    controller.set_step_cost(step_cost_f)
    controller.set_term_cost(term_cost_f)
    controller.set_g(phi_func, gamma=Gamma)
    controller.construct_prob(horizon=Horizon)

    target_end_pos=[-0.65,-0.5,0.5]
    target_quat=[0.0,-0.707,0.0,0.707]
    target_x=target_end_pos+target_quat

    while not rospy.is_shutdown():
        latest_pose=None
        latest_mat=None
        process_flag=False
        # lock
        ros_lock_1.acquire()
        if LATEST_HOMO_MATRIX is not None:
            #print(LATEST_HOMO_MATRIX)
            latest_pose = LATEST_HOMO_MATRIX[0:3,3].reshape(-1,1)
            latest_mat = LATEST_HOMO_MATRIX[0:3,0:3]
            process_flag=True
        # unlock
        ros_lock_1.release()

        if process_flag:
            latest_quat=Rot_Quat(latest_mat).full().reshape(-1,1)
            #print('pos',latest_pose.flatten())
            #print('quat',latest_quat.flatten())
            curr_mpc_state=np.concatenate((latest_pose,latest_quat),axis=0)

            ros_lock_2.acquire()
            learned_theta=np.array(LATEST_THETA_CTRL)
            ros_lock_2.release()

            vel_command_raw = np.zeros(6)
            try:
                vel_command_raw=controller.control(curr_mpc_state,weights=learned_theta,target_x=target_x)
            except:
                vel_command_raw = np.zeros(6)
                print('error in control')
            
            #print(vel_command_raw)
            cmd_msg=Float64MultiArray(data=vel_command_raw)
            #cmd_msg.data=vel_command_raw
            vel_pub.publish(cmd_msg)

        target_msg=Float64MultiArray(data=np.array(target_x))
        target_pub.publish(target_msg)
            
        rate.sleep()

if __name__=='__main__':
    main()