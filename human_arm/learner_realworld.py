import sys
import os
import time
import threading

sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
sys.path.append(os.path.abspath(os.getcwd()))

import casadi as cd
from Envs.robot_arm import End_Effector_model,Rot_Quat
from Solvers.OCsolver import  ocsolver_v4
import numpy as np


import rospy
from std_msgs.msg import Float64MultiArray


LATEST_THETA=np.ones((20,1))
LATEST_HOMO_MATRIX=None
LATEST_CORR=None
CORR_FLAG=False

ros_lock=threading.Lock()

def listener_correction(data):
    global LATEST_HOMO_MATRIX,LATEST_CORR,CORR_FLAG
    # lock
    #print(data)
    ros_lock.acquire()
    LATEST_HOMO_MATRIX=np.array(data.data)[0:16].reshape(4,4).T
    LATEST_CORR=np.array(data.data)[16:22]
    CORR_FLAG=True
    # unlock
    ros_lock.release()

    # rospy.loginfo(rospy.get_caller_id() + "I heard %s", data)
    
def main():
    global LATEST_THETA,LATEST_HOMO_MATRIX,LATEST_CORR,CORR_FLAG
    # ros utils
    rospy.init_node('learner', anonymous=True)
    rospy.Subscriber("human_correction", Float64MultiArray, listener_correction, queue_size=100)

    theta_pub = rospy.Publisher('theta', Float64MultiArray, queue_size=10)
    rate = rospy.Rate(10) #hz

    while not rospy.is_shutdown():
        corr_pose=None
        corr_mat=None
        process_flag=False
        # lock
        ros_lock.acquire()
        if CORR_FLAG:
            #print(LATEST_HOMO_MATRIX)
            corr_pose = LATEST_HOMO_MATRIX[0:3,3].reshape(-1,1)
            corr_mat = LATEST_HOMO_MATRIX[0:3,0:3]
            CORR_FLAG=False
            process_flag=True
        ros_lock.release()

        if process_flag:
            corr_quat=Rot_Quat(corr_mat).full().reshape(-1,1)
            print('pos',corr_pose.flatten())
            print('quat',corr_quat.flatten())
            corr_state=np.concatenate((corr_pose,corr_quat),axis=0)

            # TODO Update Theta
            LATEST_THETA = np.random.rand(20,1)

        print('dummy theta',LATEST_THETA.flatten())
        theta_msg=Float64MultiArray(data=LATEST_THETA.flatten())
        #cmd_msg.data=vel_command_raw
        theta_pub.publish(theta_msg)
            
        rate.sleep()

if __name__=='__main__':
    main()