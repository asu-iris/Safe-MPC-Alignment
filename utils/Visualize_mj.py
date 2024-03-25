import casadi as cd
import numpy as np
import sys,os
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
sys.path.append(os.path.abspath(os.getcwd()))
from Envs.UAV import UAV_env,Quat_Rot
from Solvers.OCsolver import ocsolver_v2

import mujoco
import mujoco.viewer

import time

class uav_visualizer_mj(object):
    def __init__(self,env:UAV_env,controller:ocsolver_v2=None) -> None:
        self.env=env
        self.controller=controller

    def render_init(self):
        filepath=os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),'mujoco_uav','bitcraze_crazyflie_2','scene.xml')
        self.m=mujoco.MjModel.from_xml_path(filepath)
        self.d=mujoco.MjData(self.m)
        self.viewer=mujoco.viewer.launch_passive(self.m, self.d)

    def render_update(self):
        pos=self.env.curr_x[0:3].flatten()
        q=self.env.curr_x[6:10].flatten()
        self.d.qpos[0:3]=pos
        self.d.qpos[3:7]=q

        mujoco.mj_forward(self.m, self.d)
        self.viewer.sync()

if __name__=='__main__':
    filepath=os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),'mujoco_uav','bitcraze_crazyflie_2','scene.xml')
    print('path',filepath)
    m=mujoco.MjModel.from_xml_path(filepath)
    d=mujoco.MjData(m)
    viewer=mujoco.viewer.launch_passive(m, d)
    #viewer=mujoco.viewer.launch(m, d)
    #viewer=mujoco.viewer.launch()
    while True:
        #print(mujoco.mj_getState(m,d))
        time.sleep(0.05)
        #d.ctrl[0]=4
        #mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)
        viewer.sync()
        #print(d.ctrl)
        d.qpos[2]+=0.01