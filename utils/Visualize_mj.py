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
    print('-----------------------')
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()

    viewer=mujoco.viewer.launch_passive(m, d)
    scene=viewer.user_scn
    scene.maxgeom=10000

    #viewer=mujoco.viewer.launch(m, d)
    #viewer=mujoco.viewer.launch()
    last_pos=None
    geom_idx=0
    while True:
        #print(mujoco.mj_getState(m,d))
        time.sleep(0.05)
        #d.ctrl[0]=4
        #mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)
        #print(viewer.scene)
        if last_pos is not None:
            #print(int(mujoco.mjtGeom.mjGEOM_LINE))
            scene.ngeom+=1
            mujoco.mjv_initGeom(
                scene.geoms[geom_idx],
                type=mujoco.mjtGeom.mjGEOM_LINE,
                size=np.zeros(3),
                pos=last_pos,
                mat=np.eye(3).flatten(),
                rgba=0.5*np.array([1, 0, 0, 1])
            )
            mujoco.mjv_makeConnector(scene.geoms[geom_idx],
                                 mujoco.mjtGeom.mjGEOM_LINE,
                                 10,
                                 d.qpos[0],d.qpos[1],d.qpos[2],
                                 last_pos[0],last_pos[1],last_pos[2])
            #viewer.user_scn.geoms[geom_idx].pos[2] = last_pos[2]
            print(d.qpos[2]-last_pos[2])
            geom_idx+=1
            
            viewer.sync()
            
        viewer.sync()
        #print(d.ctrl)
        last_pos=d.qpos[0:3].copy()
        d.qpos[2]+=0.02