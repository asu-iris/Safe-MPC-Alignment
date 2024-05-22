import casadi as cd
import numpy as np
import sys,os
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
sys.path.append(os.path.abspath(os.getcwd()))
from Envs.UAV import UAV_env,UAV_env_mj
from Envs.robot_arm import ARM_env_mj
from Solvers.OCsolver import ocsolver_v2,ocsolver_v3

import mujoco
import mujoco.viewer

import time
import cv2

class uav_visualizer_mj(object):
    def __init__(self,env:UAV_env,controller:ocsolver_v2=None) -> None:
        self.env=env
        self.controller=controller
        self.mpc_horizon=15

    def render_init(self):
        filepath=os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),'mujoco_uav','bitcraze_crazyflie_2','scene.xml')
        self.m=mujoco.MjModel.from_xml_path(filepath)
        self.d=mujoco.MjData(self.m)
        self.viewer=mujoco.viewer.launch_passive(self.m, self.d)
        self.scene=self.viewer.user_scn
        self.scene.ngeom=0

        #initialize MPC trajectory
        for i in range(self.mpc_horizon-1):
            self.scene.ngeom+=1
            mujoco.mjv_initGeom(
                self.scene.geoms[i],
                type=mujoco.mjtGeom.mjGEOM_LINE,
                size=np.zeros(3),
                pos=np.zeros(3),
                mat=np.eye(3).flatten(),
                rgba=0.5*np.array([1, 0, 0.5, 1])
            )


    def render_update(self):
        pos=self.env.curr_x[0:3].flatten()
        q=self.env.curr_x[6:10].flatten()
        self.d.qpos[0:3]=pos
        self.d.qpos[3:7]=q
        self.plot_mpc_traj()
        mujoco.mj_forward(self.m, self.d)
        #mujoco.mj_step(self.m, self.d)
        self.viewer.sync()

    def plot_mpc_traj(self):
        traj_xu=self.controller.opt_traj
        traj_plot=[]
        x_dim=13
        u_dim=4
        for i in range(self.mpc_horizon):
            xyz_pos=traj_xu[i*(x_dim+u_dim):i*(x_dim+u_dim)+3]
            traj_plot.append(xyz_pos)
        traj_plot=np.array(traj_plot)

        for i in range(self.mpc_horizon-1):
            mujoco.mjv_makeConnector(self.scene.geoms[i],
                    mujoco.mjtGeom.mjGEOM_LINE,
                    10,
                    traj_plot[i,0],traj_plot[i,1],traj_plot[i,2],
                    traj_plot[i+1,0],traj_plot[i+1,1],traj_plot[i+1,2])
    
    def close_window(self):
        self.viewer.close()
        print('window closed')

class uav_visualizer_mj_v2(object):
    def __init__(self,env:UAV_env_mj,controller:ocsolver_v2=None) -> None:
        self.env=env
        self.controller=controller
        self.mpc_horizon=15

    def render_init(self):
        cam=self.env.model.cam('track_cf2')
        #print(type(cam))
        #input()
        self.viewer=mujoco.viewer.launch_passive(self.env.model, self.env.data)
        self.viewer.cam.type=mujoco.mjtCamera.mjCAMERA_FIXED
        self.viewer.cam.fixedcamid=cam.id

        self.scene=self.viewer.user_scn
        self.scene.ngeom=0

        #initialize MPC trajectory
        for i in range(self.mpc_horizon-1):
            self.scene.ngeom+=1
            mujoco.mjv_initGeom(
                self.scene.geoms[i],
                type=mujoco.mjtGeom.mjGEOM_LINE,
                size=np.zeros(3),
                pos=np.zeros(3),
                mat=np.eye(3).flatten(),
                rgba=0.5*np.array([1, 0, 0.5, 1])
            )


    def render_update(self):
        self.plot_mpc_traj()
        self.viewer.sync()

    def plot_mpc_traj(self):
        traj_xu=self.controller.opt_traj
        traj_plot=[]
        x_dim=13
        u_dim=4
        for i in range(self.mpc_horizon):
            xyz_pos=traj_xu[i*(x_dim+u_dim):i*(x_dim+u_dim)+3]
            traj_plot.append(xyz_pos)
        traj_plot=np.array(traj_plot)

        for i in range(self.mpc_horizon-1):
            mujoco.mjv_makeConnector(self.scene.geoms[i],
                    mujoco.mjtGeom.mjGEOM_LINE,
                    10,
                    traj_plot[i,0],traj_plot[i,1],traj_plot[i,2],
                    traj_plot[i+1,0],traj_plot[i+1,1],traj_plot[i+1,2])
    
    def close_window(self):
        self.viewer.close()
        print('window closed')

class uav_visualizer_mj_v3(uav_visualizer_mj_v2):
    def __init__(self, env: UAV_env_mj, controller: ocsolver_v3 = None) -> None:
        super().__init__(env, controller)

    def render_init(self):
        super().render_init()
        # register target ball
        self.scene.ngeom+=1
        mujoco.mjv_initGeom(
            self.scene.geoms[self.scene.ngeom-1],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.2, 0, 0],
            pos=-10*np.ones(3),
            mat=np.eye(3).flatten(),
            rgba=np.array([0, 0, 1, 1])
        )
        print(self.scene.ngeom)

    def set_target_pos(self,target_pos):
        self.target_pos=target_pos

    def render_update(self):
        self.plot_target()
        #self.viewer.sync()
        #print(self.scene.geoms[self.scene.ngeom-1].pos)
        super().render_update()
        

    def plot_target(self):
        self.scene.geoms[self.scene.ngeom-1].pos=self.target_pos

class uav_visualizer_mj_v4(uav_visualizer_mj_v3):
    def __init__(self, env: UAV_env_mj, controller: ocsolver_v3 = None) -> None:
        super().__init__(env, controller)

    def render_init(self):
        super().render_init()
        self.renderer=mujoco.Renderer(self.env.model, 480, 640)
        self.renderer.update_scene(self.env.data, "cam_aux")
        
        self.scene_aux=self.renderer.scene
        self.default_ngeom=self.scene_aux.ngeom

        #initialize MPC trajectory in aux renderer
        for i in range(self.default_ngeom,self.default_ngeom+self.mpc_horizon-1):
            self.scene_aux.ngeom+=1
            mujoco.mjv_initGeom(
                self.scene_aux.geoms[i],
                type=mujoco.mjtGeom.mjGEOM_LINE,
                size=np.zeros(3),
                pos=np.zeros(3),
                mat=np.eye(3).flatten(),
                rgba=0.5*np.array([1, 0, 0.5, 1])
            )

        # register target ball in aux renderer
        self.scene_aux.ngeom+=1
        mujoco.mjv_initGeom(
            self.scene_aux.geoms[self.scene_aux.ngeom-1],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.2, 0, 0],
            pos=-10*np.ones(3),
            mat=np.eye(3).flatten(),
            rgba=np.array([0, 0, 1, 1])
        )
        print(self.scene_aux.ngeom)

    def plot_target_aux(self):
        self.scene_aux.ngeom+=1
        mujoco.mjv_initGeom(
            self.scene_aux.geoms[self.scene_aux.ngeom-1],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.2, 0, 0],
            pos=-10*np.ones(3),
            mat=np.eye(3).flatten(),
            rgba=np.array([0, 0, 1, 1])
        )
        self.scene_aux.geoms[self.scene_aux.ngeom-1].pos=self.target_pos

    def plot_mpc_traj_aux(self):
        traj_xu=self.controller.opt_traj
        traj_plot=[]
        x_dim=13
        u_dim=4
        for i in range(self.mpc_horizon):
            xyz_pos=traj_xu[i*(x_dim+u_dim):i*(x_dim+u_dim)+3]
            traj_plot.append(xyz_pos)
        traj_plot=np.array(traj_plot)

        for i in range(self.default_ngeom,self.default_ngeom+self.mpc_horizon-1):
            self.scene_aux.ngeom+=1
            mujoco.mjv_initGeom(
                self.scene_aux.geoms[i],
                type=mujoco.mjtGeom.mjGEOM_LINE,
                size=np.zeros(3),
                pos=np.zeros(3),
                mat=np.eye(3).flatten(),
                rgba=0.5*np.array([1, 0, 0.5, 1])
            )

        for i in range(self.mpc_horizon-1):
            mujoco.mjv_makeConnector(self.scene_aux.geoms[self.default_ngeom+i],
                    mujoco.mjtGeom.mjGEOM_LINE,
                    10,
                    traj_plot[i,0],traj_plot[i,1],traj_plot[i,2],
                    traj_plot[i+1,0],traj_plot[i+1,1],traj_plot[i+1,2])
            
    def render_update(self):
        super().render_update()
        self.renderer.update_scene(self.env.data, "cam_aux")
        self.plot_mpc_traj_aux()
        self.plot_target_aux()
        frame_raw=self.renderer.render() 
        frame=cv2.cvtColor(frame_raw,cv2.COLOR_RGB2BGR)
        cv2.imshow('aux_view', frame)
        k = cv2.waitKey(1)
        #print(frame.shape)
        

class arm_visualizer_mj_v1(object):
    def __init__(self,env:ARM_env_mj,controller:ocsolver_v2=None) -> None:
        self.env=env
        self.controller=controller
        self.mpc_horizon=15

    def render_init(self):
        self.viewer=mujoco.viewer.launch_passive(self.env.model, self.env.data)
        #init geom for target
        self.scene=self.viewer.user_scn
        self.scene.ngeom=0
        self.scene.ngeom+=1
        mujoco.mjv_initGeom(
            self.scene.geoms[self.scene.ngeom-1],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.02, 0, 0],
            pos=-10*np.ones(3),
            mat=np.eye(3).flatten(),
            rgba=np.array([0, 1, 1, 1])
        )

    def render_update(self):
        self.plot_target()
        self.viewer.sync()

    def set_target_pos(self,target_pos):
        self.target_pos=target_pos
    
    def plot_target(self):
        self.scene.geoms[self.scene.ngeom-1].pos=self.target_pos

    def close_window(self):
        self.viewer.close()
        print('window closed')

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