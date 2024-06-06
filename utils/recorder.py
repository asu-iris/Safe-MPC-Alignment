import casadi as cd
import numpy as np
import sys,os
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
sys.path.append(os.path.abspath(os.getcwd()))
from Envs.UAV import UAV_env,UAV_env_mj
from Envs.robot_arm import EFFECTOR_env_mj
from Solvers.OCsolver import ocsolver_v2,ocsolver_v4
from utils.Visualize_mj import uav_visualizer_mj_v4

import mujoco
import mujoco.viewer

import datetime
import cv2

class Recorder_sync(object):
    def __init__(self,env:UAV_env_mj,controller:ocsolver_v2=None,visualizer:uav_visualizer_mj_v4=None,height=480,width=640, 
                 filepath=os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),'Data','test'),
                 cam_flag=False) -> None:
        self.env=env
        self.controller=controller
        self.mpc_horizon=15

        self.height = height
        self.width = width

        self.renderer=mujoco.Renderer(self.env.model, self.height, self.width)
        self.renderer.update_scene(self.env.data, "track_cf2")
        self.scene=self.renderer.scene
        self.default_ngeom=self.scene.ngeom

        self.filepath=filepath
        self.frames=[]
        self.timestamps=[]
        self.corrections=[]

        self.cam_frames=None
        self.cap=None

        self.cam_flag=cam_flag
        if self.cam_flag:
            self.cap=cv2.VideoCapture(0)
            ret, img = self.cap.read()
            print(img.shape)
            #input()
            self.cam_frames=[]

        self.aux_frames=None
        self.aux_flag=False
        self.vis=visualizer
        if visualizer is not None:
            self.aux_frames=[]
            self.aux_flag=True

        self.target_pos=None
        #initialize MPC trajectory
        #print(self.scene.ngeom)
        #input()
    def set_target_pos(self,target):
        self.target_pos=target

    def plot_mpc_traj(self):
        traj_xu=self.controller.opt_traj
        traj_plot=[]
        x_dim=13
        u_dim=4
        for i in range(self.mpc_horizon):
            xyz_pos=traj_xu[i*(x_dim+u_dim):i*(x_dim+u_dim)+3]
            traj_plot.append(xyz_pos)
        traj_plot=np.array(traj_plot)

        for i in range(self.default_ngeom,self.default_ngeom+self.mpc_horizon-1):
            self.scene.ngeom+=1
            mujoco.mjv_initGeom(
                self.scene.geoms[i],
                type=mujoco.mjtGeom.mjGEOM_LINE,
                size=np.zeros(3),
                pos=np.zeros(3),
                mat=np.eye(3).flatten(),
                rgba=0.5*np.array([1, 0, 0.5, 1])
            )

        for i in range(self.mpc_horizon-1):
            mujoco.mjv_makeConnector(self.scene.geoms[self.default_ngeom+i],
                    mujoco.mjtGeom.mjGEOM_LINE,
                    10,
                    traj_plot[i,0],traj_plot[i,1],traj_plot[i,2],
                    traj_plot[i+1,0],traj_plot[i+1,1],traj_plot[i+1,2])

    def plot_target(self):
        # register target ball in renderer
        self.scene.ngeom+=1
        mujoco.mjv_initGeom(
            self.scene.geoms[self.scene.ngeom-1],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.2, 0, 0],
            pos=-10*np.ones(3),
            mat=np.eye(3).flatten(),
            rgba=np.array([0, 0, 1, 1])
        )

        self.scene.geoms[self.scene.ngeom-1].pos=self.target_pos
   
    def record(self,corr_flag=False,correction=None):
        self.renderer.update_scene(self.env.data, "track_cf2")
        if self.controller is not None:
            self.plot_mpc_traj()
        if self.target_pos is not None:
            self.plot_target()
        #print(self.scene.ngeom)
        #input()
        # mj frames
        frame=self.renderer.render()
        self.frames.append(frame)
        dt=datetime.datetime.now()
        dt_str=dt.strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
        self.timestamps.append(dt_str)
        # aux view
        if self.aux_flag:
            aux_frame=self.vis.frame_aux
            self.aux_frames.append(aux_frame)
        #webcam
        if self.cam_flag:
            ret, img = self.cap.read()
            scale_percent=60
            width=int(img.shape[1]*scale_percent/100)
            height=int(img.shape[0]*scale_percent/100)
            img_new=cv2.resize(img, (width,height),)
            self.cam_frames.append(img)
        # corrections
        if corr_flag==False:
            self.corrections.append(None)
        
        else:
            self.corrections.append(correction)

    def record_mj(self,corr_flag=False,correction=None):
        self.renderer.update_scene(self.env.data, "track_cf2")
        if self.controller is not None:
            self.plot_mpc_traj()
        #print(self.scene.ngeom)
        #input()
        # mj frames
        frame=self.renderer.render()
        self.frames.append(frame)
        dt=datetime.datetime.now()
        dt_str=dt.strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
        self.timestamps.append(dt_str)

        if corr_flag==False:
            self.corrections.append(None)
        
        else:
            self.corrections.append(correction)

    def record_cam(self):
        if self.cam_flag:
            ret, img = self.cap.read()
            scale_percent=60
            width=int(img.shape[1]*scale_percent/100)
            height=int(img.shape[0]*scale_percent/100)
            img_new=cv2.resize(img, (width,height),)
            self.cam_frames.append(img)
        


    def write(self):
        if not os.path.exists(self.filepath):
            os.makedirs(self.filepath)

        files=os.listdir(self.filepath)
        for file in files:
            os.remove(os.path.join(self.filepath,file))
        n_frames=len(self.frames)
        f=open(os.path.join(self.filepath,'timestamps.txt'),'w')
        for i in range(n_frames):
            print(str(i),self.timestamps[i], str(self.corrections[i]),file=f)
            frame_filename=os.path.join(self.filepath,'mj_'+str(i)+'.jpg')
            cv2.imwrite(frame_filename,cv2.cvtColor(self.frames[i],cv2.COLOR_RGB2BGR))

        if self.aux_flag:
            for i in range(n_frames):
                frame_filename=os.path.join(self.filepath,'aux_'+str(i)+'.jpg')
                cv2.imwrite(frame_filename,self.aux_frames[i])

        if self.cam_flag:
            for i in range(n_frames):
                frame_filename=os.path.join(self.filepath,'cam_'+str(i)+'.jpg')
                cv2.imwrite(frame_filename,self.cam_frames[i])
        f.close()

class Recorder_Arm(object):
    def __init__(self,env:EFFECTOR_env_mj,controller:ocsolver_v4=None,height=480,width=640, fps=12, 
                 filepath=os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),'Data','test_arm'),
                 cam_flag=False) -> None:
        self.env=env
        self.controller=controller
        self.mpc_horizon=15

        self.height = height
        self.width = width
        self.size=(width,height)
        self.fps = fps

        self.renderer=mujoco.Renderer(self.env.model, self.height, self.width)
        self.renderer.update_scene(self.env.data, "cam_1")

        self.filepath=filepath
        self.frames=[]

    def record_mj(self,corr_flag=False,correction=None):
        self.renderer.update_scene(self.env.data, "cam_1")
        frame=self.renderer.render()
        self.frames.append(frame)

    def write(self):
        if not os.path.exists(self.filepath):
            os.makedirs(self.filepath)

        files=os.listdir(self.filepath)
        for file in files:
            os.remove(os.path.join(self.filepath,file))

        video=cv2.VideoWriter(os.path.join(self.filepath,'demo.avi'),cv2.VideoWriter_fourcc(*'MJPG'),self.fps,self.size)
        for frame in self.frames:
            img=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            video.write(img)

        video.release()

class Recorder_Arm_v2(object):
    def __init__(self,env:EFFECTOR_env_mj,controller:ocsolver_v4=None,height=480,width=640, fps=12, 
                 filepath=os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),'Data','test_arm'),
                 cam_flag=False) -> None:
        self.env=env
        self.controller=controller
        self.mpc_horizon=15

        self.height = height
        self.width = width
        self.size=(width,height)
        self.fps = fps

        self.renderer=mujoco.Renderer(self.env.model, self.height, self.width)
        self.renderer.update_scene(self.env.data, "cam_1")
        self.scene=self.renderer.scene

        self.filepath=filepath
        self.frames=[]
        self.frames_2=[]

        self.target_pos=None
    
    def set_target_pos(self,target):
        self.target_pos=target


    def plot_target(self):
        # register target ball in renderer
        self.scene.ngeom+=1
        mujoco.mjv_initGeom(
            self.scene.geoms[self.scene.ngeom-1],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.02, 0, 0],
            pos=-10*np.ones(3),
            mat=np.eye(3).flatten(),
            rgba=np.array([0, 1, 1, 1])
        )

        self.scene.geoms[self.scene.ngeom-1].pos=self.target_pos

    def record_mj(self,corr_flag=False,correction=None):
        self.renderer.update_scene(self.env.data, "cam_1")
        if self.target_pos is not None:
            self.plot_target()
        frame=self.renderer.render()
        self.frames.append(frame)

        self.renderer.update_scene(self.env.data, "cam_2")
        frame=self.renderer.render()
        self.frames_2.append(frame)

    def write(self):
        if not os.path.exists(self.filepath):
            os.makedirs(self.filepath)

        files=os.listdir(self.filepath)
        for file in files:
            os.remove(os.path.join(self.filepath,file))

        for i in range(len(self.frames)):
            frame_filename=os.path.join(self.filepath,'mj_cam1_'+str(i)+'.jpg')
            cv2.imwrite(frame_filename,cv2.cvtColor(self.frames[i],cv2.COLOR_RGB2BGR))

        for i in range(len(self.frames_2)):
            frame_filename=os.path.join(self.filepath,'mj_cam2_'+str(i)+'.jpg')
            cv2.imwrite(frame_filename,cv2.cvtColor(self.frames_2[i],cv2.COLOR_RGB2BGR))
      