import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

def plot_bars(ax:Axes,color='b'):
    bar_1=np.array([[-0.15, -0.65, 0.20],
                    [-0.15, -0.35, 0.50]])
    bar_2=np.array([[-0.15, -0.65, 0.35],
                    [-0.15, -0.35, 0.65]])

    ax.plot(*bar_1.T,color=color,linewidth=3.0)
    ax.plot(*bar_2.T,color=color,linewidth=3.0)

def plot_one_traj(ax:Axes,xyz_traj,color='b',label_flag=False):
    if label_flag:
        ax.plot(*(xyz_traj.T),color=color,label='trajectory')
    else:
        ax.plot(*(xyz_traj.T),color=color)

def plot_target(ax:Axes,color='r',size=120):
    ax.scatter(-0.6,-0.5, 0.4 , color=color,s=size,marker="*",label='targets')
    ax.scatter(-0.6,-0.5, 0.5 , color=color,s=size,marker="*")
    ax.scatter(-0.6,-0.5, 0.6 , color=color,s=size,marker="*")

import sys,os
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
sys.path.append(os.path.abspath(os.getcwd()))
filepath=os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),'Data','user_study_arm_mj','user_0','traj_arm','trial_0')

savepath=os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),'Data','uav_figs')

def draw_traj_set(filenames,filepath,title='title',savepath=os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),'Data','arm_figs')):
    fig = plt.figure(figsize=(8, 6))
    fig.suptitle(title,fontsize=20, y=0.8)
    #ax_main.set_title(title,fontsize=20, y=0.8)
    ax = fig.add_subplot(121, projection='3d')
    #fig.subplots_adjust(bottom=-0.11)
    ax.set_box_aspect([1,1,1])
    ax.set_title(title,fontsize=20, y=0.8)
    ax.set_xlim(-0.7,0.4)
    ax.set_ylim(-1,0)
    ax.set_zlim(0,1)
    ax.set_xlabel("x",fontsize=20, labelpad=10)
    ax.set_ylabel("y",fontsize=20)
    ax.set_zlabel("z",fontsize=20)
    ax.tick_params(labelsize=12)
    ax.grid()
    ax.view_init(elev=5., azim=-140)
    label_flag=True
    for fname in filenames:
        traj_full=np.load(os.path.join(filepath,fname))
        plot_one_traj(ax,traj_full[:,0:3],color='b',label_flag=label_flag)
        label_flag=False

    plot_bars(ax,color="black")
    plot_target(ax)
    ax.legend(loc=(0.6, 0.3))
    savefile=os.path.join(savepath,title+'.png')
    plt.tight_layout()
    bbox = fig.bbox_inches.from_bounds(1, 1, 6, 4)
    #plt.savefig(savefile, bbox_inches=bbox)
    #plt.savefig(savefile)
    plt.show()

filenames_3=['trajectory_5_target_2_cnum_7.npy','trajectory_6_target_0_cnum_7.npy','trajectory_7_target_1_cnum_7.npy']
draw_traj_set(filenames=filenames_3,filepath=filepath,title='test')
