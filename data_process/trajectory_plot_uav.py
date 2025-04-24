import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

def plot_gate(ax:Axes,color='b'):
    points=np.array([[10,3.5,4],
                     [10,6.5,4],
                     [10,6.5,5.4],
                     [10,3.5,5.4],
                     [10,3.5,4]])
    ax.plot(*points.T,color=color,linewidth=3.0)

def plot_one_traj(ax:Axes,xyz_traj,color='b',label_flag=False):
    if label_flag:
        ax.plot(*(xyz_traj.T),color=color,label='trajectory')
    else:
        ax.plot(*(xyz_traj.T),color=color)

def plot_target(ax:Axes,color='r',size=120):
    ax.scatter(19,1,9,color=color,s=size,marker="*",label='targets')
    ax.scatter(19,5,9,color=color,s=size,marker="*")
    ax.scatter(19,9,9,color=color,s=size,marker="*")

import sys,os
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
sys.path.append(os.path.abspath(os.getcwd()))
filepath=os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),'Data','user_study_uav','user_0','trial_1')
savepath=os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),'Data','uav_figs')

def draw_traj_set(filenames,filepath,truncate_flags,title='title',savepath=os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),'Data','uav_figs')):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    #fig.subplots_adjust(bottom=-0.11)
    ax.set_box_aspect([2,1,1])
    #ax.set_title(title,fontsize=20, y=0.8)
    ax.set_xlim(0,20)
    ax.set_ylim(0,10)
    ax.set_zlim(0,10)
    ax.set_xlabel("x",fontsize=20, labelpad=10)
    ax.set_ylabel("y",fontsize=20)
    ax.set_zlabel("z",fontsize=20)
    ax.tick_params(labelsize=12)
    ax.grid()
    ax.view_init(elev=5., azim=-140)
    label_flag=True
    for fname,flag in zip(filenames,truncate_flags):
        traj_full=np.load(os.path.join(filepath,fname))
        if flag:
            traj_plot=traj_full[traj_full[:,0]<9.8]
            plot_one_traj(ax,traj_plot[:,0:3],color='b',label_flag=label_flag)
            if label_flag:
                ax.scatter(*traj_plot[-1,0:3],s=80,marker="X",color="g",label='collision')
            else:
                ax.scatter(*traj_plot[-1,0:3],s=80,marker="X",color="g")
        
        else:
            plot_one_traj(ax,traj_full[:,0:3],color='b',label_flag=label_flag)

        label_flag=False
        #plot_one_traj(ax,traj_full[:,0:3])
    plot_gate(ax,color="black")
    plot_target(ax)
    ax.legend(loc=(0.6, 0.3),fontsize=18)
    savefile=os.path.join(savepath,title+'.png')
    plt.tight_layout()
    bbox = fig.bbox_inches.from_bounds(1, 1, 6, 4)
    plt.savefig(savefile, bbox_inches=bbox)
    #plt.savefig(savefile)
    plt.show()

filenames_2=['trajectory_1_target_1_cnum_2.npy','trajectory_2_target_2_cnum_2.npy','trajectory_3_target_0_cnum_2.npy']
truncate_flags_2=[False,False,False]

filenames_6=['trajectory_5_target_2_cnum_6.npy','trajectory_6_target_0_cnum_6.npy','trajectory_7_target_1_cnum_6.npy']
truncate_flags_6=[True,True,True]

filenames_12=['trajectory_10_target_1_cnum_12.npy','trajectory_11_target_2_cnum_12.npy','trajectory_12_target_0_cnum_12.npy']
truncate_flags_12=[False,False,False]

draw_traj_set(filenames=filenames_2,
              filepath=filepath,
              truncate_flags=truncate_flags_2,
              title="After 2 Corrections")

draw_traj_set(filenames=filenames_6,
              filepath=filepath,
              truncate_flags=truncate_flags_6,
              title="After 6 Corrections")

draw_traj_set(filenames=filenames_12,
              filepath=filepath,
              truncate_flags=truncate_flags_12,
              title="After 12 Corrections")