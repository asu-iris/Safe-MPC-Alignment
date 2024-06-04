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

def plot_one_traj(ax:Axes,xyz_traj,color='b'):
    ax.plot(*(xyz_traj.T),color=color)

import sys,os
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
sys.path.append(os.path.abspath(os.getcwd()))
filepath=os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),'Data','user_study_uav','user_0','traj_uav','trial_0')
filenames=['trajectory_5_target_2_cnum_4.npy','trajectory_6_target_0_cnum_4.npy','trajectory_7_target_1_cnum_4.npy']

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect([2,1,1])
ax.set_xlim(0,20)
ax.set_ylim(0,10)
ax.set_zlim(0,10)
for fname in filenames:
    traj_full=np.load(os.path.join(filepath,fname))
    plot_one_traj(ax,traj_full[:,0:3])
plot_gate(ax,color="black")
plt.show()