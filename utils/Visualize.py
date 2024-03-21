import casadi as cd
import numpy as np
import sys,os
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
sys.path.append(os.path.abspath(os.getcwd()))
from Envs.UAV import UAV_env,Quat_Rot
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class uav_visualizer(object):
    def __init__(self,env:UAV_env,space_xyzmin,space_xyzmax,mode='cylinder') -> None:
        self.env=env
        self.space_xyzmin=np.array(space_xyzmin)
        self.space_xyzmax=np.array(space_xyzmax)
        self.mode=mode

    def render_init(self):
        if not hasattr(self, 'ax'):
            self.fig = plt.figure(figsize=(8, 6))
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.ax.set_xlabel('X (m)', fontsize=10, labelpad=5)
            self.ax.set_ylabel('Y (m)', fontsize=10, labelpad=5)
            self.ax.set_zlabel('Z (m)', fontsize=10, labelpad=5)
            self.ax.set_zlim(self.space_xyzmin[2], self.space_xyzmax[2])
            self.ax.set_ylim(self.space_xyzmin[1], self.space_xyzmax[1])
            self.ax.set_xlim(self.space_xyzmin[0], self.space_xyzmax[0])
            self.ax.set_box_aspect(aspect=self.space_xyzmax - self.space_xyzmin)
            
            #obstacle
            #c_1=plt.Circle(xy=(5.5,7),radius=2)
            #self.ax.add_patch(c_1)
            if self.mode=='cylinder':
                self.draw_cylinder()
            elif self.mode=='wall':
                self.draw_wall_with_hole_cubes()
        
        else:
            self.clear_uav()

        return self.ax
    
    def render_update(self,scale_ratio=1.0):
        plt.ion()
        if hasattr(self,'wing_1_line'):
            self.clear_uav()

        pos = self.env.curr_x.flatten()[0:3]
        quat = self.env.curr_x.flatten()[6:10]

        x,y,z=pos

        wing1_tip = np.array([+self.env.l_w/2, 0, 0])*scale_ratio
        wing2_tip = np.array([0, -self.env.l_w/2, 0])*scale_ratio
        wing3_tip = np.array([-self.env.l_w/2, 0, 0])*scale_ratio
        wing4_tip = np.array([0, +self.env.l_w/2, 0])*scale_ratio
            
        # Rotate wing tips based on quaternion
        #rot_I_B_1 = R.from_quat([q1, q2, q3, q0]).as_matrix().T
        #print('1',rot_I_B_1)
        rot_I_B = np.array(Quat_Rot(quat.flatten())) #body to the world
        #print('2',rot_I_B)
        wing1_tip = rot_I_B @ wing1_tip +pos
        wing2_tip = rot_I_B @ wing2_tip +pos
        wing3_tip = rot_I_B @ wing3_tip +pos
        wing4_tip = rot_I_B @ wing4_tip +pos

        self.body_point=self.ax.scatter(x, y, z, color='black', marker='o')
        self.wing_1_point=self.ax.scatter(wing1_tip[0], wing1_tip[1], wing1_tip[2], color='r', marker='o')
        self.wing_1_line,=self.ax.plot((x,wing1_tip[0]),(y, wing1_tip[1]),(z, wing1_tip[2]), color='r')

        self.wing_2_point=self.ax.scatter(wing2_tip[0], wing2_tip[1], wing2_tip[2], color='g', marker='o')
        self.wing_2_line,=self.ax.plot((x,wing2_tip[0]),(y, wing2_tip[1]),(z, wing2_tip[2]), color='b')

        self.wing_3_point=self.ax.scatter(wing3_tip[0], wing3_tip[1], wing3_tip[2], color='b', marker='o')
        self.wing_3_line,=self.ax.plot((x,wing3_tip[0]),(y, wing3_tip[1]),(z, wing3_tip[2]), color='r')

        self.wing_4_point=self.ax.scatter(wing4_tip[0], wing4_tip[1], wing4_tip[2], color='y', marker='o')
        self.wing_4_line,=self.ax.plot((x,wing4_tip[0]),(y, wing4_tip[1]),(z, wing4_tip[2]), color='b')

        plt.pause(0.01)

    def clear_uav(self):
        self.wing_1_line.remove()
        self.wing_2_line.remove()
        self.wing_3_line.remove()
        self.wing_4_line.remove()

        self.wing_1_point.remove()
        self.wing_2_point.remove()
        self.wing_3_point.remove()
        self.wing_4_point.remove()

        self.body_point.remove()

    def draw_cylinder(self, height=8, num_points=20):

        center = (5, 5, 3)
        radius = 2
        # Generate points for the top circle
        theta = np.linspace(0, 2 * np.pi, num_points)
        x_top = center[0] + radius * np.cos(theta)
        y_top = center[1] + radius * np.sin(theta)
        z_top = height * np.ones(num_points)

        # Generate points for the bottom circle
        x_bottom = center[0] + radius * np.cos(theta)
        y_bottom = center[1] + radius * np.sin(theta)
        z_bottom = 0 * np.ones(num_points)

        self.ax.plot_trisurf(x_top, y_top, z_top, color='b', alpha=0.5)
        self.ax.plot_trisurf(x_bottom, y_bottom, z_bottom, color='b', alpha=0.5)
        # Plot the sides of the cylinder
        # for i in range(num_points):
        #    ax.plot([x_top[i], x_bottom[i]], [y_top[i], y_bottom[i]], [z_top[i], z_bottom[i]], color='b')

        verts = []
        for i in range(num_points - 1):
            verts.append([(x_top[i], y_top[i], z_top[i]), (x_top[i + 1], y_top[i + 1], z_top[i + 1]),
                          (x_bottom[i + 1], y_bottom[i + 1], z_bottom[i + 1]), (x_bottom[i], y_bottom[i], z_bottom[i])])
        verts = np.array(verts)
        poly = Poly3DCollection(verts, alpha=0.5, color='b')
        self.ax.add_collection3d(poly)

    def draw_wall_with_hole_cubes(self):
        # Define wall dimensions
        wall_width = 1 #x
        wall_height = 10 #y
        wall_depth = 10 #z

        wall_x=5
        wall_y=0
        # Define cube dimensions
        cube_size = 1.0

        # Define hole dimensions
        hole_width = 4
        hole_height = 3
        hole_depth = 2
        hole_x = 3
        hole_y = 2
        hole_z = 4

        wall_color = (0.5, 0.5, 0.5, 0.6)
        # Plot cubes for the wall
        for x in np.arange(wall_x, wall_x+wall_width, cube_size):
            for y in np.arange(wall_y, wall_y+wall_height, cube_size):
                for z in np.arange(0, wall_depth, cube_size):
                    # Exclude cubes inside the hole
                    if not (hole_y <= y < hole_y + hole_height and hole_z <= z < hole_z + hole_depth):
                        self.ax.bar3d(x, y, z, cube_size, cube_size, cube_size, color=wall_color)





        
