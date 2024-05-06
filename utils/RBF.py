import numpy as np
import casadi as cd

from utils.Casadi_Quat import Angle_Axis_Quat,Quat_mul,q_dist_1,q_dist_2

def IM_rbf_xy(x,y,x_c:float,y_c:float,epsilon:float):
    return 1/(1 + epsilon**2 * ((x-x_c)**2 + (y-y_c)**2))

def gau_rbf_xyz(x,y,z,x_c:float,y_c:float,z_c:float,epsilon:float):
    return cd.exp(- epsilon**2 * ((x-x_c)**2 + (y-y_c)**2 + (z-z_c)**2))

def gau_rbf_xy(x,y,x_c:float,y_c:float,epsilon:float):
    return cd.exp(-epsilon**2 * ((x-x_c)**2 + (y-y_c)**2))

def gau_rbf_xz(x,z,x_c:float,z_c:float,epsilon:float):
    return cd.exp(- epsilon**2 * ((x-x_c)**2 + (z-z_c)**2))

def IM_rbf_xyz(x,y,z,x_c:float,y_c:float,z_c:float,epsilon:float):
    return 1/(1 + epsilon**2 * ((x-x_c)**2 + (y-y_c)**2 + (z-z_c)**2))

def generate_phi_rbf(Horizon=20,X_c=np.linspace(3, 7, 5),Y_c=np.linspace(3, 7, 5),Z_c=None, 
                     epsilon=1.2,bias=-5,mode='gau_rbf_xy'):
    x_dim = 13
    u_dim = 4
    traj = cd.SX.sym('xi', (x_dim + u_dim) * Horizon + x_dim)
    x_pos = traj[5 * (x_dim + u_dim)]
    y_pos = traj[5 * (x_dim + u_dim) + 1]
    z_pos= traj[5 * (x_dim + u_dim) + 2]

    phi_list = []
    phi_list.append(bias)  # -4:16
    if mode=='gau_rbf_xy':
        grid_x, grid_y = np.meshgrid(X_c, Y_c)
        grid_x = grid_x.reshape(-1, 1)
        grid_y = grid_y.reshape(-1, 1)
        centers = np.concatenate([grid_x, grid_y], axis=1)
        for center in centers:
            print(center)
            phi_i = gau_rbf_xy(x_pos, y_pos, center[0], center[1], epsilon)
            phi_list.append(phi_i)

        phi = cd.vertcat(*phi_list)
        return cd.Function('phi', [traj], [phi])
    
    if mode=='IM_rbf_xy':
        grid_x, grid_y = np.meshgrid(X_c, Y_c)
        grid_x = grid_x.reshape(-1, 1)
        grid_y = grid_y.reshape(-1, 1)
        centers = np.concatenate([grid_x, grid_y], axis=1)
        for center in centers:
            print(center)
            phi_i = IM_rbf_xy(x_pos, y_pos, center[0], center[1], epsilon)
            phi_list.append(phi_i)

        phi = cd.vertcat(*phi_list)
        return cd.Function('phi', [traj], [phi])
    
    if mode=='gau_rbf_xyz':
        grid_x, grid_y, grid_z = np.meshgrid(X_c, Y_c ,Z_c, indexing='ij')
        grid_x = grid_x.reshape(-1, 1)
        grid_y = grid_y.reshape(-1, 1)
        grid_z = grid_z.reshape(-1, 1)
        #print(Z_c)
        #print(grid_z)
        centers = np.concatenate([grid_x, grid_y, grid_z], axis=1)
        for center in centers:
            print(center)
            phi_i = gau_rbf_xyz(x_pos, y_pos, z_pos, center[0], center[1], center[2], epsilon)
            phi_list.append(phi_i)

        phi = cd.vertcat(*phi_list)
        return cd.Function('phi', [traj], [phi])
    
    if mode=='gau_rbf_xyz_cum':
        grid_x, grid_y, grid_z = np.meshgrid(X_c, Y_c ,Z_c, indexing='ij')
        grid_x = grid_x.reshape(-1, 1)
        grid_y = grid_y.reshape(-1, 1)
        grid_z = grid_z.reshape(-1, 1)
        #print(Z_c)
        #print(grid_z)
        centers = np.concatenate([grid_x, grid_y, grid_z], axis=1)
        for center in centers:
            print(center)
            phi_i=0
            disc=1
            for i in range(10):
                x_pos_cum = traj[(5+i) * (x_dim + u_dim)]
                y_pos_cum = traj[(5+i) * (x_dim + u_dim) + 1]
                z_pos_cum = traj[(5+i) * (x_dim + u_dim) + 2]
                phi_i += disc * gau_rbf_xyz(x_pos_cum, y_pos_cum, z_pos_cum, center[0], center[1], center[2], epsilon)
                disc *= 0.9
            phi_list.append(phi_i)

        phi = cd.vertcat(*phi_list)
        return cd.Function('phi', [traj], [phi])
    
    if mode=='gau_rbf_sep_cum':
        grid_x, grid_y = np.meshgrid(X_c, Y_c)
        grid_x = grid_x.reshape(-1, 1)
        grid_y = grid_y.reshape(-1, 1)
        centers = np.concatenate([grid_x, grid_y], axis=1)
        for center in centers:
            print(center)
            phi_i=0
            disc=1
            for i in range(10):
                x_pos_cum = traj[(5+i) * (x_dim + u_dim)]
                y_pos_cum = traj[(5+i) * (x_dim + u_dim) + 1]
                z_pos_cum = traj[(5+i) * (x_dim + u_dim) + 2]
                phi_i += disc * gau_rbf_xy(x_pos_cum, y_pos_cum, center[0], center[1], epsilon)
                disc *= 0.8 #v2 0.9 v1
            phi_list.append(phi_i)

        grid_x, grid_z = np.meshgrid(X_c, Z_c)
        grid_x = grid_x.reshape(-1, 1)
        grid_z = grid_z.reshape(-1, 1)
        centers = np.concatenate([grid_x, grid_z], axis=1)
        for center in centers:
            print(center)
            phi_i=0
            disc=1
            for i in range(10):
                x_pos_cum = traj[(5+i) * (x_dim + u_dim)]
                y_pos_cum = traj[(5+i) * (x_dim + u_dim) + 1]
                z_pos_cum = traj[(5+i) * (x_dim + u_dim) + 2]
                phi_i += disc * gau_rbf_xy(x_pos_cum, z_pos_cum, center[0], center[1], epsilon)
                disc *= 0.8
            phi_list.append(phi_i)
        phi = cd.vertcat(*phi_list)
        return cd.Function('phi', [traj], [phi])

def gen_eval_rbf(weights,mode='gau_rbf_xy',X_c=np.linspace(3, 7, 5),Y_c=np.linspace(3, 7, 5),Z_c=None, 
                epsilon=1.2, bias=-5,):
    if mode=='gau_rbf_xy':
        pos=cd.SX.sym('x',2)
        x_pos=pos[0]
        y_pos=pos[1]
        phi_list = []
        phi_list.append(bias)  # -4:16
        #X_c = np.linspace(4, 6, 3)
        grid_x, grid_y = np.meshgrid(X_c, Y_c)
        grid_x = grid_x.reshape(-1, 1)
        grid_y = grid_y.reshape(-1, 1)
        centers = np.concatenate([grid_x, grid_y], axis=1)
        for center in centers:
            #print(center)
            phi_i = gau_rbf_xy(x_pos, y_pos, center[0], center[1], epsilon)
            phi_list.append(-phi_i)

        phi = cd.vertcat(*phi_list)
        expand_weights=cd.vertcat(cd.SX(1),weights)
        val= phi.T @ expand_weights
        return cd.Function('phi', [pos], [val])
    
    if mode=='IM_rbf_xy':
        pos=cd.SX.sym('x',2)
        x_pos=pos[0]
        y_pos=pos[1]
        phi_list = []
        phi_list.append(bias)  # -4:16
        #X_c = np.linspace(4, 6, 3)
        grid_x, grid_y = np.meshgrid(X_c, Y_c)
        grid_x = grid_x.reshape(-1, 1)
        grid_y = grid_y.reshape(-1, 1)
        centers = np.concatenate([grid_x, grid_y], axis=1)
        for center in centers:
            #print(center)
            phi_i = IM_rbf_xy(x_pos, y_pos, center[0], center[1], epsilon)
            phi_list.append(-phi_i)

        phi = cd.vertcat(*phi_list)
        expand_weights=cd.vertcat(cd.SX(1),weights)
        val= phi.T @ expand_weights
        return cd.Function('phi', [pos], [val])

#######################################################

def rbf_general(epsilon,dist):
    return cd.exp(- epsilon**2 * (dist**2))

def quat_rbf(epsilon,quat_center,q):
    return rbf_general(epsilon,q_dist_1(q,quat_center))

def sigmoid(x,softness=1):
    return 1/(1+cd.exp(-softness*x))

def generate_rbf_quat(Horizon,x_center,x_half,ref_axis,num,bias=-2,epsilon=1,mode='default'):
    x_dim=7
    u_dim=6

    phi_list = []
    phi_list.append(bias)  # -4:16

    traj = cd.SX.sym('xi', (x_dim + u_dim) * Horizon + x_dim)
    org_quat=np.array([0.0,-0.707,0.0,0.707])

    if mode=='default':
        next_quat=traj[x_dim+u_dim+3:x_dim+u_dim+7]
        next_x=traj[x_dim+u_dim]
        for theta in np.linspace(-np.pi/4,np.pi*3/4,num):
            rot_quat=Angle_Axis_Quat(theta,ref_axis)
            dst_quat=Quat_mul(rot_quat,org_quat)
            phi_list.append(sigmoid(x_half**2 - (next_x-x_center)**2,softness=50)*quat_rbf(epsilon,dst_quat,next_quat))

        phi = cd.vertcat(*phi_list)
        return cd.Function('phi', [traj], [phi])
    
    if mode=='cumulative':
        for theta in np.linspace(-np.pi/4,np.pi*3/4,num):
            rot_quat=Angle_Axis_Quat(theta,ref_axis)
            dst_quat=Quat_mul(rot_quat,org_quat)
            phi_i=0
            disc=1
            for i in range(5):
                next_quat=traj[(i+1)*(x_dim+u_dim)+3:(i+1)*(x_dim+u_dim)+7]
                next_x = traj[(i+1)*(x_dim+u_dim)]
                phi_i += sigmoid(x_half**2 - (next_x-x_center)**2,softness=50)*quat_rbf(epsilon,dst_quat,next_quat) * disc
                disc *= 0.9
            phi_list.append(phi_i)

        phi = cd.vertcat(*phi_list)
        return cd.Function('phi', [traj], [phi])
    
def generate_rbf_quat_z(Horizon,x_center,x_half,ref_axis,num_q,z_min,z_max
                        ,num_z,bias=-2,epsilon_q=1,epsilon_z=1,mode='default'):
    x_dim=7
    u_dim=6

    phi_list_p2 = []

    traj = cd.SX.sym('xi', (x_dim + u_dim) * Horizon + x_dim)
    quat_func=generate_rbf_quat(Horizon,x_center,x_half,ref_axis,num_q
                                ,bias=bias,epsilon=epsilon_q,mode=mode)
    phi_p1=quat_func(traj)

    if mode=='default':
        next_z=traj[x_dim+u_dim+2:x_dim+u_dim+3]
        next_x=traj[x_dim+u_dim]
        for z in np.linspace(z_min,z_max,num_z):
            phi_list_p2.append(sigmoid(x_half**2 - (next_x-x_center)**2,softness=50)*rbf_general(epsilon=epsilon_z,dist=next_z-z))

        phi_p2 = cd.vertcat(*phi_list_p2)
        phi = cd.vertcat(phi_p1,phi_p2)
        return cd.Function('phi', [traj], [phi])
    
    if mode=='cumulative':
        for z in np.linspace(z_min,z_max,num_z):
            phi_i=0
            disc=1
            for i in range(5):
                next_z=traj[(i+1)*(x_dim+u_dim)+2:(i+1)*(x_dim+u_dim)+3]
                next_x = traj[(i+1)*(x_dim+u_dim)]
                phi_i += sigmoid(x_half**2 - (next_x-x_center)**2,softness=50)*rbf_general(epsilon=epsilon_z,dist=next_z-z) * disc
                disc *= 0.9

            phi_list_p2.append(phi_i)

        phi_p2 = cd.vertcat(*phi_list_p2)
        phi = cd.vertcat(phi_p1,phi_p2)
        return cd.Function('phi', [traj], [phi])