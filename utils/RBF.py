import numpy as np
import casadi as cd

def gau_rbf_xy(x,y,x_c:float,y_c:float,epsilon:float):
    return cd.exp(-epsilon**2 * ((x-x_c)**2 + (y-y_c)**2))

def IM_rbf_xy(x,y,x_c:float,y_c:float,epsilon:float):
    return 1/(1 + epsilon**2 * ((x-x_c)**2 + (y-y_c)**2))

def gau_rbf_xyz(x,y,z,x_c:float,y_c:float,z_c:float,epsilon:float):
    return cd.exp(- epsilon**2 * ((x-x_c)**2 + (y-y_c)**2 + (z-z_c)**2))

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
            for i in range(3):
                x_pos_cum = traj[(5+i) * (x_dim + u_dim)]
                y_pos_cum = traj[(5+i) * (x_dim + u_dim) + 1]
                z_pos_cum = traj[(5+i) * (x_dim + u_dim) + 2]
                phi_i += disc * gau_rbf_xyz(x_pos_cum, y_pos_cum, z_pos_cum, center[0], center[1], center[2], epsilon)
                disc *= 0.5
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

def generate_phi_rbf_cum(Horizon=20):
    x_dim = 13
    u_dim = 4
    traj = cd.SX.sym('xi', (x_dim + u_dim) * Horizon + x_dim)
    x_pos = traj[5 * (x_dim + u_dim)]
    y_pos = traj[5 * (x_dim + u_dim) + 1]
    z_pos_1 = traj[2 * (x_dim + u_dim) + 2]

    sum_phi = np.zeros(3)
    for t in range(Horizon):
        phi_list = []
        phi_list.append(-2)  # -4:16
        X_c = np.linspace(4, 6, 3)
        Y_c = np.linspace(4, 6, 3)
        grid_x, grid_y = np.meshgrid(X_c, Y_c)
        grid_x = grid_x.reshape(-1, 1)
        grid_y = grid_y.reshape(-1, 1)
        centers = np.concatenate([grid_x, grid_y], axis=1)
        for center in centers:
            print(center)
            phi_i = rbf(x_pos, y_pos, center[0], center[1], 1.5)
            phi_list.append(-phi_i)

        phi = cd.vertcat(*phi_list)
    return cd.Function('phi', [traj], [phi])