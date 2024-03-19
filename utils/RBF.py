import numpy as np
import casadi as cd

def rbf(x,y,x_c:float,y_c:float,sigma:float):
    return cd.exp(-sigma * ((x-x_c)**2 + (y-y_c)**2))

def generate_phi_rbf(Horizon=20):
    x_dim = 13
    u_dim = 4
    traj = cd.SX.sym('xi', (x_dim + u_dim) * Horizon + x_dim)
    x_pos = traj[5 * (x_dim + u_dim)]
    y_pos = traj[5 * (x_dim + u_dim) + 1]
    z_pos_1 = traj[2 * (x_dim + u_dim) + 2]

    phi_list = []
    phi_list.append(-5.2)  # -4:16
    #X_c = np.linspace(4, 6, 3)
    X_c = np.linspace(3, 7, 5)
    Y_c = np.linspace(3, 7, 5)
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

def gen_eval_rbf(weights):
    pos=cd.SX.sym('x',2)
    x_pos=pos[0]
    y_pos=pos[1]
    phi_list = []
    phi_list.append(-5)  # -4:16
    #X_c = np.linspace(4, 6, 3)
    X_c = np.linspace(3, 7, 5)
    Y_c = np.linspace(3, 7, 5)
    grid_x, grid_y = np.meshgrid(X_c, Y_c)
    grid_x = grid_x.reshape(-1, 1)
    grid_y = grid_y.reshape(-1, 1)
    centers = np.concatenate([grid_x, grid_y], axis=1)
    for center in centers:
        #print(center)
        phi_i = rbf(x_pos, y_pos, center[0], center[1], 1.5)
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