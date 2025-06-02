import numpy as np
import casadi as cd

def gen_poly_feats(Horizon=15,bias = -1, con_idx = 5):
    x_dim = 13
    u_dim = 4
    traj = cd.SX.sym('xi', (x_dim + u_dim) * Horizon + x_dim)
    x_pos = traj[con_idx * (x_dim + u_dim)]/15
    y_pos = traj[con_idx * (x_dim + u_dim) + 1]/5
    z_pos= traj[con_idx * (x_dim + u_dim) + 2]/10

    phi_list = [bias]

    phi_list.append(x_pos**3)
    phi_list.append(x_pos**2)
    phi_list.append(x_pos)
    phi_list.append(y_pos**3)
    phi_list.append(y_pos**2)
    phi_list.append(y_pos)
    phi_list.append(z_pos**3)
    phi_list.append(z_pos**2)
    phi_list.append(z_pos)
    phi_list.append(x_pos*y_pos)
    phi_list.append(x_pos*z_pos)
    phi_list.append(y_pos*z_pos)

    phi = cd.vertcat(*phi_list)
    return cd.Function('phi', [traj], [phi])

def gen_poly_feats_single(bias = -1):
    state_vec = cd.SX.sym('state', 13)
    x_pos = state_vec[0]/15
    y_pos = state_vec[1]/5
    z_pos= state_vec[2]/10

    phi_list = [bias]

    phi_list.append(x_pos**3)
    phi_list.append(x_pos**2)
    phi_list.append(x_pos)
    phi_list.append(y_pos**3)
    phi_list.append(y_pos**2)
    phi_list.append(y_pos)
    phi_list.append(z_pos**3)
    phi_list.append(z_pos**2)
    phi_list.append(z_pos)
    phi_list.append(x_pos*y_pos)
    phi_list.append(x_pos*z_pos)
    phi_list.append(y_pos*z_pos)

    phi = cd.vertcat(*phi_list)
    return cd.Function('phi', [state_vec], [phi])

def gen_poly_feats_v(Horizon=15,bias = -1, v_idx = 1):
    x_dim = 13
    u_dim = 4
    traj = cd.SX.sym('xi', (x_dim + u_dim) * Horizon + x_dim)

    vx = traj[v_idx * (x_dim + u_dim) + 3]**2
    vy = traj[v_idx * (x_dim + u_dim) + 4]**2
    vz = traj[v_idx * (x_dim + u_dim) + 5]**2

    phi_list = [bias, vx, vy, vz]
    phi = cd.vertcat(*phi_list)
    return cd.Function('phi', [traj], [phi])

def gen_poly_feats_single_v(bias = -1):
    state_vec = cd.SX.sym('state', 13)

    vx = state_vec[3]**2
    vy = state_vec[4]**2
    vz = state_vec[5]**2

    phi_list = [bias, vx, vy, vz]
    phi = cd.vertcat(*phi_list)
    return cd.Function('phi', [state_vec], [phi])

def gen_poly_feats_v_full(Horizon=15,bias = -1, start_idx = 1, end_idx = 7):
    x_dim = 13
    u_dim = 4
    traj = cd.SX.sym('xi', (x_dim + u_dim) * Horizon + x_dim)
    phi_list = [bias]
    for idx in range(start_idx, end_idx + 1):
        vx = traj[idx * (x_dim + u_dim) + 3]**2
        vy = traj[idx * (x_dim + u_dim) + 4]**2
        vz = traj[idx * (x_dim + u_dim) + 5]**2
        v2 = vx + vy + vz
        phi_list.append(v2)

    phi = cd.vertcat(*phi_list)
    return cd.Function('phi', [traj], [phi])

