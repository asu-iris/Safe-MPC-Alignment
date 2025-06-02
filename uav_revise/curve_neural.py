import numpy as np
import torch
import sys,os
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
sys.path.append(os.path.abspath(os.getcwd()))
from torch_sdf import NeRFMLP, sdf_param

from Envs.UAV import UAV_env_mj, UAV_model
from Envs.UAV_torch import UAVModelTorch
from Solvers.OCsolver_linearize import ocsolver_lin
from Solvers.Cutter_torch import cutter_torch, Neural_phi
from Solvers.MVEsolver import mvesolver

from torch.autograd.functional import jacobian
import mujoco.viewer

def calc_g(model, traj_xu, weights ,idx):
    x_dim = 13
    u_dim = 4

    traj_tensor = torch.tensor(traj_xu,dtype = torch.float32, device = "cuda", requires_grad=True)
    xpos = traj_tensor[(x_dim+u_dim)*idx:(x_dim+u_dim)*idx + 3]/15
    feats, _ = model(xpos)
    feats = torch.concat((0.1 * torch.ones(1,dtype=torch.float32,device="cuda"),feats))

    w_tensor = torch.tensor(weights,dtype = torch.float32, device = "cuda")
    w_tensor = torch.cat((torch.ones(1,dtype = torch.float32, device = "cuda"),w_tensor))
    # print("weights", w_tensor)
    g = (feats * w_tensor).sum()

    g.backward()
    g_grad = traj_tensor.grad

    return g.detach().cpu().numpy(), g_grad.cpu().numpy()

def calc_g_gt(model,traj_xu,idx):
    x_dim = 13
    u_dim = 4

    traj_tensor = torch.tensor(traj_xu,dtype = torch.float32, device = "cuda", requires_grad=True)
    xpos = traj_tensor[(x_dim+u_dim)*idx:(x_dim+u_dim)*idx + 3]/15
    _, g = model(xpos)

    g.backward()
    g_grad = traj_tensor.grad

    return g.detach().cpu().numpy(), g_grad.cpu().numpy()

def phi_func_idx(model, traj_xu,idx):
    x_dim = 13
    u_dim = 4
    xpos = traj_xu[(x_dim+u_dim)*idx:(x_dim+u_dim)*idx + 3]/15
    feats, _ = model(xpos)
    feats = torch.concat((0.1 * torch.ones(1,dtype=torch.float32,device="cuda"),feats))
    return feats

Horizon =  10  # 25
uav_params = {'gravity': 9.8, 'm': 0.1, 'J_B': 0.01 * np.eye(3), 'l_w': 1.2, 'dt': 0.15, 'c': 1}
filepath = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                        'mujoco_uav', 'bitcraze_crazyflie_2',
                        'scene_revise_geom_nn.xml')
print('path', filepath)
uav_env = UAV_env_mj(filepath, lock_flag=True)
uav_model = UAV_model(**uav_params)
dyn_f = uav_model.get_dyn_f()

torch_model = UAVModelTorch(**uav_params)

step_cost_vec = np.array([0.05, 400, 1, 5, 0.01]) * 1e-2
step_cost_f = uav_model.get_step_cost_param(step_cost_vec)
term_cost_vec = np.array([10, 100, 1, 5]) * 1e0
term_cost_f = uav_model.get_terminal_cost_param(term_cost_vec)

controller = ocsolver_lin('uav control')
controller.set_state_param(13, None, None)
controller.set_ctrl_param(4, [-1e10, -1e10, -1e10, -1e10], [1e10, 1e10, 1e10, 1e10])
controller.set_dyn(dyn_f)
controller.set_step_cost(step_cost_f)
controller.set_term_cost(term_cost_f)
controller.set_gamma(50)
controller.construct_prob(horizon=Horizon)

hypo_lbs = -0.2 * np.ones(16)
hypo_ubs = 0.1 * np.ones(16)

# hypo_lbs = -0.3 * np.ones(16)
# hypo_ubs = 0.2 * np.ones(16)


hb_calculator = cutter_torch('uav cut')
hb_calculator.from_controller(controller)
hb_calculator.construct_graph(horizon=Horizon)

mve_calc = mvesolver('uav_mve', 16)
mve_calc.set_init_constraint(hypo_lbs, hypo_ubs)

geom_model = NeRFMLP(sdf_param())
geom_model.load_state_dict(torch.load("../Data/uav_revise/sdf_weights.pt"))
geom_model.to("cuda")

#model test
init_r = np.array([15.0,0.0,10.0])
init_r_tensor = torch.tensor(init_r,dtype=torch.float32, device="cuda").flatten()
print("sdf",geom_model(init_r_tensor/15)[1] * 15)

phi_func = lambda t: phi_func_idx(geom_model,t,5)
phi_module = Neural_phi(phi_func,torch_model.transition,13,4,Horizon)

init_r = np.array([0.0,0.0,5.0]).reshape(-1, 1)
init_v = np.zeros((3, 1))
init_q = np.reshape(np.array([1, 0, 0, 0]), (-1, 1))
# print(Quat_Rot(init_q))
init_w_B = np.zeros((3, 1))
init_x = np.concatenate([init_r, init_v, init_q, init_w_B], axis=0)
init_r_tensor = torch.tensor(init_r,dtype=torch.float32, device="cuda").flatten()

zero_u = np.zeros(4)
ones_u = np.ones(4) * 0.25
center_traj = np.concatenate([np.concatenate((init_x.flatten(),ones_u))]*Horizon)
center_traj = np.concatenate((center_traj,init_x.flatten()))
center_tensor = torch.tensor(center_traj,dtype=torch.float32, device="cuda")

target_r = np.array([15.0,0.0,10.0])
target_x = np.concatenate([target_r.reshape(-1,1), init_v, init_q, init_w_B], axis=0)
target_r_tensor = torch.tensor(target_r,dtype=torch.float32, device="cuda").flatten()

#correction interface
x_thrust = np.array([-1, 0, 1, 0])
y_thrust = np.array([0, 1, 0, -1])
z_thrust = np.array([1, 1, 1, 1])

viewer = mujoco.viewer.launch_passive(uav_env.model,uav_env.data)

curve_points = np.load("../Data/uav_revise/curve_points.npy")
circle_points = np.load("../Data/uav_revise/circle_points.npy")

single_phi_func = lambda r: torch.cat((0.1*torch.ones(1,dtype = torch.float32, device = "cuda"),geom_model(r/15)[0])).cpu().numpy()
with torch.no_grad():
    phi_init = single_phi_func(init_r_tensor)
    phi_target = single_phi_func(target_r_tensor)

mve_calc.add_constraint(phi_init[1:], -phi_init[0])
mve_calc.add_constraint(phi_target[1:], -phi_target[0])

# subsample_points = curve_points[::40]
# for p in subsample_points:
#     p_tensor = torch.tensor(p,dtype=torch.float32, device="cuda").flatten()
#     with torch.no_grad():
#         phi_p = geom_model(p_tensor/15)[0].cpu().numpy()
#     mve_calc.add_constraint(phi_p[1:], - 0.1 * phi_p[0])
        
weights,C = mve_calc.solve()
print(weights)
input()

traj_cnt = 0
while True:
    uav_env.set_init_state(init_x)
    center_traj = np.concatenate([np.concatenate((init_x.flatten(),ones_u))]*Horizon)
    center_traj = np.concatenate((center_traj,init_x.flatten()))
    controller.reset_warmstart()
    traj = []
    for i in range(5000):
        x = uav_env.get_curr_state()

        uav_p = x[0:3].flatten()
        dists = np.linalg.norm(circle_points - uav_p,axis = 2)
        point_id = np.unravel_index(np.argmin(dists), dists.shape)
        dist = dists[point_id]
        point_id = np.unravel_index(np.argmin(dists), dists.shape)

        direction = curve_points[point_id[0]]-uav_p

        # if i == 0:
        center_x = x.copy()
        if i > 0:
            center_x[0:3] = uav_p.reshape(-1,1)
        center_x[3:6] = 0
        center_x[6] = 1
        center_x[7:] = 0
        center_traj = np.concatenate([np.concatenate((center_x.flatten(),ones_u))]*Horizon)
        center_traj = np.concatenate((center_traj,center_x.flatten()))
        center_tensor = torch.tensor(center_traj,dtype=torch.float32, device="cuda")

        g_center,g_grad = calc_g(geom_model, center_tensor, weights, idx=5)
        # g_center,g_grad = calc_g_gt(geom_model, center_tensor, idx=5)
        print("drone pos", uav_p)
        print("g center",g_center)
        print("g grad", g_grad)
        # input()

        initial_traj = np.concatenate([np.concatenate((init_x.flatten(),ones_u))]*Horizon)
        u = controller.control(x, center_traj,g_center, g_grad, target_r=target_r, initial_guess=center_traj[13:])
    
        print(u)
        uav_env.step(u)
        traj.append(uav_env.get_curr_state())
        # input()

        if dist<=0.7:
            corr = y_thrust * direction[1] + z_thrust*direction[2]
            # corr = y_thrust * 1
            print("direction",direction)
            corr_e = np.concatenate([corr.reshape(-1, 1), np.zeros((4 * (Horizon - 1), 1))])
            opt_traj_tensor = torch.tensor(controller.opt_traj,dtype=torch.float32, device="cuda").flatten()

            with torch.no_grad():
                phi = phi_module.calc_phi(opt_traj_tensor).cpu().numpy()
                phi_jac = phi_module.calc_phi_jac(opt_traj_tensor).cpu().numpy()

            print("phi",phi.shape)

            h, b, h_phi, b_phi = hb_calculator.calc_planes(x, controller.opt_traj,
                                                            human_corr=corr_e,
                                                            target_r=target_r,
                                                            phi = phi,
                                                            phi_jacobi=phi_jac,
                                                            gamma = 50)
            # print(h.shape, b.shape)
            # print(h_phi.shape, b_phi.shape)
            mve_calc.add_constraint(h, b)
            mve_calc.add_constraint(h_phi, b_phi)

            weights, C = mve_calc.solve()
            print("learned", weights)
            # input()
            break

        
        viewer.sync()

    # np.save(os.path.join('../Data/uav_revise/traj_neural', 'traj_{}.npy'.format(traj_cnt)),np.array(traj))
    traj_cnt+=1
    # input()
        # center_traj = controller.opt_traj
        # center_tensor = torch.tensor(center_traj,dtype=torch.float32, device="cuda")