import torch
import torch.nn.functional as F

class UAVModelTorch:
    def __init__(self, gravity, m, J_B, l_w, dt, c):
        self.g = gravity
        self.g_I = torch.tensor([0.0, 0.0, -self.g], dtype = torch.float32, device="cuda")
        self.m = m
        self.J_B = torch.tensor(J_B, dtype = torch.float32, device="cuda")
        self.J_B_inv = torch.linalg.inv(self.J_B)
        self.l_w = l_w
        self.dt = dt
        self.K_tau = torch.tensor([
            [0, -l_w / 2, 0, l_w / 2],
            [-l_w / 2, 0, l_w / 2, 0],
            [c, -c, c, -c]
        ], dtype = torch.float32, device="cuda")

    def transition(self, x_t, u):
        r_I = x_t[0:3]
        v_I = x_t[3:6]
        q_BI = x_t[6:10]
        w_B = x_t[10:]

        # print(q_BI)
        R_I_B = quat_rot(q_BI)
        thrust = u.sum()
        f_B = torch.tensor([0.0, 0.0, 1.0], dtype=u.dtype, device=u.device) * thrust
        f_I = R_I_B @ f_B
        
        d_r_I = v_I
        d_v_I = self.g_I + f_I / self.m
        d_q = 0.5 * omega(w_B) @ q_BI
        d_w_B = self.J_B_inv @ (self.K_tau @ u - torch.cross(w_B, self.J_B @ w_B))

        r_I_1 = r_I + self.dt * d_r_I
        v_I_1 = v_I + self.dt * d_v_I
        q_BI_1 = q_BI + self.dt * d_q
        w_B_1 = w_B + self.dt * d_w_B

        x_t_1 = torch.cat([r_I_1, v_I_1, q_BI_1, w_B_1])
        return x_t_1


def quat_rot(q):
    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
    return torch.stack([
        torch.stack([1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)]),
        torch.stack([2*(q1*q2 + q0*q3), 1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)]),
        torch.stack([2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1 - 2*(q1**2 + q2**2)])
    ])


def omega(w):
    wx, wy, wz = w[0], w[1], w[2]
    # zeros = torch.zeros(1, dtype=w.dtype, device=w.device)
    zeros = torch.tensor(0, dtype=w.dtype, device=w.device)

    return torch.stack([
        torch.stack([zeros, -wx, -wy, -wz]),
        torch.stack([ wx, zeros,  wz, -wy]),
        torch.stack([ wy, -wz, zeros,  wx]),
        torch.stack([ wz,  wy, -wx, zeros])
    ], dim=0).squeeze(-1)

# test_quat = torch.tensor([1.0, 0.0, 0, 0], dtype = torch.float32, device="cuda")
# print(quat_rot(test_quat))

