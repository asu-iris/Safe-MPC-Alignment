import numpy as np
import casadi as cd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R
import mujoco


class UAV_env(object):
    def __init__(self, gravity, m, J_B, l_w, dt, c) -> None:
        self.g = gravity
        self.g_I = np.reshape(np.array([0, 0, -self.g]), (-1, 1))
        self.m = m
        self.J_B = J_B
        self.l_w = l_w
        self.dt = dt

        self.K_tau = np.array([[0, -self.l_w / 2, 0, self.l_w / 2],
                               [-self.l_w / 2, 0, self.l_w / 2, 0],
                               [c, -c, c, -c]])

    def set_init_state(self, x: np.ndarray):
        self.x_0 = np.array(x)

        self.clear_traj()
        self.x_traj.append(self.x_0.flatten())

        self.curr_x = self.x_0

    def get_curr_state(self):
        return np.copy(self.curr_x[:])

    def clear_traj(self):
        self.x_traj = []
        self.u_traj = []

    def step(self, u):  # u: T_1, T_2, T_3, T_4
        u = np.array(u).reshape((-1, 1))
        assert u.shape == (4, 1), "wrong control input dim"
        self.r_I = self.curr_x[0:3]
        self.v_I = self.curr_x[3:6]
        self.q_BI = self.curr_x[6:10]  # from body to the world!
        self.w_B = self.curr_x[10:]

        self.R_I_B = np.array(Quat_Rot(self.q_BI.flatten()))  # rotation matrix: body to the world
        self.R_B_I = self.R_I_B.T  # rotation matrix: world to body
        # print('R_B_I',self.R_B_I)

        thrust = u.T @ np.ones((4, 1))
        # print(thrust)
        f_I = self.R_I_B @ np.reshape(np.concatenate([np.zeros((2, 1)), thrust]), (-1, 1))
        # print(f_I)
        d_r_I = self.v_I
        d_v_I = self.g_I + f_I / self.m
        d_q = 0.5 * np.array(Omega(self.w_B.flatten())) @ self.q_BI

        d_w_B = np.linalg.inv(self.J_B) @ (
                self.K_tau @ u - np.reshape(np.cross(self.w_B.flatten(), (self.J_B @ self.w_B).flatten()), (-1, 1)))
        self.r_I += self.dt * d_r_I
        self.v_I += self.dt * d_v_I
        self.q_BI += self.dt * d_q
        self.q_BI /= np.linalg.norm(self.q_BI)
        self.w_B += self.dt * d_w_B

        self.curr_x = np.concatenate([self.r_I, self.v_I, self.q_BI, self.w_B], axis=0)
        self.x_traj.append(self.curr_x.flatten())

    def get_pos(self):
        return self.curr_x[0:3]

    def show_animation(self, flag_2d=False, center=(4, 4, 4), radius=2, mode=None):
        def draw_quadrotor(ax_3d, ax_2d, pos, quat, wing_length):
            # Extracting position and attitude information
            x, y, z = pos
            q0, q1, q2, q3 = quat

            # Defining quadrotor wings tips (IN BODY FRAME)
            # wing1_tip = np.array([x-wing_length, y, z])
            # wing2_tip = np.array([x+wing_length, y, z])
            # wing3_tip = np.array([x, y-wing_length, z])
            # wing4_tip = np.array([x, y+wing_length, z])

            wing1_tip = np.array([+wing_length / 2, 0, 0])
            wing2_tip = np.array([0, +wing_length / 2, 0])
            wing3_tip = np.array([-wing_length / 2, 0, 0])
            wing4_tip = np.array([0, -wing_length / 2, 0])

            # Rotate wing tips based on quaternion
            # rot_I_B_1 = R.from_quat([q1, q2, q3, q0]).as_matrix().T
            # print('1',rot_I_B_1)
            rot_I_B = np.array(Quat_Rot(quat.flatten()))  # body to the world
            # print('2',rot_I_B)
            wing1_tip = rot_I_B @ wing1_tip + pos
            wing2_tip = rot_I_B @ wing2_tip + pos
            wing3_tip = rot_I_B @ wing3_tip + pos
            wing4_tip = rot_I_B @ wing4_tip + pos

            # Plotting quadrotor wings
            ax_3d.scatter(x, y, z, color='black', marker='o')
            ax_3d.scatter(wing1_tip[0], wing1_tip[1], wing1_tip[2], color='r', marker='o')
            ax_3d.plot((x, wing1_tip[0]), (y, wing1_tip[1]), (z, wing1_tip[2]), color='r')

            ax_3d.scatter(wing2_tip[0], wing2_tip[1], wing2_tip[2], color='b', marker='o')
            ax_3d.plot((x, wing2_tip[0]), (y, wing2_tip[1]), (z, wing2_tip[2]), color='b')

            ax_3d.scatter(wing3_tip[0], wing3_tip[1], wing3_tip[2], color='r', marker='o')
            ax_3d.plot((x, wing3_tip[0]), (y, wing3_tip[1]), (z, wing3_tip[2]), color='r')

            ax_3d.scatter(wing4_tip[0], wing4_tip[1], wing4_tip[2], color='b', marker='o')
            ax_3d.plot((x, wing4_tip[0]), (y, wing4_tip[1]), (z, wing4_tip[2]), color='b')
            if flag_2d:
                # 2d projection
                ax_2d.scatter(x, y, color='black', marker='o')
                ax_2d.scatter(wing1_tip[0], wing1_tip[1], color='r', marker='o')
                ax_2d.plot((x, wing1_tip[0]), (y, wing1_tip[1]), color='r')

                ax_2d.scatter(wing2_tip[0], wing2_tip[1], color='r', marker='o')
                ax_2d.plot((x, wing2_tip[0]), (y, wing2_tip[1]), color='r')

                ax_2d.scatter(wing3_tip[0], wing3_tip[1], color='b', marker='o')
                ax_2d.plot((x, wing3_tip[0]), (y, wing3_tip[1]), color='b')

                ax_2d.scatter(wing4_tip[0], wing4_tip[1], color='b', marker='o')
                ax_2d.plot((x, wing4_tip[0]), (y, wing4_tip[1]), color='b')

            return

        def draw_circle(ax):
            # Create data for a sphere
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            c_x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
            c_y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
            c_z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))

            # Plot the sphere
            ax_3d.plot_surface(c_x, c_y, c_z, color='b', alpha=0.5)

        def draw_cylinder(ax, height=8, num_points=20):
            # Generate points for the top circle
            theta = np.linspace(0, 2 * np.pi, num_points)
            x_top = center[0] + radius * np.cos(theta)
            y_top = center[1] + radius * np.sin(theta)
            z_top = height * np.ones(num_points)

            # Generate points for the bottom circle
            x_bottom = center[0] + radius * np.cos(theta)
            y_bottom = center[1] + radius * np.sin(theta)
            z_bottom = 0 * np.ones(num_points)

            ax.plot_trisurf(x_top, y_top, z_top, color='b', alpha=0.5)
            ax.plot_trisurf(x_bottom, y_bottom, z_bottom, color='b', alpha=0.5)
            # Plot the sides of the cylinder
            # for i in range(num_points):
            #    ax.plot([x_top[i], x_bottom[i]], [y_top[i], y_bottom[i]], [z_top[i], z_bottom[i]], color='b')

            verts = []
            for i in range(num_points - 1):
                verts.append([(x_top[i], y_top[i], z_top[i]), (x_top[i + 1], y_top[i + 1], z_top[i + 1]),
                              (x_bottom[i + 1], y_bottom[i + 1], z_bottom[i + 1]),
                              (x_bottom[i], y_bottom[i], z_bottom[i])])
            verts = np.array(verts)
            poly = Poly3DCollection(verts, alpha=0.5, color='b')
            ax.add_collection3d(poly)

        def draw_trajectory(ax):
            arr_traj = np.array(self.x_traj)
            traj_x_poses = arr_traj[:, 0]
            traj_y_poses = arr_traj[:, 1]
            traj_z_poses = arr_traj[:, 2]
            ax.plot(traj_x_poses, traj_y_poses, traj_z_poses, color='r', alpha=0.5)

        fig = plt.figure()
        if flag_2d:
            ax_3d = fig.add_subplot(121, projection='3d')
            ax_2d = fig.add_subplot(122)

        else:
            ax_3d = fig.add_subplot(111, projection='3d')
            ax_2d = None

        ax_3d.set_box_aspect([1, 1, 1])
        positions = np.array(self.x_traj)[:, 0:3]
        quaternions = np.array(self.x_traj)[:, 6:10]

        def update(frame):
            ax_3d.clear()
            if flag_2d:
                ax_2d.clear()
            draw_quadrotor(ax_3d, ax_2d, positions[frame], quaternions[frame], self.l_w)

            if mode == 'ball':
                draw_circle(ax_3d)

            if mode == 'cylinder':
                draw_cylinder(ax_3d)

            draw_trajectory(ax_3d)
            ax_3d.set_xlim([-1, 8])
            ax_3d.set_ylim([-1, 8])
            ax_3d.set_zlim([-1, 8])
            ax_3d.set_xlabel('X')
            ax_3d.set_ylabel('Y')
            ax_3d.set_zlabel('Z')
            ax_3d.set_title('Quadrotor Trajectory')
            if flag_2d:
                ax_2d.set_xlim([-1, 8])
                ax_2d.set_ylim([-1, 8])
                ax_2d.set_xlabel('X')
                ax_2d.set_ylabel('Y')
                ax_2d.set_title('Quadrotor Trajectory (Top View)')
            return

        # Animate the quadrotor trajectory
        ani = FuncAnimation(fig, update, frames=len(positions), interval=1000 * self.dt)
        plt.show()

    def draw_curr_2d(self, ax):
        pos = self.curr_x.flatten()[0:3]
        quat = self.curr_x.flatten()[6:10]
        x, y, z = pos
        q0, q1, q2, q3 = quat

        # Defining quadrotor wings tips (IN BODY FRAME)
        # wing1_tip = np.array([x-wing_length, y, z])
        # wing2_tip = np.array([x+wing_length, y, z])
        # wing3_tip = np.array([x, y-wing_length, z])
        # wing4_tip = np.array([x, y+wing_length, z])

        wing1_tip = np.array([+self.l_w / 2, 0, 0])
        wing2_tip = np.array([0, -self.l_w / 2, 0])
        wing3_tip = np.array([-self.l_w / 2, 0, 0])
        wing4_tip = np.array([0, +self.l_w / 2, 0])

        # Rotate wing tips based on quaternion
        # rot_I_B_1 = R.from_quat([q1, q2, q3, q0]).as_matrix().T
        # print('1',rot_I_B_1)
        rot_I_B = np.array(Quat_Rot(quat.flatten()))  # body to the world
        # print('2',rot_I_B)
        wing1_tip = rot_I_B @ wing1_tip + pos
        wing2_tip = rot_I_B @ wing2_tip + pos
        wing3_tip = rot_I_B @ wing3_tip + pos
        wing4_tip = rot_I_B @ wing4_tip + pos

        ax.scatter(x, y, color='black', marker='o')
        ax.scatter(wing1_tip[0], wing1_tip[1], color='r', marker='o')
        ax.plot((x, wing1_tip[0]), (y, wing1_tip[1]), color='r')

        ax.scatter(wing2_tip[0], wing2_tip[1], color='g', marker='o')
        ax.plot((x, wing2_tip[0]), (y, wing2_tip[1]), color='b')

        ax.scatter(wing3_tip[0], wing3_tip[1], color='b', marker='o')
        ax.plot((x, wing3_tip[0]), (y, wing3_tip[1]), color='r')

        ax.scatter(wing4_tip[0], wing4_tip[1], color='y', marker='o')
        ax.plot((x, wing4_tip[0]), (y, wing4_tip[1]), color='b')

        arr_traj = np.array(self.x_traj)
        traj_x_poses = arr_traj[:, 0]
        traj_y_poses = arr_traj[:, 1]
        traj_z_poses = arr_traj[:, 2]
        ax.plot(traj_x_poses, traj_y_poses, color='r', alpha=0.5)
        return


class UAV_model(object):
    def __init__(self, gravity, m, J_B, l_w, dt, c) -> None:
        self.g = gravity
        self.g_I = cd.DM(np.array([0, 0, -self.g]))
        self.m = m
        self.J_B = J_B
        self.l_w = l_w
        self.dt = dt

        self.K_tau = cd.DM(np.array([[0, -self.l_w / 2, 0, self.l_w / 2],
                                     [-self.l_w / 2, 0, self.l_w / 2, 0],
                                     [c, -c, c, -c]]))

    def get_dyn_f(self):
        self.x_t = cd.SX.sym('x_t', 13)
        self.r_I = self.x_t[0:3]
        self.v_I = self.x_t[3:6]
        self.q_BI = self.x_t[6:10]  # from body to the world!
        self.w_B = self.x_t[10:]

        self.u = cd.SX.sym('u_t', 4)

        self.R_I_B = Quat_Rot(self.q_BI)  # rotation matrix: body to the world
        thrust = self.u.T @ cd.DM(np.ones((4, 1)))
        f_I = self.R_I_B @ cd.vertcat(0, 0, thrust)
        # print(f_I)
        d_r_I = self.v_I
        # print('d_r shape',d_r_I.shape)
        d_v_I = self.g_I + f_I / self.m
        # print('d_v shape',d_v_I.shape)
        d_q = 0.5 * Omega(self.w_B) @ self.q_BI
        # print('d_q shape',d_q.shape)
        d_w_B = cd.inv(self.J_B) @ (self.K_tau @ self.u - cd.cross(self.w_B, (self.J_B @ self.w_B)))
        # print('d_w shape',d_w_B.shape)

        self.r_I_1 = self.r_I + self.dt * d_r_I
        self.v_I_1 = self.v_I + self.dt * d_v_I
        self.q_BI_1 = self.q_BI + self.dt * d_q
        # self.q_BI_1 = self.q_BI_1 / cd.norm_2(self.q_BI_1 )
        self.w_B_1 = self.w_B + self.dt * d_w_B

        self.x_t_1 = cd.vertcat(self.r_I_1, self.v_I_1, self.q_BI_1, self.w_B_1)
        return cd.Function('uav_dynamics', [self.x_t, self.u], [self.x_t_1])

    def get_step_cost(self, param_vec: np.ndarray, target_pos=7 * np.ones(3)):
        target_r = cd.DM(target_pos)  # target pos is [5,5,5]
        target_v = cd.DM(np.zeros(3))  # target v is [0,0,0]
        target_q = cd.DM(np.array([1, 0, 0, 0]))  # target q is the same as world frame
        target_w = cd.DM(np.zeros(3))  # target w is [0,0,0]

        self.x_t = cd.SX.sym('x_t', 13)
        self.r_I = self.x_t[0:3]
        self.v_I = self.x_t[3:6]
        self.q_BI = self.x_t[6:10]  # from body to the world!
        self.w_B = self.x_t[10:]

        self.u = cd.SX.sym('u_t', 4)

        p_vec = cd.DM(param_vec)
        w_r = cd.diag([1, 1, 1.0])
        l_vec = cd.vertcat(cd.sumsqr(w_r @ (self.r_I - target_r)), cd.sumsqr(self.v_I - target_v),
                           q_dist(self.q_BI, target_q), \
                           cd.sumsqr(self.w_B - target_w), cd.sumsqr(self.u))
        self.c = p_vec.T @ l_vec
        return cd.Function('step_cost', [self.x_t, self.u], [self.c])

    def get_terminal_cost(self, param_vec: np.ndarray, target_pos=7 * np.ones(3)):
        target_r = cd.DM(target_pos)  # target pos is [5,5,5]
        target_v = cd.DM(np.zeros(3))  # target v is [0,0,0]
        target_q = cd.DM(np.array([1, 0, 0, 0]))  # target q is the same as world frame
        target_w = cd.DM(np.zeros(3))  # target w is [0,0,0]

        self.x_t = cd.SX.sym('x_t', 13)
        self.r_I = self.x_t[0:3]
        self.v_I = self.x_t[3:6]
        self.q_BI = self.x_t[6:10]  # from body to the world!
        self.w_B = self.x_t[10:]

        p_vec = cd.DM(param_vec)
        weight_mat_r = cd.diag([1, 1, 1])
        l_vec = cd.vertcat(cd.sumsqr(weight_mat_r @ (self.r_I - target_r)), cd.sumsqr(self.v_I - target_v),
                           q_dist(self.q_BI, target_q), \
                           cd.sumsqr(self.w_B - target_w))
        self.c_T = p_vec.T @ l_vec
        return cd.Function('terminal_cost', [self.x_t], [self.c_T])


class UAV_env_mj(object):
    def __init__(self, xml_path, wing_length=0.8, c=1) -> None:
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.x_traj = []
        self.u_traj = []
        self.l_w = wing_length

        self.K = np.array([[1, 1, 1, 1],
                           [0, -self.l_w / 2, 0, self.l_w / 2],
                           [-self.l_w / 2, 0, self.l_w / 2, 0],
                           [c, -c, c, -c]])

    def set_init_state(self, x: np.ndarray):
        mujoco.mj_resetData(self.model, self.data)
        x = x.flatten()
        self.data.qpos[0:3] = x[0:3]  # position
        self.data.qpos[3:7] = x[6:10]  # quaternion
        self.data.qvel[0:3] = x[3:6]  # vel
        self.data.qvel[3:6] = x[10:]  # angular vel

        self.clear_traj()

    def step(self, u):  # u:[T1,T2,T3,T4]
        inner_ctrl = self.K @ u.reshape(-1, 1)
        # rotation=R.from_quat(self.data.qpos[3:7]).as_matrix().T
        # inner_ctrl[1:]=rotation @ inner_ctrl[1:]
        self.data.ctrl = inner_ctrl.flatten()
        # print('ctrl',self.data.ctrl)
        mujoco.mj_step(self.model, self.data)
        self.x_traj.append(self.get_curr_state().flatten())

    def get_curr_state(self):
        x = self.data.qpos[0:3]
        v = self.data.qvel[0:3]
        q = self.data.qpos[3:7]
        # q=R.from_quat(q).inv().as_quat()
        w = self.data.qvel[3:6]

        return np.concatenate([x, v, q, w]).reshape(-1, 1)

    def clear_traj(self):
        self.x_traj = []
        self.u_traj = []


def Quat_Rot(q):
    # Rot = cd.vertcat(
    #        cd.horzcat(1 - 2 * (q[2] ** 2 + q[3] ** 2), 2 * (q[1] * q[2] + q[0] * q[3]), 2 * (q[1] * q[3] - q[0] * q[2])),
    #        cd.horzcat(2 * (q[1] * q[2] - q[0] * q[3]), 1 - 2 * (q[1] ** 2 + q[3] ** 2), 2 * (q[2] * q[3] + q[0] * q[1])),
    #       cd.horzcat(2 * (q[1] * q[3] + q[0] * q[2]), 2 * (q[2] * q[3] - q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2))
    #   )
    Rot = cd.vertcat(
        cd.horzcat(1 - 2 * (q[2] ** 2 + q[3] ** 2), 2 * (q[1] * q[2] - q[0] * q[3]), 2 * (q[1] * q[3] + q[0] * q[2])),
        cd.horzcat(2 * (q[1] * q[2] + q[0] * q[3]), 1 - 2 * (q[1] ** 2 + q[3] ** 2), 2 * (q[2] * q[3] - q[0] * q[1])),
        cd.horzcat(2 * (q[1] * q[3] - q[0] * q[2]), 2 * (q[2] * q[3] + q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2))
    )
    return Rot


def Omega(w):
    Omeg = cd.vertcat(
        cd.horzcat(0, -w[0], -w[1], -w[2]),
        cd.horzcat(w[0], 0, w[2], -w[1]),
        cd.horzcat(w[1], -w[2], 0, w[0]),
        cd.horzcat(w[2], w[1], -w[0], 0),
    )
    return Omeg


def q_dist(q_1, q_2):
    I = cd.DM(np.eye(3))
    return 0.5 * cd.trace(I - Quat_Rot(q_2).T @ Quat_Rot(q_1))


if __name__ == '__main__':
    init_r = np.zeros((3, 1))
    init_v = np.zeros((3, 1))
    init_q = np.reshape(np.array([1, 0, 0, 0]), (-1, 1))
    # print(Quat_Rot(init_q))
    init_w_B = np.zeros((3, 1))
    init_x = np.concatenate([init_r, init_v, init_q, init_w_B], axis=0)
    # print(init_x)
    uav_params = {'gravity': 10, 'm': 1, 'J_B': np.eye(3), 'l_w': 0.5, 'dt': 0.05, 'c': 1}
    uav_env = UAV_env(**uav_params)
    uav_env.set_init_state(init_x)

    uav_model = UAV_model(**uav_params)
    dyn_f = uav_model.get_dyn_f()

    step_cost_vec = np.array([0.05, 0.1, 1, 0.1, 0.05])
    step_cost_f = uav_model.get_step_cost(step_cost_vec)
    term_cost_vec = np.array([0.1, 0.2, 2, 0.1])
    term_cost_f = uav_model.get_terminal_cost(term_cost_vec)

    u = 2.6 * np.ones((4, 1))
    u[2] += 0.1
    u[0] -= 0.1

    for i in range(100):
        x = uav_env.get_curr_state()
        # r=x[0:3]
        # v=x[3:6]
        # q=x[6:10]
        # w=x[10:]
        # print('l1',cd.sumsqr(r.flatten()-5*np.ones(3)))
        # print('l2',cd.sumsqr(v.flatten()-np.zeros(3)))
        # print('l3',q_dist(q.flatten(),np.array([1,0,0,0])))
        # print('l4',cd.sumsqr(w.flatten()-np.zeros(3)))
        # print('l5',cd.sumsqr(u.flatten()))
        # print('1',x.T)
        uav_env.step(u)
        # print('2',x)
        x_1 = dyn_f(x, u)
        # print(uav_env.get_curr_state()-x_1)
        print('step cost', step_cost_f(x, u))

    print('term_cost', term_cost_f(uav_env.get_curr_state()))

    uav_env.show_animation(flag_2d=False)
    print(q_dist(np.array([1, 0, 0, 0]), np.array([-1, 0, 0, 0])))
