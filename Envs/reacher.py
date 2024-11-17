import numpy as np
import casadi as cd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class ReacherModel(object):
    def __init__(self, project_name='two-link robot arm'):
        self.project_name = project_name

    def initDyn(self, l1=None, m1=None, l2=None, m2=None, g=10):

        # declare system parameters
        parameter = []
        if l1 is None:
            self.l1 = cd.SX.sym('l1')
            parameter.append(self.l1)
        else:
            self.l1 = l1

        if m1 is None:
            self.m1 = cd.SX.sym('m1')
            parameter.append(self.m1)
        else:
            self.m1 = m1

        if l2 is None:
            self.l2 = cd.SX.sym('l2')
            parameter.append(self.l2)
        else:
            self.l2 = l2

        if m2 is None:
            self.m2 = cd.SX.sym('m2')
            parameter.append(self.m2)
        else:
            self.m2 = m2

        self.dyn_auxvar = cd.vcat(parameter)
        self.dt=0.1

        # set variable
        self.q1, self.dq1, self.q2, self.dq2 = cd.SX.sym('q1'), cd.SX.sym('dq1'), cd.SX.sym('q2'), cd.SX.sym('dq2')
        self.X = cd.vertcat(self.q1, self.q2, self.dq1, self.dq2)
        u1, u2 = cd.SX.sym('u1'), cd.SX.sym('u2')
        self.U = cd.vertcat(u1, u2)

        # Declare model equations (discrete-time)
        r1 = self.l1 / 2
        r2 = self.l2 / 2
        I1 = self.l1 * self.l1 * self.m1 / 12
        I2 = self.l2 * self.l2 * self.m2 / 12
        M11 = self.m1 * r1 * r1 + I1 + self.m2 * (self.l1 * self.l1 + r2 * r2 + 2 * self.l1 * r2 * cd.cos(self.q2)) + I2
        M12 = self.m2 * (r2 * r2 + self.l1 * r2 * cd.cos(self.q2)) + I2
        M21 = M12
        M22 = self.m2 * r2 * r2 + I2
        M = cd.vertcat(cd.horzcat(M11, M12), cd.horzcat(M21, M22))
        h = self.m2 * self.l1 * r2 * cd.sin(self.q2)
        C1 = -h * self.dq2 * self.dq2 - 2 * h * self.dq1 * self.dq2
        C2 = h * self.dq1 * self.dq1
        C = cd.vertcat(C1, C2)
        ddq = cd.mtimes(cd.inv(M), -C  + self.U)  # joint acceleration
        self.f = cd.vertcat(self.dq1, self.dq2, ddq)  # continuous state-space representation

        self.new_X = self.X + self.dt * self.f

        return cd.Function('dynamics',[self.X,self.U],[self.new_X])
    
    def get_step_cost(self, P_mat:np.ndarray, Q_mat:np.ndarray):
        target=np.array([np.pi/2,0,0,0])
        self.c=(self.X-target).T @ P_mat @ (self.X-target) + self.U.T @ Q_mat @ self.U
        return cd.Function('step_cost',[self.X,self.U],[self.c])
    
    def get_terminal_cost(self,T_mat:np.ndarray):
        target=np.array([np.pi/2,0,0,0])
        self.h=(self.X-target).T @ T_mat @ (self.X-target)
        return cd.Function('terminal_cost',[self.X],[self.h])
    

class Reacher_Env(object):
    def __init__(self,l_1,m_1,l_2,m_2) -> None:
        self.l_1=l_1
        self.l_2=l_2
        self.m_1=m_1
        self.m_2=m_2

        self.dt=0.1

        self.model = ReacherModel()
        self.dyn = self.model.initDyn(self.l_1,self.m_1,self.l_2,self.m_2)

    def set_init_state(self,x:np.ndarray):
        self.x_0=np.array(x)

        self.clear_traj()
        self.x_traj.append(self.x_0)

        self.curr_x=self.x_0

    def get_curr_state(self):
        return self.curr_x
    
    def step(self,u):
        self.u_traj.append(u)

        new_x=self.dyn(self.curr_x,u).full().flatten()


        self.curr_x=new_x
        self.x_traj.append(np.array(self.curr_x))

    def clear_traj(self):
        self.x_traj=[]
        self.u_traj=[]
    
    def get_arm_position(self,x):
        q1 = x[0]
        q2 = x[1]
        x1 = self.l_1 * np.cos(q1)
        y1 = self.l_1 * np.sin(q1)
        x2 = self.l_2 * np.cos(q1 + q2) + x1
        y2 = self.l_2 * np.sin(q1 + q2) + y1
        position = np.array(([x1, y1, x2, y2]))

        return position
    
    def show_animation(self):
        # cd.Function to update the position of the pendulum
        def update(x):
            # Update the position of the pendulum
            position = self.get_arm_position(x)
            #print(position)
            line.set_data([0, position[0],position[2]], [0, position[1],position[3]])
            return line,

        #obstacles
        obs_theta = np.linspace(np.pi/9,np.pi/3,10)
        obs_x = 1.5 * np.cos(obs_theta)
        obs_y = 2.5 * np.sin(obs_theta)
        # Create a figure and axis
        fig, ax = plt.subplots()
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_aspect('equal')
        for i in range(10):
            circ = plt.Circle((obs_x[i],obs_y[i]),max(0.05*(5-np.abs(i-5)),0.1),color='b')
            ax.add_patch(circ)
        ax.grid()

        # Initialize the pendulum
        line, = ax.plot([], [], 'o-', lw=2)

        # Set up the animation
        ani = FuncAnimation(fig, update, frames=self.x_traj, interval=int(1000*self.dt), blit=True)

        # Display the animation
        plt.show()

if __name__=='__main__':   
    env = Reacher_Env(1.0,1.0,1.0,1.0)
    env.set_init_state(np.array([-np.pi/2,0,0,0]))
    for i in range(200):
        print('x',env.get_curr_state())
        u= 0.3 * np.random.randn(2)
        print('u',u)
        env.step(u)
    #print(env.x_traj)
    env.show_animation()

    