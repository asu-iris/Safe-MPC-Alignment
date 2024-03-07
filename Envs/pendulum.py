import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from casadi import *

class Pendulum_Env(object):
    def __init__(self,g,l,m,d,dt) -> None:
        self.g=g
        self.l=l
        self.m=m
        self.d=d

        self.I= self.m * (self.l **2) /3
        self.dt=dt
        self.x_traj=[]
        self.u_traj=[]
        self.acc_traj=[]
        
        self.noise_flag=True
        self.noise_cov=np.array([[0.0001,0],
                                 [0,0.0001]])

    def set_init_state(self,x:np.ndarray):
        self.x_0=np.array(x)

        self.clear_traj()
        self.x_traj.append(self.x_0)

        self.curr_x=self.x_0

    def get_curr_state(self):
        return self.curr_x
    
    def set_noise(self, noise_flag, noise_cov=None):
        self.noise_flag=noise_flag
        if noise_cov:
            self.noise_cov(noise_cov)
        
    def step(self,u):
        self.u_traj.append(u)

        new_x=np.zeros(2)
        #angular_acc= -(self.g/self.l) * np.sin(self.curr_x[0]) - (self.d/(self.m * self.l **2) ) * self.curr_x[1]  + u/(self.m*self.l**2)
        angular_acc= (-0.5*self.l * self.m * self.g * np.sin(self.curr_x[0]) + u - self.d * self.curr_x[1]) /self.I
        self.acc_traj.append(-(self.g/self.l) * np.sin(self.curr_x[0]))
        new_x[0]=self.curr_x[0] + self.dt * self.curr_x[1] 
        new_x[1]=self.curr_x[1] + self.dt * angular_acc

        if self.noise_flag:
            new_x+=np.random.multivariate_normal([0,0],self.noise_cov)

        self.curr_x=new_x
        self.x_traj.append(np.array(self.curr_x))

    def clear_traj(self):
        self.x_traj=[]
        self.u_traj=[]

    def show_animation(self):
        # Function to update the position of the pendulum
        def update(x):
            # Update the position of the pendulum
            line.set_data([0, self.l*np.cos(x[0]-np.pi/2)], [0, self.l*np.sin(x[0]-np.pi/2)])
            return line,

        # Create a figure and axis
        fig, ax = plt.subplots()
        ax.set_xlim(-1.5*self.l, 1.5*self.l)
        ax.set_ylim(-1.5*self.l, 1.5*self.l)
        ax.set_aspect('equal')
        ax.grid()

        # Initialize the pendulum
        line, = ax.plot([], [], 'o-', lw=2)

        # Set up the animation
        ani = FuncAnimation(fig, update, frames=self.x_traj, interval=50, blit=True)

        # Display the animation
        plt.show()

    def show_motion_scatter(self):
        plt.figure()
        plt.scatter(np.array(self.x_traj)[:,0],np.array(self.x_traj)[:,1])
        plt.show()

class Pendulum_Model(object):
    def __init__(self,g,l,m,d,dt) -> None:
        self.g=g
        self.l=l
        self.m=m
        self.d=d

        self.I= self.m * (self.l **2) /3
        self.dt=dt

        assert self.g, 'missing g'
        assert self.l, 'missing l'
        assert self.m, 'missing m'
        assert self.d, 'missing d'
        assert self.dt, 'missing dt'

        self.alpha,self.dalpha=SX.sym('alpha'),SX.sym('dalpha')
        self.U=SX.sym('u')
        self.X=vertcat(self.alpha,self.dalpha)
        self.dX=vertcat(self.dalpha,(-0.5*self.l * self.m * self.g * np.sin(self.alpha) + self.U - self.d * self.dalpha) /self.I)

    def get_dyn_f(self):
        self.dyn=self.X + self.dt * self.dX
        return Function('dynamics',[self.X,self.U],[self.dyn])
    
    def get_step_cost(self, P_mat:np.ndarray, Q_val:float):
        target=np.array([np.pi,0])
        self.c=(self.X-target).T @ P_mat @ (self.X-target) + self.U * Q_val * self.U
        return Function('step_cost',[self.X,self.U],[self.c])
    
    def get_terminal_cost(self,T_mat:np.ndarray):
        target=np.array([np.pi,0])
        self.h=(self.X-target).T @ T_mat @ (self.X-target)
        return Function('terminal_cost',[self.X],[self.h])
"""
pdl=Pendulum_Env(10,1,1,0.4,0.05)
pdl.set_init_state(np.array([np.pi/4,0]))
for i in range(160):
    pdl.step(-4*pdl.curr_x[1])
    #pdl.step(0)

plt.figure()
plt.plot(np.array(pdl.x_traj)[:,0])
plt.show()
pdl.show_animation()

""" 
if __name__=='__main__':   
    p_model=Pendulum_Model(10,1,1,0.4,0.05)
    #print(p_model.casadi_dyn_f())
    step_func=p_model.get_step_cost(np.eye(2),0.4)
    print(step_func(np.array([np.pi/2,0]),1))
