import numpy as np
import casadi as cd

import mujoco
import mujoco.viewer

class Robot_Arm_model(object):
    def __init__(self,dt) -> None:
        self.dt=dt
        self.DHForm_p1 = cd.vertcat(
            cd.horzcat(0,0.333,0),
            cd.horzcat(0,0,-cd.pi/2),
            cd.horzcat(0,0.316,cd.pi/2),
            cd.horzcat(0.0825,0,cd.pi/2),
            cd.horzcat(-0.0825,0.384,-cd.pi/2),
            cd.horzcat(0,0,cd.pi/2),
            cd.horzcat(0.088,0,cd.pi/2),
            cd.horzcat(0,0.107,0), #flange
        )

    def get_dyn_f(self):
        self.x_t=cd.SX.sym('x_t',7)
        self.u_t=cd.SX.sym('u_t',7)
        self.x_t_1= self.dt*self.u_t + self.x_t
        return cd.Function('arm_dynamics', [self.x_t, self.u], [self.x_t_1])
    
def DHLine_to_Mat(a,d,alpha,theta):
    A_2=cd.vertcat(
        cd.horzcat(cd.cos(theta),-cd.sin(theta),0,0),
        cd.horzcat(cd.sin(theta),cd.cos(theta),0,0),
        cd.horzcat(0,0,1,d),
        cd.horzcat(0,0,0,1),
    )

    A_1=cd.vertcat(
        cd.horzcat(1,0,0,a),
        cd.horzcat(0,cd.cos(alpha),-cd.sin(alpha),0),
        cd.horzcat(0,cd.sin(alpha),cd.cos(alpha),0),
        cd.horzcat(0,0,0,1),
    )

    return A_2 @ A_1

def DHForm_to_Mat(DHForm):
    Trans_Mat=cd.DM(np.eye(4)) #for end effector to world
    num_joint=DHForm.shape[0]
    for i in np.arange(num_joint-1,-1,-1):
        Trans_Mat = DHLine_to_Mat(*DHForm[i]) @ Trans_Mat
    return Trans_Mat

if __name__=='__main__':
    test_DHForm=np.array([[1,0,0,np.pi/4],
                          [1,0,0,np.pi/4]])
    #viewer=mujoco.viewer.launch()
    q=DHForm_to_Mat(test_DHForm) @ np.array([0,0,0,1]).reshape(-1,1)
    print(q)