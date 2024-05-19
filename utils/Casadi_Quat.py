import casadi as cd
import numpy as np

def Quat_Rot(q):
    Rot = cd.vertcat(
        cd.horzcat(1 - 2 * (q[2] ** 2 + q[3] ** 2), 2 * (q[1] * q[2] - q[0] * q[3]), 2 * (q[1] * q[3] + q[0] * q[2])),
        cd.horzcat(2 * (q[1] * q[2] + q[0] * q[3]), 1 - 2 * (q[1] ** 2 + q[3] ** 2), 2 * (q[2] * q[3] - q[0] * q[1])),
        cd.horzcat(2 * (q[1] * q[3] - q[0] * q[2]), 2 * (q[2] * q[3] + q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2))
    )
    return Rot

def Rot_Quat(R):
    eta=0.5*cd.sqrt(cd.trace(R)+1)
    eps_0=0.5*cd.sign(R[2,1]-R[1,2])*cd.sqrt(R[0,0]-R[1,1]-R[2,2]+1)
    eps_1=0.5*cd.sign(R[0,2]-R[2,0])*cd.sqrt(R[1,1]-R[2,2]-R[0,0]+1)
    eps_2=0.5*cd.sign(R[1,0]-R[0,1])*cd.sqrt(R[2,2]-R[0,0]-R[1,1]+1)
    return cd.vertcat(eta,eps_0,eps_1,eps_2)

def q_dist_1(q_1, q_2):
    I = cd.DM(np.eye(3))
    return 0.5 * cd.trace(I - Quat_Rot(q_2).T @ Quat_Rot(q_1))

def q_dist_2(q_1,q_2):
    return 1 - (q_1.T @ q_2)**2

def Quat_mul(q_1,q_2):
    q_1_cd=cd.DM(q_1)
    q_2_cd=cd.DM(q_2)
    #print(q_1_cd.shape)
    #print(q_2_cd.shape)
    eta_1=q_1_cd[0]
    eta_2=q_2_cd[0]
    eps_1=q_1_cd[1:]
    eps_2=q_2_cd[1:]

    return cd.vertcat(eta_1*eta_2-eps_1.T @ eps_2,
                      eta_1*eps_2+eta_2*eps_1 + cd.cross(eps_1,eps_2))

def Angle_Axis_Quat(angle,axis):
    return cd.vertcat(cd.cos(angle/2),cd.sin(angle/2)*axis)