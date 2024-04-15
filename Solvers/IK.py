import numpy as np
import casadi as cd
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
sys.path.append(os.path.abspath(os.getcwd()))
from Envs.robot_arm import DH_to_Mat

class IKsolver(object):
    def __init__(self,name) -> None:
        self.name=name
        self.print_level=0
        self.construct_prob()

    def construct_prob(self):
        target_pos=cd.SX.sym('target',3)
        q_init=cd.SX.sym('q',7)
        q_sol=cd.SX.sym('q',7)
        err= cd.sumsqr((DH_to_Mat(q_sol) @ cd.DM(np.array([0,0,0,1])))[0:3]-target_pos)

        opts = {'ipopt.print_level': self.print_level, 'ipopt.sb': 'yes', 'print_time': self.print_level}
        # prob = {'f': err, 'x': q_sol,
        #         'g': None, 'p': target_pos}
        prob = {'f': err, 'x': q_sol,
                 'p': target_pos}
        self.solver_func = cd.nlpsol('solver', 'ipopt', prob, opts)

    def solve(self,q_init,target_pos):
        initial_guess=q_init
        lbx = q_init - cd.pi/4
        ubx = q_init + cd.pi/4
        sol = self.solver_func(x0=initial_guess,
                               lbx=lbx, ubx=ubx,
                               lbg=[],
                               ubg=[],
                               p=target_pos)
        
        return sol['x'].full()
    

if __name__=='__main__':
    solver=IKsolver('test')
    ini_joint=np.zeros((7,1))
    ini_joint[0]=0
    ini_joint[3]=-1.5
    ini_joint[5]=1.5
    q_target=solver.solve(q_init=ini_joint,target_pos=np.array([0.45,0.14,0.54]))
    print(q_target.flatten())
    print(np.sign(q_target-ini_joint))
    print((DH_to_Mat(q_target) @ cd.DM(np.array([0,0,0,1])))[0:3])


