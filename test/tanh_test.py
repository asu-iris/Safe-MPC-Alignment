import os
import sys
import casadi as cd
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
sys.path.append(os.path.abspath(os.getcwd()))
from Envs.reacher import Reacher_Env,ReacherModel
from Solvers.OCsolver import ocsolver_v2
from Solvers.Cutter import cutter_v2
from Solvers.MVEsolver import mvesolver
import numpy as np
from matplotlib import pyplot as plt
from utils.Correction import Correction_Agent


def generate_phi_xy():
    x2 = cd.SX.sym('x2')
    y2 = cd.SX.sym('y2')

    phi_1 = cd.tanh(x2*y2)
    phi_2 = cd.tanh(y2**3)
    phi_3 = cd.tanh(x2**2)
    phi_4 = cd.tanh(y2**2)
    phi_5 = cd.tanh(x2)
    phi_6 = cd.tanh(y2)
    phi=cd.vertcat(cd.DM(-1),phi_1,phi_2,phi_3,phi_4,phi_5,phi_6)
    return  cd.Function('phi_xy',[x2,y2],[phi])


phi_func_xy=generate_phi_xy()
samples = np.random.uniform((-3,-3),(3,3),size=(10000,2))

x2=samples[:,0].reshape((1,-1))
y2=samples[:,1].reshape((1,-1))

phi_arr = phi_func_xy(x2,y2).full()
print(phi_arr.shape)
test_theta = np.array([0.6,-0.0,0.7,-0.4,0.4,-0.8])*1.3
g = -1 + np.dot(test_theta,phi_arr[1:,:])

keep = samples[g<0]

plt.figure()
plt.scatter(keep[:,0],keep[:,1])
plt.gca().set_aspect('equal')
plt.xlim(-2.5,2.5)
plt.ylim(-2.5,2.5)
plt.show()