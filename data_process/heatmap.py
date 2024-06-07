import numpy as np
import matplotlib.pyplot as plt
import sys,os
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
sys.path.append(os.path.abspath(os.getcwd()))

import casadi as cd
from utils.RBF import gen_eval_func_uav,gen_eval_func_arm

rbf_X_c = np.array([9.8])
rbf_Y_c = np.linspace(0, 10, 10)  # 5
rbf_Z_c = np.linspace(0, 10, 10)  # 5

weights_init=np.ones(20)
weights_learned=np.array([ 10.88026447,  27.00317417,  46.44679587,  14.87593348,
       -42.67265184, -37.33379115,  11.29852723,  33.03116974,
        16.42123528,   9.77181548,  10.9587004 ,  12.61127868,
         4.95083013, -15.23866117,  -7.70462811,  23.80022951,
        32.87481303,  20.86143877,  13.2182444 ,  10.83844088])


#print(g_func(np.array([[10,1,1],[10,2,2]]).T))

def heatmap_weight_uav(weights):
    savepath=os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),'Data','uav_figs')
    g_func=gen_eval_func_uav(weights=weights,X_c=rbf_X_c,Y_c=rbf_Y_c,Z_c=rbf_Z_c,epsilon=0.45,bias=-1)
    y = np.linspace(0, 10, 100)
    z = np.linspace(0, 10, 100)
    Y,Z = np.meshgrid(y, z)
    X=10*np.ones((10000,1))
    Y_v=Y.reshape(-1,1)
    Z_v=Z.reshape(-1,1)
    P= np.concatenate((X,Y_v,Z_v),axis=1)
    G=g_func(P.T).full().reshape(100,100)
    print(G.shape)

    gate_points=np.array([[3.5,4],
                     [6.5,4],
                     [6.5,5.4],
                     [3.5,5.4],
                     [3.5,4]])
    
    plt.figure(figsize=(8, 6))
    plt.plot(*gate_points.T,color='black',linewidth=3.0)
    contour = plt.contour(Y, Z, G, levels=[0.0], colors='black', linewidths=2, linestyles='dashed')
    plt.pcolormesh( Y, Z, G, cmap='RdBu')
    cbar=plt.colorbar(label='Function Value')
    cbar.ax.tick_params(labelsize=20) 
    cbar.set_label('g Value',size=20)
    plt.title('Heatmap of Learned constraint: UAV', fontsize=20)
    plt.xlabel('y', fontsize=20)
    plt.ylabel('z', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(savepath,'heatmap.png'))

    plt.show()

def heatmap_weight_arm(weights):
    savepath=os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),'Data','arm_figs')
    g_func=gen_eval_func_arm(weights=weights,z_min=0.2,z_max=0.9, num_z=10, bias=-0.8, epsilon_z=12, epsilon_q=1.8)
    theta = np.linspace(-np.pi/4, 3*np.pi/4, 100)
    z = np.linspace(0.2, 0.9, 100)
    T,Z = np.meshgrid(theta, z)
    T_v=T.reshape(-1,1)
    Z_v=Z.reshape(-1,1)
    P= np.concatenate((T_v,Z_v),axis=1)
    G=g_func(P.T).full().reshape(100,100)
    print(G.shape)

    plt.figure(figsize=(8, 6))
    contour = plt.contour(T, Z, G, levels=[0.0], colors='black', linewidths=2, linestyles='dashed')
    plt.pcolormesh( T, Z, G, cmap='RdBu')
    cbar=plt.colorbar(label='Function Value')
    cbar.ax.tick_params(labelsize=20) 
    cbar.set_label('g Value',size=20)
    plt.title('Heatmap of Learned constraint: ARM', fontsize=20)
    plt.xlabel('theta', fontsize=20)
    plt.ylabel('z', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()

    plt.show()


arm_weights=np.array([ 0.8605393 ,  0.8317794 ,  1.11404594,  0.48366286, -0.83400422,
       -1.20513047, -0.080032  , -0.34014744,  0.37021907,  0.97919927,
        1.03243613,  1.04706619, -0.06139236, -1.80106347,  1.84331149,
        2.98620837,  2.9307057 ,  1.16053416,  1.        ,  1.        ])
print(arm_weights)
heatmap_weight_arm(arm_weights)