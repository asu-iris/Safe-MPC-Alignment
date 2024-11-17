import numpy as np
import cvxpy as cp
from matplotlib import pyplot as plt

import os
# MVE calculation based on cvxpy
class mvesolver(object):
    def __init__(self,name,dim,path=os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),'Data','mve_checkpoint')) -> None:
        self.name=name
        self.dim=dim
        self.constraint_a=[]
        self.constraint_b=[]

        self.recovery_path=path

    def set_init_constraint(self,theta_lb,theta_ub): #init box constraint
        self.constraint_a=[]
        self.constraint_b=[]
        self.lb=np.array(theta_lb)
        self.ub=np.array(theta_ub)
        I=np.eye(self.dim)
        for i in range(self.dim):
            self.constraint_a.append(I[:,i:i+1])
            self.constraint_b.append(self.ub[i])
            self.constraint_a.append(-I[:,i:i+1])
            self.constraint_b.append(-self.lb[i])

    def add_constraint(self,a,b):#ax <= b
        self.constraint_a.append(np.reshape(np.array(a),(-1,1)))
        self.constraint_b.append(float(b))

    def solve(self,tolerance=0.001): #solve the MVE and its center
        C = cp.Variable((self.dim, self.dim), symmetric=True)
        d = cp.Variable(self.dim)
        # create the constraints
        constraints = [C >> tolerance]
        for a,b in zip(self.constraint_a,self.constraint_b):
            constraints.append(cp.norm2(C @ a) + a.T @ d <= b)
        prob = cp.Problem(cp.Minimize(-cp.log_det(C)), constraints)
        #prob.solve(solver=cp.MOSEK, verbose=False)
        prob.solve(verbose=False)
        #print(prob.status)
        if prob.status=='infeasible':
            raise RuntimeError('MVEsolver: infeasible region')
        if prob.status=='unbounded':
            raise RuntimeError('MVEsolver: unbounded region')
        return d.value, C.value
    
    def draw(self, C=None, d=None, ref=None): #plot 2D edges and MVE
        theta = np.arange(-np.pi-0.1,np.pi+0.1,0.1)
        range_x = np.arange(self.lb[0]-1,self.ub[0]+1,0.1)
        range_y = np.arange(self.lb[1]-1,self.ub[1]+1,0.1)
        if C is not None and d is not None:
            circle = np.vstack((np.cos(theta), np.sin(theta)))
            #print(np.matmul(C,circle))
            ellipsis = np.matmul(C,circle)+d.reshape(-1,1)
            #print(ellipsis.shape)
            plt.plot(ellipsis[0], ellipsis[1])

        # obtain the line data
        lines = []
        for i in range(len(self.constraint_a)):
            n = self.constraint_a[i].flatten()
            b = self.constraint_b[i]
            if abs(n[1]) < 1e-10:
                y = range_y
                x = np.array([b/n[0]]*range_y.size)
            else:
                x = range_x
                y = (b-n[0]*x)/n[1]
            lines.append( (x,y) )
        for line in lines:
            plt.plot(line[0],line[1])
        plt.scatter(d[0],d[1])
        if ref is not None:
            plt.scatter(ref[0],ref[1])
        plt.axis([self.lb[0]-1, self.ub[0]+1,self.lb[1]-1, self.ub[1]+1])
        plt.show()

    def savefig(self, C=None, d=None, ref=None, dir:str='./'):
        theta = np.arange(-np.pi-0.1,np.pi+0.1,0.1)
        range_x = np.arange(self.lb[0]-1,self.ub[0]+1,0.1)
        range_y = np.arange(self.lb[1]-1,self.ub[1]+1,0.1)
        if C is not None and d is not None:
            circle = np.vstack((np.cos(theta), np.sin(theta)))
            #print(np.matmul(C,circle))
            ellipsis = np.matmul(C,circle)+d.reshape(-1,1)
            #print(ellipsis.shape)
            plt.plot(ellipsis[0], ellipsis[1])

        # obtain the line data
        lines = []
        for i in range(len(self.constraint_a)):
            n = self.constraint_a[i].flatten()
            b = self.constraint_b[i].flatten()
            if abs(n[1]) < 1e-10:
                y = range_y
                x = np.array([b/n[0]]*range_y.size)
            else:
                x = range_x
                y = (b-n[0]*x)/n[1]
            lines.append( (x,y) )
        for line in lines:
            plt.plot(line[0],line[1])
        plt.scatter(d[0],d[1])
        if ref is not None:
            plt.scatter(ref[0],ref[1])
        plt.axis([self.lb[0]-1, self.ub[0]+1,self.lb[1]-1, self.ub[1]+1])
        plt.title('mve search')
        plt.savefig(dir)

    def checkpoint(self):
        # create the path
        if not os.path.exists(self.recovery_path):
            os.makedirs(self.recovery_path)

        if len(self.constraint_a) <= 2*self.dim:
            return
        #print(self.constraint_a)
        #print(self.constraint_b)
        A = np.concatenate(self.constraint_a,axis=1)
        #print(A)
        B = np.array(self.constraint_b).reshape(-1,1)

        data=np.concatenate((A.T,B),axis=1)
        np.save(os.path.join(self.recovery_path,'checkpoint.npy'),data)

    def recover(self):
        self.constraint_a=[]
        self.constraint_b=[]
        data=np.load(os.path.join(self.recovery_path,'checkpoint.npy'))
        A_T=data[:,:-1]
        B=data[:,-1]
        for i in range(A_T.shape[0]):
            self.constraint_a.append(A_T[i].reshape(-1,1))
        self.constraint_b=list(B)
        d,C=self.solve()
        return d
    
    def get_constraints(self):
        A = np.concatenate(self.constraint_a,axis=1)
        #print(A)
        B = np.array(self.constraint_b).reshape(-1,1)
        return A.T, B

if __name__=='__main__':
    testsol=mvesolver('test',2)
    testsol.set_init_constraint([-5,-5],[2,2])
    testsol.add_constraint(np.array([1,1]),0)
    testsol.add_constraint(np.array([0,1]),-1.5)
    testsol.add_constraint(np.array([2,-1]),0)
    #testsol.add_constraint(np.array([-2,1]),0)
    d_sol,C_sol= testsol.solve()
    print(d_sol)
    testsol.draw(C_sol, d_sol)

    testsol.checkpoint()
    testsol_1=mvesolver('test_1',2)
    testsol_1.set_init_constraint([-5,-5],[2,2])
    d_rec=testsol_1.recover()
    d_sol,C_sol= testsol_1.solve()
    testsol_1.draw(C_sol, d_sol)
    print(d_rec)
