import numpy as np
import casadi as cd

def rbf(x,y,x_c:float,y_c:float,sigma:float):
    return cd.exp(-sigma * ((x-x_c)**2 + (y-y_c)**2))