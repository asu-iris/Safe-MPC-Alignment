import numpy as np

class LowPassFilter_Vel(object):
    def __init__(self,k_v=0,k_w=0) -> None:
        self.y_past=np.zeros(6)
        self.k_v=k_v
        self.k_w=k_w

    def filter(self,x):
        y=np.zeros(6)
        y[0:3]=self.y_past[0:3]*self.k_v + x[0:3]*(1-self.k_v)
        y[3:6]=self.y_past[3:6]*self.k_w + x[3:6]*(1-self.k_w)
        self.y_past=np.array(y)
        return y
    
    def reset(self):
        self.y_past=np.zeros(6)