import numpy as np
from sklearn.linear_model import LinearRegression

class LRConstraintLearner(object):
    def __init__(self) -> None:
        self.X_data_list=[]
        self.Y_data_list=[]

    def add_data(self,x,y):
        self.X_data_list.append(x)
        self.Y_data_list.append(y)

    def fit_weight(self):
        X_data=np.array(self.X_data_list)
        Y_data=np.array(self.Y_data_list)
        reg = LinearRegression(fit_intercept=False)
        reg.fit(X_data,Y_data)

        return reg.coef_.copy()
    