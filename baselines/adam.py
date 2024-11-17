import numpy as np

class AdamWithGrad(object):
    def __init__(self,gamma,beta_1,beta_2) -> None:
        self.gamma = gamma
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def reset(self,init_theta):
        self.m=0
        self.v=0
        self.curr_theta = init_theta

    def step(self,grad):
        assert grad.shape == self.curr_theta.shape
        self.m = self.beta_1 * self.m + (1-self.beta_1) * grad
        self.v = self.beta_2 * self.v + (1-self.beta_2) * (grad**2)

        m_hat = self.m/(1-self.beta_1)
        v_hat = self.v/(1-self.beta_2)

        self.curr_theta = self.curr_theta - self.gamma * m_hat / (np.sqrt(v_hat) + 1e-10)
        return self.curr_theta

    