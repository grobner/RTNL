import numpy as np
from mps_evolved import Evolution_MPS

class ELM_MPS:
    def __init__(self, rho, parameter_size, input_size, d_bond, std, device, input_scale=0.1):
        self.rho = rho
        self.parameter_size = parameter_size
        self.input_size = input_size
        self.d_bond = d_bond
        self.std = std

        self.evolution_mps = Evolution_MPS(rho=rho, parameter_size=parameter_size, input_size=input_size, d_bond=d_bond, std=std, device=device, input_scale=input_scale)
        self.x = np.zeros(self.parameter_size)

    def reservoir_output(self, x_t):

        x_allT = []

        for i in range(x_t.shape[0]):
            self.x = self.evolution_mps(self.x, x_t[i])
            x_allT.append(self.x)

        return self.x.reshape(self.parameter_size), np.array(x_allT).reshape(-1, self.parameter_size)

    def fit(self, x, y):

        H_T, H_allT = self.reservoir_output(x)

        H_pinv = np.linalg.pinv(H_allT)

        self.output_layer = H_pinv @ y

    def predict(self, x):
        H_T, H_allT = self.reservoir_output(x)
        out = H_allT @ self.output_layer

        return out
