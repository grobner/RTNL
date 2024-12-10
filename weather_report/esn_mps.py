import numpy as np
from mps_evolved import Evolution_MPS


class ESN_MPS:
    def __init__(self, rho, parameter_size, input_size, d_bond, std, device='cpu'):
        self.rho = rho
        self.parameter_size = parameter_size
        self.input_size = input_size
        self.d_bond = d_bond
        self.std = std
        self.output_layer = np.random.normal(size=(parameter_size, input_size))
        self.device = device

        self.evolution_mps = Evolution_MPS(rho=rho, parameter_size=parameter_size, input_size=input_size, d_bond=d_bond, std=std, device=device)

    def fit(self, x, y):
        x_out = np.zeros((x.shape[0], self.parameter_size))
        for i in range(x.shape[0]):
            x_out[i] = self.evolution_mps(x[i])

        H_pinv = np.linalg.pinv(x_out)
        self.output_layer = H_pinv @ y

        y_predict = x_out @ self.output_layer
        return y_predict

    def run(self, x):

        test_len = len(x)
        y_pred = np.zeros((x.shape[0], self.input_size))
        y = x[0]
        for n in range(test_len):
            y = self.evolution_mps(y) @ self.output_layer
            y_pred[n] = y

        return y_pred

    def predict(self, x):

        test_len = len(x)
        y_pred = np.zeros((x.shape[0], self.input_size))
        for n in range(test_len):
            y = self.evolution_mps(x[n]) @ self.output_layer
            y_pred[n] = y

        return y_pred
