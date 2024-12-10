import numpy as np
import torch
from mps_batch import MatrixProductState


class Evolution_MPS:
    def __init__(self, rho, parameter_size, input_size, d_bond, std, device = 'cpu'):
        self.rho = rho
        self.parameter_size = parameter_size
        self.input_size = input_size
        self.mps = MatrixProductState(n_sites=parameter_size + 1,
                                    n_labels=parameter_size,
                                    d_phys=2,
                                    d_bond=d_bond,
                                    std=std,
                                    device=device)
        self.win = (2*np.random.rand(parameter_size, input_size)-1)
        self.iota = 0.1
        self.x = np.zeros(self.parameter_size)

    def _sigmoid(self, x):
        return np.exp(np.minimum(x, 0)) / (1 + np.exp(- np.abs(x)))

    def __call__(self, zeta):
        x_tmp = torch.Tensor(self.x).reshape(1, self.parameter_size, 1)
        x_tmp = torch.concat((x_tmp, 1 - x_tmp), dim = 2)
        zeta = zeta.reshape(self.input_size)
        self.x = self._sigmoid(self.rho * self.mps(x_tmp).cpu().numpy() + self.iota * self.win @ zeta)
        return self.x
