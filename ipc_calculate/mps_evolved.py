import numpy as np
import torch
from mps_batch import MatrixProductState


class Evolution_MPS:
    def __init__(self, rho, input_size, d_bond, std, device = 'cuda:0'):
        self.rho = rho
        self.input_size = input_size
        self.mps = MatrixProductState(n_sites=input_size + 1,
                                      n_labels=input_size,
                                      d_phys=2,
                                      d_bond=d_bond,
                                      std=std,
                                      device=device)
        self.win = (2*np.random.rand(input_size)-1)
        self.iota = 0.1

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __call__(self, x, zeta):
        x_tmp = torch.Tensor(x).reshape(1, self.input_size, 1)
        x_tmp = torch.concat((x_tmp, 1 - x_tmp), dim = 2)

        return self._sigmoid(self.rho * self.mps(x_tmp).cpu().numpy() + self.iota * self.win * zeta)
