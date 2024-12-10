import torch
from mps_batch import MatrixProductState
from torch import nn
from logging import getLogger, INFO, Formatter, StreamHandler
import numpy as np

logger = getLogger(__name__)
logger.setLevel(INFO)
format = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

ch = StreamHandler()
ch.setLevel(INFO)
ch.setFormatter(format)
logger.addHandler(ch)

class ELM_MPS:
    def __init__(self, input_size, h_size, num_classes, d_phys, d_bond, std, device):
        self.input_size = input_size
        self.h_size = h_size
        self.outpout_size = num_classes
        self.device = device

        self.mps_layer = MatrixProductState(n_sites=input_size + 1, n_labels=self.h_size, d_phys=d_phys, d_bond=d_bond, std=std, device=device)

        self.output_layer = nn.init.constant_(torch.empty(self.h_size, self.outpout_size), 0)

        self.activation = torch.sigmoid

    def fit(self, x, y):
        logger.info('training dataset')
        temp = self.mps_layer(x)
        y = y.to(self.device)
        H = self.activation(temp)

        H_pinv = torch.pinverse(H)
        self.output_layer = H_pinv.mm(y)

    def predict(self, x):
        logger.info('predict')
        temp = self.mps_layer(x)
        H = self.activation(temp)
        out = H.mm(self.output_layer)

        return out
