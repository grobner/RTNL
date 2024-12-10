#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import argparse
from generate_lorenz_attractor import LorenzSystem
from esn_mps import ESN_MPS
import tensornetwork as tn
import logging
from logging import getLogger, Formatter, StreamHandler
import torch
import os
import pickle

tn.set_default_backend('pytorch')
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True

if __name__ == '__main__':

    logger = getLogger(__name__)
    logger.setLevel(logging.INFO)
    format = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    ch = StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(format)
    logger.addHandler(ch)

    parser = argparse.ArgumentParser()

    parser.add_argument('--d_bond', type=int)
    parser.add_argument('--std', type=float)
    parser.add_argument('--rho', type=float)
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    d_bond = args.d_bond
    std = args.std
    rho = args.rho
    seed = args.seed

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        logger.info('use GPU')
    else:
        device = torch.device('cpu')
        logger.info('use CPU')

    logger.info(f'd_bond : {d_bond}, std : {std}, seed : {seed}')

    T_train = 100
    T_test = 25
    dt = 0.02
    x0 = np.array([1, 1, 1])
    N = 64

    logger.info('start Lorenz dynamics')
    dynamics = LorenzSystem(10.0, 28.0, 8.0/3.0)
    data = dynamics.integrate(x0, T_train + T_test, dt)

    train_U = data[:int(T_train/dt)]
    train_D = data[1:int(T_train/dt)+1]

    test_U = data[int(T_train/dt):int((T_train+T_test)/dt)]
    test_D = data[1+int(T_train/dt):int((T_train+T_test)/dt)+1]

    np.random.seed(seed)

    model = ESN_MPS(rho=rho, parameter_size=N, input_size=train_U.shape[1], d_bond=d_bond, std=std, device=device)
    logger.info('training')
    train_Y = model.fit(train_U, train_D)
    logger.info('predict')
    test_Y = model.predict(test_U)

    valid_time = 0
    eps = 1
    for n in range(int(T_test/dt)):
        dif = np.sqrt(((test_D[n,:] - test_Y[n,:]) ** 2).mean())
        if dif > eps:
            valid_time = float(n)*dt
            break

    print('valid time = ', valid_time)

    os.makedirs('results', exist_ok=True)
    os.makedirs('results/training_model', exist_ok=True)
    os.makedirs('results/prediction_data', exist_ok=True)

    np.savez(f'results/prediction_data/{d_bond}_{std}_{rho}_{seed}.npz', train_U=train_U, train_D=train_D, test_U=test_U, test_D=test_D, train_Y=train_Y, test_Y=test_Y)

    with open(f'results/training_model/{d_bond}_{std}_{rho}_{seed}.pickle', 'wb') as file:
        pickle.dump(model, file)
