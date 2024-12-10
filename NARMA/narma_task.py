import argparse
import tensornetwork as tn
from narma_dataset import NARMA
from elm_mps import ELM_MPS
from logging import getLogger, Formatter, StreamHandler
import logging

import numpy as np
import torch
import os
import csv

import pickle

logger = getLogger(__name__)
logger.setLevel(logging.INFO)
format = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

ch = StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(format)
logger.addHandler(ch)

tn.set_default_backend('pytorch')

parser = argparse.ArgumentParser()

parser.add_argument('--d_bond', type=int)
parser.add_argument('--std', type=float)
parser.add_argument('--rho', type=float)
parser.add_argument('--parameter_size', type=int, default=64)
parser.add_argument('--seed', type=int, default=0)

args = parser.parse_args()
d_bond = args.d_bond
std = args.std
rho = args.rho
parameter_size = args.parameter_size
seed= args.seed

input_size = 1

order = 3
T_train = 900
T_test = 100
y_init = [0.0] * order

narma = NARMA(order, 0.3, 0.05, 1.5, 0.1)
u, y = narma.generate_data(T_train+T_test, y_init)

train_U = u[:T_train].reshape(-1, 1)
train_Y = y[:T_train].reshape(-1, 1)
test_U = u[T_train:].reshape(-1, 1)
test_Y = y[T_train:].reshape(-1, 1)

np.random.seed(seed)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    logger.info('use GPU')
else:
    device = torch.device('cpu')
    logger.info('use CPU')

logger.info(f'd_bond : {d_bond}, std : {std}, rho : {rho}, parameter_size : {parameter_size}, seed : {seed}')

elm_mps = ELM_MPS(rho=rho, parameter_size=parameter_size, input_size=input_size, d_bond=d_bond, std=std, device=device)


logger.info('fitting')
elm_mps.fit(train_U, train_Y)
logger.info('predict')
y_predict_train = elm_mps.predict(train_U)
y_predict_test = elm_mps.predict(test_U)

rmse_train = np.sqrt(np.mean((train_Y - y_predict_train) ** 2))
rmse_test = np.sqrt(np.mean((test_Y - y_predict_test) ** 2))

nrmse_train = rmse_train / np.std(train_Y)
nrmse_test = rmse_test / np.std(test_Y)

logger.info(f'rmse_train : {rmse_train}')
logger.info(f'rmse_train : {nrmse_train}')
logger.info(f'rmse_test : {rmse_test}')
logger.info(f'rmse_test : {nrmse_test}')

os.makedirs(f'narma{order}_results', exist_ok=True)
os.makedirs(f'narma{order}_results/mps_model', exist_ok=True)
csv_path = f'narma{order}_results/result.csv'

if not os.path.exists(csv_path):
    with open(csv_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['d_bond', 'std', 'rho', 'seed', 'parameter_size', 'train_rmse', 'test_rmse'])


with open(f'narma{order}_results/result.csv', 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([d_bond, std, rho, seed, parameter_size, rmse_train, rmse_test])

with open(f'narma{order}_results/mps_model/model_{d_bond}_{std}_{rho}_{parameter_size}_{seed}.pickle', 'wb') as file:
	pickle.dump(elm_mps, file)


train_Y_predict = np.concatenate([train_Y, y_predict_train], axis=1)
test_Y_predict = np.concatenate([test_Y, y_predict_test], axis=1)
train_test_Y_predict = np.concatenate([train_Y_predict, test_Y_predict], axis=0)

os.makedirs(f'narma{order}_results/train_test_Y_predict', exist_ok=True)
np.save(f'narma{order}_results/train_test_Y_predict/{d_bond}_{std}_{rho}_{parameter_size}_{seed}.npy', train_test_Y_predict)


