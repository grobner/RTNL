import argparse
import csv
import logging
import os
import pickle
from logging import Formatter, StreamHandler, getLogger

import numpy as np
import tensornetwork as tn
import torch
from elm_mps import ELM_MPS
from getdata import read_sunspot_data

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
parser.add_argument('--step', type=int, default=1)

args = parser.parse_args()
d_bond = args.d_bond
std = args.std
rho = args.rho
parameter_size = args.parameter_size
seed= args.seed
step = args.step

input_size = 1

# 黒点数データ
sunspots = read_sunspot_data(filename='SN_ms_tot_V2.0.txt')

# データのスケーリング
data_scale = 1.0e-3
data = sunspots*data_scale


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

# 訓練・検証データ長
T_train = 2500
T_test = data.size-T_train-step

# 訓練・検証用情報
train_U = data[:T_train].reshape(-1, 1)
train_Y = data[step:T_train+step].reshape(-1, 1)

test_U = data[T_train:T_train+T_test].reshape(-1, 1)
test_Y = data[T_train+step:T_train+T_test+step].reshape(-1, 1)

logger.info(f'{step}step ahead prediction')
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

os.makedirs(f'sunspot_{step}step_results_N{parameter_size}', exist_ok=True)
os.makedirs(f'sunspot_{step}step_results_N{parameter_size}/mps_model', exist_ok=True)
csv_path = f'sunspot_{step}step_results_N{parameter_size}/result.csv'

if not os.path.exists(csv_path):
    with open(csv_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['d_bond', 'std', 'rho', 'seed', 'parameter_size', 'train_rmse', 'test_rmse', 'train_nrmse', 'test_nrmse'])


with open(f'sunspot_{step}step_results_N{parameter_size}/result.csv', 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([d_bond, std, rho, seed, parameter_size, rmse_train, rmse_test, nrmse_train, nrmse_test])

with open(f'sunspot_{step}step_results_N{parameter_size}/mps_model/model_{d_bond}_{std}_{rho}_{parameter_size}_{seed}.pickle', 'wb') as file:
    pickle.dump(elm_mps, file)


train_Y_predict = np.concatenate([train_Y, y_predict_train], axis=1)
test_Y_predict = np.concatenate([test_Y, y_predict_test], axis=1)
train_test_Y_predict = np.concatenate([train_Y_predict, test_Y_predict], axis=0)

os.makedirs(f'sunspot_{step}step_results_N{parameter_size}/train_test_Y_predict', exist_ok=True)
np.save(f'sunspot_{step}step_results_N{parameter_size}/train_test_Y_predict/{d_bond}_{std}_{rho}_{parameter_size}_{seed}.npy', train_test_Y_predict)


