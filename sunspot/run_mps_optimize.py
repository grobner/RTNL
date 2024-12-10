from getdata import read_sunspot_data
from logging import getLogger, Formatter, StreamHandler, FileHandler
import logging
from elm_mps import ELM_MPS
import optuna
import torch
import tensornetwork as tn
import numpy as np
import os

import pickle

tn.set_default_backend('pytorch')
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True

logger = getLogger(__name__)
logger.setLevel(logging.INFO)
format = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

ch = StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(format)
logger.addHandler(ch)


step = 1
input_size = 1
rho = 1
N_x = 64
folder = f'sunspot_{step}step_results_N{N_x}'

optuna.logging.get_logger("optuna").addHandler(logging.FileHandler(f'{folder}/optuna.log'))

sunspots = read_sunspot_data(file_name='SN_ms_tot_V2.0.txt')

data_scale = 1.0e-3
data = sunspots*data_scale

T_train = 2500
T_test = data.size-T_train-step

train_U = data[:T_train].reshape(-1, 1)
train_D = data[step:T_train+step].reshape(-1, 1)

test_U = data[T_train:T_train+T_test].reshape(-1, 1)
test_D = data[T_train+step:T_train+T_test+step].reshape(-1, 1)

logger.info(f'{step}step ahead prediction')

save_array = np.load('../order_parameter/results/orderline_64.npy')

std_points = save_array[0][:, 0]
dbond_points = save_array[0][:, 1]

def piecewise_linear(std):
    # Limit if x has a value outside the range
    if std <= std_points[0]:
        return dbond_points[0]
    elif std >= std_points[-1]:
        return dbond_points[-1]
    else:
        # Linear interpolation of the corresponding segment based on the value of x
        i = np.searchsorted(std_points, std) - 1  # Index of the corresponding interval
        std0, std1 = std_points[i], std_points[i + 1]
        dbond0, dbond1 = dbond_points[i], dbond_points[i + 1]
        # linear interpolation
        return dbond0 + (dbond1 - dbond0) * (std - std0) / (std1 - std0)


if torch.cuda.is_available():
    device = torch.device('cuda:0')
    logger.info('use GPU')
else:
    device = torch.device('cpu')
    logger.info('use CPU')

def objective(trial):
    std = trial.suggest_float('std', 0.001, 0.1)
    d_bond = trial.suggest_int('d_bond', 1, piecewise_linear(std) + 5)
    rmses_test = []

    for seed in range(10):
        np.random.seed(seed)

        model = ELM_MPS(rho=rho, parameter_size=N_x, input_size=train_U.shape[1], d_bond=d_bond, std=std, device=device)

        logger.info('fitting')
        model.fit(train_U, train_D)
        logger.info('predict')
        train_Y = model.predict(train_U)
        test_Y = model.predict(test_U)

        rmse_test = np.sqrt(np.mean((test_D - test_Y) ** 2))

        logger.info(f'rmse_test : {rmse_test}')

        rmses_test.append(rmse_test)

    avg_rmse_test = np.mean(rmses_test)
    logger.info(f'Average rmse test : {avg_rmse_test}')
    return avg_rmse_test

def save_result(params):
    d_bond = params['d_bond']
    std = params['std']

    for seed in range(10):
        np.random.seed(seed)
        model = ELM_MPS(rho=rho, parameter_size=N_x, input_size=train_U.shape[1], d_bond=d_bond, std=std, device=device)

        logger.info('fitting')
        model.fit(train_U, train_D)
        logger.info('predict')
        train_Y = model.predict(train_U)
        test_Y = model.predict(test_U)

        os.makedirs(f'{folder}', exist_ok=True)
        os.makedirs(f'{folder}/esn_model', exist_ok=True)
        os.makedirs(f'{folder}/esn_prediction_data', exist_ok=True)

        np.savez(f'{folder}/esn_prediction_data/optimized_mps_{N_x}_{seed}_2.npz', train_U=train_U, train_D=train_D, test_U=test_U, test_D=test_D, train_Y=train_Y, test_Y=test_Y)

        with open(f'{folder}/esn_model/optimized_mps_{N_x}_{seed}_2.pickle', 'wb') as file:
            pickle.dump(model, file)


if __name__ == '__main__':

    study = optuna.create_study(direction='minimize')  # We want to minimize the rmse of test
    study.optimize(objective, n_trials=100)

    print('Best parameters: ', study.best_params)
    os.makedirs(f'{folder}/optuna', exist_ok=True)
    with open(f'{folder}/optuna/best_params_mps_{N_x}_2.pickle', 'wb') as file:
        pickle.dump(study.best_params, file)

    save_result(study.best_params)
