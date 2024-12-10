import argparse
from sunspot.esn_model import ESN
from getdata import read_sunspot_data
from logging import getLogger, Formatter, StreamHandler
import logging
import optuna

import numpy as np
import os

import pickle

logger = getLogger(__name__)
logger.setLevel(logging.INFO)
format = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

ch = StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(format)
logger.addHandler(ch)

step = 1
input_size = 1
N_x = 64
folder = f'sunspot_{step}step_results_N{N_x}'

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

def objective(trial):
    spectral_radius = trial.suggest_float('spectral_radius', 0.8, 1.2)
    density = trial.suggest_float('density', 0.001, 0.1)
    rmses_test = []

    for seed in range(10):
        np.random.seed(seed)

        model = ESN(N_x, train_U.shape[1],
                    density=density, spectral_radius=spectral_radius, seed=seed)

        logger.info('fitting')
        model.fit(train_U, train_D)
        logger.info('predict')
        train_Y = model.predict(train_U)
        test_Y = model.predict(test_U)

        rmse_train = np.sqrt(np.mean((train_D - train_Y) ** 2))
        rmse_test = np.sqrt(np.mean((test_D - test_Y) ** 2))

        nrmse_train = rmse_train / np.std(train_D)
        nrmse_test = rmse_test / np.std(test_D)

        logger.info(f'rmse_train : {rmse_train}')
        logger.info(f'rmse_train : {nrmse_train}')
        logger.info(f'rmse_test : {rmse_test}')
        logger.info(f'rmse_test : {nrmse_test}')

        rmses_test.append(rmse_test)

    avg_rmse_test = np.mean(rmses_test)
    logger.info(f'Average rmse test : {avg_rmse_test}')
    return avg_rmse_test

def save_result(params):
    spectral_radius = params['spectral_radius']
    density = params['density']

    for seed in range(10):
        np.random.seed(seed)
        model = ESN(N_x, train_U.shape[1],
                    density=density, spectral_radius=spectral_radius, seed=seed)

        logger.info('fitting')
        model.fit(train_U, train_D)
        logger.info('predict')
        train_Y = model.predict(train_U)
        test_Y = model.predict(test_U)

        os.makedirs(f'{folder}', exist_ok=True)
        os.makedirs(f'{folder}/esn_model', exist_ok=True)
        os.makedirs(f'{folder}/esn_prediction_data', exist_ok=True)

        np.savez(f'{folder}/esn_prediction_data/optimized_esn_{N_x}_{seed}.npz', train_U=train_U, train_D=train_D, test_U=test_U, test_D=test_D, train_Y=train_Y, test_Y=test_Y)

        with open(f'{folder}/esn_model/optimized_esn_{N_x}_{seed}.pickle', 'wb') as file:
            pickle.dump(model, file)


if __name__ == '__main__':

    study = optuna.create_study(direction='minimize')  # We want to minimize the rmse of test
    study.optimize(objective, n_trials=100)

    print('Best parameters: ', study.best_params)
    os.makedirs(f'{folder}/optuna', exist_ok=True)
    with open(f'{folder}/optuna/best_params_{N_x}.pickle', 'wb') as file:
        pickle.dump(study.best_params, file)

    save_result(study.best_params)
