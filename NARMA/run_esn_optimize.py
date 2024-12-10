import argparse
import tensornetwork as tn
from narma_dataset import NARMA
from NARMA.esn_model import ESN
from logging import getLogger, Formatter, StreamHandler
import logging

import numpy as np
import os

import pickle
import optuna


logger = getLogger(__name__)
logger.setLevel(logging.INFO)
format = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

ch = StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(format)
logger.addHandler(ch)

input_size = 1

parser = argparse.ArgumentParser(description='Optimize ESN parameters for NARMA.')
parser.add_argument('--order', type=int, default=5, help='Order of the NARMA task (default: 5)')
args = parser.parse_args()

order = args.order
T_train = 900
T_test = 100
y_init = [0.0] * order
N_x = 64


narma = NARMA(order, 0.3, 0.05, 1.5, 0.1)
u, y = narma.generate_data(T_train+T_test, y_init)

train_U = u[:T_train].reshape(-1, 1)
train_D = y[:T_train].reshape(-1, 1)
test_U = u[T_train:].reshape(-1, 1)
test_D = y[T_train:].reshape(-1, 1)

logger.info(f'{order}step ahead prediction')

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
        logger.info(f'nrmse_train : {nrmse_train}')
        logger.info(f'rmse_test : {rmse_test}')
        logger.info(f'nrmse_test : {nrmse_test}')

        rmses_test.append(rmse_test)

    avg_rmse_test = np.mean(rmses_test)
    logger.info(f'Average rmse test : {avg_rmse_test}')
    return avg_rmse_test

def save_result(params):
    spectral_radius = params['spectral_radius']
    density = params['density']

    os.makedirs(f'narma{order}_results', exist_ok=True)
    os.makedirs(f'narma{order}_results/esn_model', exist_ok=True)
    os.makedirs(f'narma{order}_results/esn_prediction_data', exist_ok=True)

    for seed in range(10):
        np.random.seed(seed)
        model = ESN(N_x, train_U.shape[1],
                    density=density, spectral_radius=spectral_radius, seed=seed)

        logger.info('fitting')
        model.fit(train_U, train_D)
        logger.info('predict')
        train_Y = model.predict(train_U)
        test_Y = model.predict(test_U)

        np.savez(f'narma{order}_results/esn_prediction_data/optimized_esn_{N_x}_{seed}.npz', train_U=train_U, train_D=train_D, test_U=test_U, test_D=test_D, train_Y=train_Y, test_Y=test_Y)

        with open(f'narma{order}_results/esn_model/optimized_esn_{N_x}_{seed}.pickle', 'wb') as file:
            pickle.dump(model, file)


if __name__ == '__main__':

    study = optuna.create_study(direction='minimize')  # We want to minimize the rmse of test
    study.optimize(objective, n_trials=100)

    print('Best parameters: ', study.best_params)
    os.makedirs(f'narma{order}_results/optuna', exist_ok=True)
    with open(f'narma{order}_results/optuna/best_params_esn_{N_x}.pickle', 'wb') as file:
        pickle.dump(study.best_params, file)

    save_result(study.best_params)
