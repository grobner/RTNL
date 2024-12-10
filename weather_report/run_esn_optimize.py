#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from weather_report.esn_model import ESN
import logging
from logging import getLogger, Formatter, StreamHandler
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

index = 400
N_x = 64

data = np.load('train_data.npz')

train_U = data['X_train']
train_D = data['y_train']

test_U = data['X_test']
test_D = data['y_test']

def objective(trial):
    spectral_radius = trial.suggest_float('spectral_radius', 0.8, 1.2)
    density = trial.suggest_float('density', 0.001, 0.1)
    rmses_test = []

    for seed in range(10):
        np.random.seed(seed)
        logger.info(f'seed : {seed}')
        model = ESN(N_x, train_U.shape[1],
                    density=density, spectral_radius=spectral_radius, seed=seed)
        logger.info('training')
        model.fit(train_U, train_D)
        logger.info('predict')
        test_Y = model.predict(test_U)

        rmse_test = np.sqrt(np.mean((test_D[:index] - test_Y[:index]) ** 2))

        logger.info(f'rmse_test : {rmse_test}')

        rmses_test.append(rmse_test)

    avg_rmse_test = np.mean(rmses_test)
    logger.info(f'Average rmse test : {avg_rmse_test}')
    return avg_rmse_test

def save_result(params):
    spectral_radius = params['spectral_radius']
    density = params['density']
    os.makedirs(f'results', exist_ok=True)
    os.makedirs(f'results/esn_model', exist_ok=True)
    os.makedirs(f'results/esn_prediction_data', exist_ok=True)

    for seed in range(10):
        np.random.seed(seed)
        logger.info(f'seed : {seed}')
        model = ESN(N_x, train_U.shape[1],
                    density=density, spectral_radius=spectral_radius, seed=seed)
        logger.info('training')
        model.fit(train_U, train_D)
        logger.info('predict')
        test_Y = model.predict(test_U)

        rmse_test = np.sqrt(np.mean((test_D - test_Y) ** 2))

        logger.info(f'rmse_test : {rmse_test}')

        np.savez(f'results/esn_prediction_data/optimized_esn_{N_x}_{seed}.npz', train_U=train_U, train_D=train_D, test_U=test_U, test_D=test_D, test_Y=test_Y)

        with open(f'results/esn_model/optimized_esn_{N_x}_{seed}.pickle', 'wb') as file:
            pickle.dump(model, file)

if __name__ == '__main__':

    study = optuna.create_study(direction='minimize')  # We want to minimize the rmse of test
    study.optimize(objective, n_trials=100)

    print('Best parameters: ', study.best_params)
    os.makedirs('results/optuna', exist_ok=True)
    with open(f'results/optuna/best_params_{N_x}.pickle', 'wb') as file:
        pickle.dump(study.best_params, file)
    # with open(f'results/optuna/best_params_{N_x}.pickle', 'rb') as file:
    #     params = pickle.load(file)
    # save_result(params)

    save_result(study.best_params)
