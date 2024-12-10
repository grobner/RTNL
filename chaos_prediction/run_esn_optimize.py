#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import argparse
from generate_lorenz_attractor import LorenzSystem
from chaos_prediction.esn_model import ESN
import logging
from logging import getLogger, Formatter, StreamHandler
import os
import pickle
import optuna

# Set up logging
logger = getLogger(__name__)
logger.setLevel(logging.INFO)
format = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch = StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(format)
logger.addHandler(ch)

# Configurations
T_train = 100
T_test = 25
dt = 0.02
x0 = np.array([1, 1, 1])
index = 400

logger.info('Start Lorenz dynamics')
dynamics = LorenzSystem(10.0, 28.0, 8.0/3.0)
data = dynamics.integrate(x0, T_train + T_test, dt)

# Prepare train and test data
train_U = data[:int(T_train/dt)]
train_D = data[1:int(T_train/dt)+1]

test_U = data[int(T_train/dt):int((T_train+T_test)/dt)]
test_D = data[1+int(T_train/dt):int((T_train+T_test)/dt)+1]

N_x = 64

# Objective function for RMSE minimization
def objective(trial):
    spectral_radius = trial.suggest_float('spectral_radius', 0.8, 1.2)
    density = trial.suggest_float('density', 0.001, 0.1)
    rmses_test = []

    for seed in range(10):
        np.random.seed(seed)
        model = ESN(N_x, train_U.shape[1],
                    density=density, spectral_radius=spectral_radius, seed=seed)
        logger.info(f'Starting training for seed: {seed}')
        model.fit(train_U, train_D)
        logger.info('Predicting')
        test_Y = model.run(test_U)

        # Calculate RMSE for test data
        rmse_test = np.sqrt(np.mean((test_D[:index] - test_Y[:index]) ** 2))
        logger.info(f'RMSE for seed {seed}: {rmse_test}')
        rmses_test.append(rmse_test)

    # Average RMSE across seeds
    avg_rmse_test = np.mean(rmses_test)
    logger.info(f'Average RMSE test: {avg_rmse_test}')
    return avg_rmse_test

# Save results function
def save_result(params):
    spectral_radius = params['spectral_radius']
    density = params['density']

    for seed in range(10):
        np.random.seed(seed)
        model = ESN(N_x, train_U.shape[1],
                    density=density, spectral_radius=spectral_radius, seed=seed)
        logger.info(f'Starting training for seed: {seed}')
        model.fit(train_U, train_D)
        logger.info('Predicting')
        test_Y = model.run(test_U)

        # Save predictions and model
        os.makedirs('results/esn_model', exist_ok=True)
        os.makedirs('results/esn_prediction_data', exist_ok=True)

        np.savez(f'results/esn_prediction_data/optimized_esn_{N_x}_{seed}.npz',
                 train_U=train_U, train_D=train_D, test_U=test_U, test_D=test_D, test_Y=test_Y)

        with open(f'results/esn_model/optimized_esn_{N_x}_{seed}.pickle', 'wb') as file:
            pickle.dump(model, file)

# Main script
if __name__ == '__main__':
    # Set up Optuna study for RMSE minimization
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)

    # Save the best parameters
    print('Best parameters: ', study.best_params)
    os.makedirs('results/optuna', exist_ok=True)
    with open(f'results/optuna/best_params_{N_x}.pickle', 'wb') as file:
        pickle.dump(study.best_params, file)

    save_result(study.best_params)
