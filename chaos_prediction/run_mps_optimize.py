import numpy as np
import argparse
from generate_lorenz_attractor import LorenzSystem
from esn_mps import ESN_MPS
import logging
from logging import getLogger, Formatter, StreamHandler
import os
import pickle
import optuna
import torch
import tensornetwork as tn

# Logger setup
logger = getLogger(__name__)
logger.setLevel(logging.INFO)
format = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch = StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(format)
logger.addHandler(ch)

# Lorenz system setup
T_train = 100
T_test = 25
dt = 0.02
x0 = np.array([1, 1, 1])

logger.info('Start Lorenz dynamics')
dynamics = LorenzSystem(10.0, 28.0, 8.0/3.0)
data = dynamics.integrate(x0, T_train + T_test, dt)

# Prepare train and test data
train_U = data[:int(T_train/dt)]
train_D = data[1:int(T_train/dt)+1]

test_U = data[int(T_train/dt):int((T_train+T_test)/dt)]
test_D = data[1+int(T_train/dt):int((T_train+T_test)/dt)+1]

# Model parameters
N_x = 64
rho = 1
index = 400

# Set backend and seed
tn.set_default_backend('pytorch')
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True

# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logger.info(f'Using device: {device}')

# Load saved array for piecewise linear function
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

# Objective function for RMSE minimization
def objective(trial):
    std = trial.suggest_float('std', 0.001, 0.1)
    d_bond = trial.suggest_int('d_bond', 1, piecewise_linear(std) + 5)
    rmses_test = []

    for seed in range(10):
        np.random.seed(seed)

        model = ESN_MPS(rho=rho, parameter_size=N_x, input_size=train_U.shape[1], d_bond=d_bond, std=std, device=device)

        logger.info(f'Training with seed: {seed}')
        model.fit(train_U, train_D)

        logger.info('Predicting')
        test_Y = model.predict(test_U)

        # Calculate RMSE for test data
        rmse_test = np.sqrt(np.mean((test_D[:index] - test_Y[:index]) ** 2))
        logger.info(f'RMSE for seed {seed}: {rmse_test}')

        rmses_test.append(rmse_test)

    avg_rmse_test = np.mean(rmses_test)
    logger.info(f'Average RMSE test: {avg_rmse_test}')
    return avg_rmse_test

# Save model and predictions
def save_result(params):
    d_bond = params['d_bond']
    std = params['std']

    os.makedirs('results/esn_model', exist_ok=True)
    os.makedirs('results/esn_prediction_data', exist_ok=True)

    for seed in range(10):
        np.random.seed(seed)
        model = ESN_MPS(rho=rho, parameter_size=N_x, input_size=train_U.shape[1], d_bond=d_bond, std=std, device=device)

        logger.info(f'Training with seed: {seed}')
        model.fit(train_U, train_D)

        logger.info('Predicting')
        test_Y = model.predict(test_U)

        np.savez(f'results/esn_prediction_data/optimized_mps_{N_x}_{seed}.npz',
                 train_U=train_U, train_D=train_D, test_U=test_U, test_D=test_D, test_Y=test_Y)

        with open(f'results/esn_model/optimized_mps_{N_x}_{seed}.pickle', 'wb') as file:
            pickle.dump(model, file)

# Main script
if __name__ == '__main__':
    # Set up Optuna study for RMSE minimization
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)

    # Save the best parameters
    print('Best parameters: ', study.best_params)
    os.makedirs('results/optuna', exist_ok=True)
    with open(f'results/optuna/best_params_mps_{N_x}.pickle', 'wb') as file:
        pickle.dump(study.best_params, file)

    save_result(study.best_params)
