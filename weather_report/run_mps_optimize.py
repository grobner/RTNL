from logging import getLogger, Formatter, StreamHandler
import logging
from esn_mps import ESN_MPS
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

optuna.logging.get_logger("optuna").addHandler(logging.FileHandler('results/optuna/optuna.log'))

rho = 1
N_x = 64
data = np.load('train_data.npz')

train_U = data['X_train']
train_D = data['y_train']

test_U = data['X_test']
test_D = data['y_test']

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
    std = trial.suggest_float('std', 0.01, 0.1)
    d_bond = trial.suggest_int('d_bond', 1, piecewise_linear(std) + 5)
    rmses_test = []

    for seed in range(10):
        np.random.seed(seed)

        model = ESN_MPS(rho=rho, parameter_size=N_x, input_size=train_U.shape[1], d_bond=d_bond, std=std, device=device)

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
        model = ESN_MPS(rho=rho, parameter_size=N_x, input_size=train_U.shape[1], d_bond=d_bond, std=std, device=device)

        logger.info('fitting')
        model.fit(train_U, train_D)
        logger.info('predict')
        train_Y = model.predict(train_U)
        test_Y = model.predict(test_U)

        os.makedirs(f'results', exist_ok=True)
        os.makedirs(f'results/esn_model', exist_ok=True)
        os.makedirs(f'results/esn_prediction_data', exist_ok=True)

        np.savez(f'results/esn_prediction_data/optimized_mps_{N_x}_{seed}.npz', train_U=train_U, train_D=train_D, test_U=test_U, test_D=test_D, test_Y=test_Y)

        with open(f'results/esn_model/optimized_mps_{N_x}_{seed}.pickle', 'wb') as file:
            pickle.dump(model, file)


if __name__ == '__main__':

    study = optuna.create_study(direction='minimize')  # We want to minimize the rmse of test
    study.optimize(objective, n_trials=100)

    print('Best parameters: ', study.best_params)
    os.makedirs('results/optuna', exist_ok=True)
    with open(f'results/optuna/best_params_mps_{N_x}.pickle', 'wb') as file:
        pickle.dump(study.best_params, file)

    save_result(study.best_params)
