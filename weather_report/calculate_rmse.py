import numpy as np
from pathlib import Path
import glob
import csv


def calculate_RMSE(true_data, predict_data):
    rmse = np.sqrt(np.mean((true_data - predict_data) ** 2))
    return rmse

def calculate_MAE(true_data, predict_data):
    mae = np.mean(np.abs(true_data - predict_data))
    return mae

def filename_parser(filename):
    base = Path(filename).stem
    d_bond, std, rho, seed = base.split('_')
    return d_bond, std, rho, seed

files = glob.glob('results/prediction_data/*.npz')

with open('results/rmse.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['d_bond', 'std', 'rho', 'seed', 'train_rmse', 'test_rmse', 'train_mae', 'test_mae'])
    for file in files:
        print(file)
        results = np.load(file)
        train_D = results['train_D']
        test_D = results['test_D']
        train_Y = results['train_Y']
        test_Y = results['test_Y']

        train_rmse = calculate_RMSE(train_D, train_Y)
        test_rmse = calculate_RMSE(test_D, test_Y)
        train_mae = calculate_MAE(train_D, train_Y)
        test_mae = calculate_MAE(test_D, test_Y)
        d_bond, std, rho, seed = filename_parser(file)

        writer.writerow([d_bond, std, rho, seed, train_rmse, test_rmse, train_mae, test_mae])
