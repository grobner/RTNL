import torch
import numpy as np
import tensornetwork as tn
import logging
from logging import getLogger, Formatter, StreamHandler
import argparse
from elm_mps import ELM_MPS
import pickle
from utilities import create_train_test_dataset
import csv
import os

logger = getLogger(__name__)
logger.setLevel(logging.INFO)
format = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

ch = StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(format)
logger.addHandler(ch)

x_train, y_train, x_test, y_test = create_train_test_dataset()

n_labels = len(np.unique(y_train))

tn.set_default_backend('pytorch')
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--h_size', type=int)
parser.add_argument('--d_bond', type=int)
parser.add_argument('--std', type=float)
parser.add_argument('--seed', type=int, default=0)

args = parser.parse_args()

h_size = args.h_size
d_bond = args.d_bond
std = args.std
seed = args.seed

if torch.cuda.is_available():
  device = torch.device('cuda:0')
  logger.info('use GPU')
else:
  device = torch.device('cpu')
  logger.info('use CPU')

logger.info(f'h_size : {h_size}, d_bond : {d_bond}, std : {std}, seed : {seed}')

np.random.seed(seed)

elm = ELM_MPS(input_size = x_train.shape[1], h_size = h_size, num_classes=10, d_phys=2, d_bond=d_bond, std=std, device=device)

elm.fit(x_train, y_train)

y_predict = elm.predict(x_train)
y_predict = y_predict.to('cpu')
istrue_train = y_train.argmax(axis=1) == y_predict.argmax(axis=1)
istrue_train = istrue_train.float()
train_accuracy = istrue_train.mean().item()
logger.info(f'train accuracy : {train_accuracy}')

y_predict = elm.predict(x_test)
y_predict = y_predict.to('cpu')
istrue_test = y_predict.argmax(axis=1) == y_test.argmax(axis=1)
istrue_test = istrue_test.float()
test_accuracy = istrue_test.mean().item()
logger.info(f'test accuracy : {test_accuracy}')

os.makedirs('results', exist_ok=True)
os.makedirs('results/training_model', exist_ok=True)

csv_path = 'results/result.csv'
if not os.path.exists(csv_path):
  with open('results/result.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['h_size', 'd_bond', 'std', 'seed', 'train_acc', 'test_acc'])

with open('results/result.csv', 'a', newline='') as f:
  writer = csv.writer(f)
  writer.writerow([h_size, d_bond, std, seed, train_accuracy, test_accuracy])

with open(f'results/training_model/{d_bond}_{std}_{seed}.pickle', 'wb') as file:
  pickle.dump(elm, file)
