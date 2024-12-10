import numpy as np
from sklearn.model_selection import train_test_split
import torch
from sklearn.datasets import load_digits

def data_encoder(data):
  return np.array([1 - data, data]).transpose([1, 2, 0])

def to_one_hot(labels, n_labels=10):
  one_hot = np.zeros((len(labels), n_labels))
  one_hot[np.arange(len(labels)), labels] = 1
  return one_hot

def create_train_test_dataset():
    data_digit = load_digits()
    X = data_digit['data']
    Y = data_digit['target']

    X = X / 16

    X = data_encoder(X)
    Y = to_one_hot(Y)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    x_train = torch.tensor(x_train, dtype=torch.float)
    y_train = torch.tensor(y_train, dtype=torch.float)
    x_test = torch.tensor(x_test, dtype=torch.float)
    y_test = torch.tensor(y_test, dtype=torch.float)

    return x_train, y_train, x_test, y_test

