import numpy as np
from sklearn.model_selection import train_test_split
import torch
from sklearn.datasets import load_digits

def data_encoder(data, input_size):
  return data.reshape(-1, 64 // input_size, input_size)

def to_one_hot(labels, n_labels=10):
  one_hot = np.zeros((len(labels), n_labels))
  one_hot[np.arange(len(labels)), labels] = 1
  return one_hot

def create_train_test_dataset(input_size : int):
    data_digit = load_digits()
    X = data_digit['data']
    Y = data_digit['target']

    X = X / 16

    X = data_encoder(X, input_size)
    Y = to_one_hot(Y)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    return x_train, y_train, x_test, y_test
