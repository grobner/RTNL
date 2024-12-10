import pickle
import scipy
import numpy as np
from math import log
import pandas as pd
from canonicalize_mps import canonicalize_mps

def calculate_entropy_from_S(S):
    norm_S = np.linalg.norm(S)
    S = S / norm_S
    SvN = 0
    for s in S:
        p = s ** 2
        SvN += -p * log(p)
    return SvN

def calculate_entropy(model_file):
    with open(model_file, 'rb') as file:
        elm = pickle.load(file)

    mps = canonicalize_mps(elm)
    mps.position(5)
    temp = mps.tensors[5]

    temp2 = temp.reshape(np.prod(temp.shape[0:2]), temp.shape[2])
    U, S, Vt = scipy.linalg.svd(temp2, full_matrices=False)

    svn = calculate_entropy_from_S(S)
    return svn
seeds = list(range(6)) # FIXME: 6 -> 10
bonds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
stds = [0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
rho = 1.0
entropy_data = []
for seed in seeds:
    for bond in bonds:
        for std in stds:
            entropy = calculate_entropy(f'results/function/evolutionfunc_{bond}_{std}_{rho}_{seed}.pickle')
            entropy_data.append([bond, std, seed, entropy])
            print(bond, std, seed, entropy)
df_entropy = pd.DataFrame(entropy_data, columns=['bond', 'std', 'seed', 'entropy'])

df_entropy.to_csv('results/entropy.csv', index=False)
