import cupy as cp
import numpy as np
from mps_evolved import Evolution_MPS
import argparse
import tensornetwork as tn
from tqdm import tqdm
import logging
from logging import getLogger, Formatter, StreamHandler
from utils.information_processing_capacity import single_input_ipc
import pickle
import os
import csv

logger = getLogger(__name__)
logger.setLevel(logging.INFO)
format = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

ch = StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(format)
logger.addHandler(ch)

tn.set_default_backend('pytorch')

parser = argparse.ArgumentParser()

parser.add_argument('--d_bond', type=int)
parser.add_argument('--std', type=float)
parser.add_argument('--rho', type=float)
parser.add_argument('--seed', type=int, default=0)

args = parser.parse_args()
d_bond = args.d_bond
std = args.std
rho = args.rho
seed = args.seed

##### Parameters for mps #####
N = 64      # Number of nodes
Two = 10000 # Washout time
T = int(1e5)# Time length except washout
iota = 0.1  # Input intensity
# rhos = 0.1*np.arange(1,6)  # Spectral radius

os.makedirs('results', exist_ok=True)
os.makedirs('results/function', exist_ok=True)
os.makedirs('results/input', exist_ok=True)

##### Input #####
np.random.seed(seed)
zeta = 2*np.random.rand(Two+T)-1

# Parameters for IPC
poly = 'legendre'
distr = 'uniform'
degdelays = [[1,2000],[2,300],[3,50],[4,30],[5,15]]
# Class for IPC
ipc = single_input_ipc(zeta,Two,degdelays,poly=poly,distr=distr,zerobased=True)

gpu_id = 0
cp.cuda.Device(gpu_id).use()

# Directory
pkldir = 'ipc/pkl'

logger.info(f'd_bond : {d_bond}, std : {std}, rho : {rho}, seed : {seed}')

evolution_func = Evolution_MPS(rho = rho,input_size = N, d_bond=d_bond, std = std)
with open(f'results/function/evolutionfunc_{d_bond}_{std}_{rho}_{seed}.pickle', 'wb') as file:
	pickle.dump(evolution_func, file)

# Matrix Product State
x = np.zeros((N,Two+T))
logger.info('calculate x')
for t in tqdm(range(1,Two+T)):
	x[:, t] = evolution_func(x[:, t-1], zeta[t-1])
np.save(f'results/input/x_{d_bond}_{std}_{rho}_{seed}.npy', x)

logger.info('calculate ipc')
##### Compute IPC of ESN state #####
ipc.svd(x)
path = '%s/state_%d_%d_%5.3f_%5.3f_%5.3f_%d'%(pkldir,N,d_bond,std,rho,iota,seed)
ipc.save_config(path)
Ctot = 0
Cdegs = []
for deg,delay in degdelays:
	ipcs,surs = ipc.compute(deg,delay)
	truncated = ipc.threshold(ipcs,surs,deg,delay,th_scale=1.2)
	Ctot_deg = np.sum(truncated['ipcs'].values)
	logger.info(f'deg {deg} delay {delay} Ctot_deg {Ctot_deg}')
	Ctot += Ctot_deg
	Cdegs.append(Ctot_deg)
logger.info(f'degs {ipc.degs} Ctot {Ctot} rank {ipc.rank}')
logger.info('--------------------------------------------------------------------------------\n\n\n')

csv_path = 'results/result.csv'
if not os.path.exists(csv_path):
	with open('results/result.csv', 'w') as f:
		writer = csv.writer(f)
		writer.writerow(['d_bond', 'std', 'rho', 'seed', 'Ctot_deg1', 'Ctot_deg2', 'Ctot_deg3', 'Ctot_deg4', 'Ctot_deg5', 'Ctot'])

with open('results/result.csv', 'a', newline='') as f:
	writer = csv.writer(f)
	writer.writerow([d_bond, std, rho, seed, Cdegs[0], Cdegs[1], Cdegs[2], Cdegs[3], Cdegs[4], Ctot])
