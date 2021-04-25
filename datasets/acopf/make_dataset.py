import os
import pickle
import torch

import sys
sys.path.insert(1, os.path.join(sys.path[0], os.pardir, os.pardir))
from utils import ACOPFProblem

nbus = 57
num = 1200

torch.set_default_dtype(torch.float64)
filepath = os.path.join('matlab_datasets','FeasiblePairs_Case{}.mat'.format(nbus))
problem = ACOPFProblem(filepath)

# Cut down number of samples if needed
problem._X = problem.X[:num, :]
problem._Y = problem.Y[:num, :]
problem._num =  problem.X.shape[0]

with open("./acopf{}_dataset".format(nbus), 'wb') as f:
    pickle.dump(problem, f)


