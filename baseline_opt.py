try:
    import waitGPU
    waitGPU.wait(utilization=50, memory_ratio=0.5, available_memory=5000, interval=9, nproc=1, ngpu=1)
except ImportError:
    pass

import torch
import torch.nn as nn
import torch.optim as optim
torch.set_default_dtype(torch.float64)

import operator
from functools import reduce
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import pickle
import time
from setproctitle import setproctitle
import os
import argparse

from utils import my_hash, str_to_bool
import default_args

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def main():
    parser = argparse.ArgumentParser(description='baseline_opt')
    parser.add_argument('--probType', type=str, default='acopf57',
        choices=['simple', 'nonconvex', 'acopf57'], help='problem type')
    parser.add_argument('--simpleVar', type=int, 
        help='number of decision vars for simple problem')
    parser.add_argument('--simpleIneq', type=int,
        help='number of inequality constraints for simple problem')
    parser.add_argument('--simpleEq', type=int,
        help='number of equality constraints for simple problem')
    parser.add_argument('--simpleEx', type=int,
        help='total number of datapoints for simple problem')
    parser.add_argument('--nonconvexVar', type=int,
        help='number of decision vars for nonconvex problem')
    parser.add_argument('--nonconvexIneq', type=int,
        help='number of inequality constraints for nonconvex problem')
    parser.add_argument('--nonconvexEq', type=int,
        help='number of equality constraints for nonconvex problem')
    parser.add_argument('--nonconvexEx', type=int,
        help='total number of datapoints for nonconvex problem')
    parser.add_argument('--corrEps', type=float,
        help='correction procedure tolerance')

    args = parser.parse_args()
    args = vars(args) # change to dictionary
    defaults = default_args.baseline_opt_default_args(args['probType'])
    for key in defaults.keys():
        if args[key] is None:
            args[key] = defaults[key]
    print(args)

    setproctitle('baselineOpt-{}'.format(args['probType']))

    # Load data, and put on GPU if needed
    prob_type = args['probType']
    if prob_type == 'simple':
        torch.set_default_dtype(torch.float64)
        filepath = os.path.join('datasets', 'simple', "random_simple_dataset_var{}_ineq{}_eq{}_ex{}".format(
            args['simpleVar'], args['simpleIneq'], args['simpleEq'], args['simpleEx']))
    elif prob_type == 'nonconvex':
        filepath = os.path.join('datasets', 'nonconvex', "random_nonconvex_dataset_var{}_ineq{}_eq{}_ex{}".format(
            args['nonconvexVar'], args['nonconvexIneq'], args['nonconvexEq'], args['nonconvexEx']))
    elif prob_type == 'acopf57':
        filepath = os.path.join('datasets', 'acopf', 'acopf57_dataset')
    else:
        raise NotImplementedError

    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    for attr in dir(data):
        var = getattr(data, attr)
        if not callable(var) and not attr.startswith("__") and torch.is_tensor(var):
            try:
                setattr(data, attr, var.to(DEVICE))
            except AttributeError:
                pass
    data._device = DEVICE


    ## Run pure optimization baselines
    if prob_type == 'simple':
        solvers = ['osqp', 'qpth']
    elif prob_type == 'nonconvex':
        solvers = ['ipopt']
    else:
        solvers = ['pypower']

    for solver in solvers:
        save_dir = os.path.join('results', str(data), 'baselineOpt-{}'.format(solver),
            'run', str(time.time()).replace('.', '-'))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        Yvalid_opt, valid_time_total, valid_time_parallel = data.opt_solve(data.validX, solver_type=solver, tol=args['corrEps'])
        Ytest_opt, test_time_total, test_time_parallel = data.opt_solve(data.testX, solver_type=solver, tol=args['corrEps'])
        opt_results = get_opt_results(data, args, torch.tensor(Yvalid_opt).to(DEVICE),
                                       torch.tensor(Ytest_opt).to(DEVICE))
        opt_results.update(
            dict([('test_time', test_time_parallel), ('valid_time', valid_time_parallel), ('train_time', 0),
                ('test_time_total', test_time_total), ('valid_time_total', valid_time_total), ('train_time_total', 0)]))
        with open(os.path.join(save_dir, 'results.dict'), 'wb') as f:
            pickle.dump(opt_results, f)

def get_opt_results(data, args, Yvalid, Ytest, Yvalid_precorr=None, Ytest_precorr=None):
    eps_converge = args['corrEps']
    results = {}
    results['valid_eval'] = data.obj_fn(Yvalid).detach().cpu().numpy()
    results['valid_ineq_max'] = torch.max(data.ineq_dist(data.validX, Yvalid), dim=1)[0].detach().cpu().numpy()
    results['valid_ineq_mean'] = torch.mean(data.ineq_dist(data.validX, Yvalid), dim=1).detach().cpu().numpy()
    results['valid_ineq_num_viol_0'] = torch.sum(data.ineq_dist(data.validX, Yvalid) > eps_converge,
                                               dim=1).detach().cpu().numpy()
    results['valid_ineq_num_viol_1'] = torch.sum(data.ineq_dist(data.validX, Yvalid) > 10 * eps_converge,
                                               dim=1).detach().cpu().numpy()
    results['valid_ineq_num_viol_2'] = torch.sum(data.ineq_dist(data.validX, Yvalid) > 100 * eps_converge,
                                               dim=1).detach().cpu().numpy()
    results['valid_eq_max'] = torch.max(torch.abs(data.eq_resid(data.validX, Yvalid)), dim=1)[0].detach().cpu().numpy()
    results['valid_eq_mean'] = torch.mean(torch.abs(data.eq_resid(data.validX, Yvalid)),
                                              dim=1).detach().cpu().numpy()
    results['valid_eq_num_viol_0'] = torch.sum(torch.abs(data.eq_resid(data.validX, Yvalid)) > eps_converge,
                                                   dim=1).detach().cpu().numpy()
    results['valid_eq_num_viol_1'] = torch.sum(torch.abs(data.eq_resid(data.validX, Yvalid)) > 10 * eps_converge,
                                             dim=1).detach().cpu().numpy()
    results['valid_eq_num_viol_2'] = torch.sum(torch.abs(data.eq_resid(data.validX, Yvalid)) > 100 * eps_converge,
                                                   dim=1).detach().cpu().numpy()

    if Yvalid_precorr is not None:
        results['valid_correction_dist'] = torch.norm(Yvalid - Yvalid_precorr, dim=1).detach().cpu().numpy()
    results['test_eval'] = data.obj_fn(Ytest).detach().cpu().numpy()
    results['test_ineq_max'] = torch.max(data.ineq_dist(data.testX, Ytest), dim=1)[0].detach().cpu().numpy()
    results['test_ineq_mean'] = torch.mean(data.ineq_dist(data.testX, Ytest), dim=1).detach().cpu().numpy()
    results['test_ineq_num_viol_0'] = torch.sum(data.ineq_dist(data.testX, Ytest) > eps_converge,
                                              dim=1).detach().cpu().numpy()
    results['test_ineq_num_viol_1'] = torch.sum(data.ineq_dist(data.testX, Ytest) > 10 * eps_converge,
                                              dim=1).detach().cpu().numpy()
    results['test_ineq_num_viol_2'] = torch.sum(data.ineq_dist(data.testX, Ytest) > 100 * eps_converge,
                                              dim=1).detach().cpu().numpy()
    results['test_eq_max'] = torch.max(torch.abs(data.eq_resid(data.testX, Ytest)), dim=1)[0].detach().cpu().numpy()
    results['test_eq_mean'] = torch.mean(torch.abs(data.eq_resid(data.testX, Ytest)),
                                              dim=1).detach().cpu().numpy()
    results['test_eq_num_viol_0'] = torch.sum(torch.abs(data.eq_resid(data.testX, Ytest)) > eps_converge,
                                                   dim=1).detach().cpu().numpy()
    results['test_eq_num_viol_1'] = torch.sum(torch.abs(data.eq_resid(data.testX, Ytest)) > 10 * eps_converge,
                                             dim=1).detach().cpu().numpy()
    results['test_eq_num_viol_2'] = torch.sum(torch.abs(data.eq_resid(data.testX, Ytest)) > 100 * eps_converge,
                                                   dim=1).detach().cpu().numpy()
    if Ytest_precorr is not None:
        results['test_correction_dist'] = torch.norm(Ytest - Ytest_precorr, dim=1).detach().cpu().numpy()
    return results

# Modifies stats in place
def dict_agg(stats, key, value, op='concat'):
    if key in stats.keys():
        if op == 'sum':
            stats[key] += value
        elif op == 'concat':
            stats[key] = np.concatenate((stats[key], value), axis=0)
        else:
            raise NotImplementedError
    else:
        stats[key] = value

if __name__=='__main__':
    main()