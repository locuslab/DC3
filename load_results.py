import os
import pickle
import numpy as np

# Load and save experiment status and summary stats
#   Assumes set of experiments and flag settings used in run_expers.sh
def main():
    exper_dirs = get_experiment_dirs('results')
    num_running_done, all_stats = get_status_results(exper_dirs)
    with open('exper_status.dict', 'wb') as f:
        pickle.dump(num_running_done, f)
    with open('results_summary.dict', 'wb') as f:
        pickle.dump(all_stats, f)


# Get directory names for each experiment
#   Returns dict of dirnames indexed by exper_name
def get_experiment_dirs(path_prefix):
    exper_dirs = {}

    eq = 50
    for ineq in [10, 30, 50, 70, 90]:
        exper_dirs['simple_ineq{}_eq{}'.format(ineq, eq)] = 'SimpleProblem-100-{}-{}-10000'.format(ineq, eq)
        
    ineq = 50
    for eq in [10, 30, 70, 90]:
        exper_dirs['simple_ineq{}_eq{}'.format(ineq, eq)] = 'SimpleProblem-100-{}-{}-10000'.format(ineq, eq)
        
    exper_dirs['nonconvex'] = 'NonconvexProblem-100-50-50-10000'

    exper_dirs['acopf'] = 'ACOPF-57-0-0.5-0.7-0.0833-0.0833'

    for key in exper_dirs.keys():
        exper_dirs[key] = os.path.join(path_prefix, exper_dirs[key])

    return exper_dirs


# Get dictionaries with experiment status and summary stats
def get_status_results(exper_dirs):
    num_running_done = {}
    all_stats = {}

    opt_methods = dict([
            ('simple', ['osqp', 'qpth']), ('nonconvex', ['ipopt']), ('acopf', ['pypower'])
    ])
    nn_baseline_dirs = [('baseline_nn', 'baselineNN'), ('baseline_eq_nn', 'baselineEqNN')]

    for exper, exper_dir in exper_dirs.items():
        print(exper)
        
        exper_status_dict = {}
        stats_dict = {}
        
        if os.path.exists(exper_dir):

            ## Get mapping of subdirs to methods
            
            # DC3
            method_path = os.path.join(exper_dir, 'method')
            dir_method_map = get_dc3_path_mapping(method_path)

            # baselines
            all_methods_dirs = nn_baseline_dirs + \
                [('baseline_opt_{}'.format(x), 'baselineOpt-{}'.format(x)) for x in \
                    opt_methods[exper.split('_')[0]]]
            for (method, dirname) in all_methods_dirs:
                path = os.path.join(exper_dir, dirname)
                if os.path.exists(path):
                    path = os.path.join(path, os.listdir(path)[0])
                    dir_method_map[method] = path

            
            ## Get stats
            for method, path in dir_method_map.items():
                print(method)
                aggregate_for_method(method, path, exper_status_dict, stats_dict)
        
        num_running_done[exper] = exper_status_dict
        all_stats[exper] = stats_dict
        
    return num_running_done, all_stats


# Get mapping from DC3 method/ablation name to subdirectory
def get_dc3_path_mapping(method_path):
    dir_method_map = {}
    if os.path.exists(method_path):
        for args_dir in os.listdir(method_path):
            replicate_dirs = os.listdir(os.path.join(method_path, args_dir))
            path = os.path.join(method_path, args_dir, replicate_dirs[0])
            if os.path.exists(os.path.join(path, 'args.dict')):
                with open(os.path.join(path, 'args.dict'), 'rb') as f:
                    args = pickle.load(f)
    
            if not args['useCompl']:
                chosen_method = 'method_no_compl'
            elif not args['useTrainCorr']:
                chosen_method = 'method_no_corr'
            elif args['softWeight'] == 0:
                chosen_method = 'method_no_soft'
            else:
                chosen_method = 'method' 

            dir_method_map[chosen_method] = os.path.join(method_path, args_dir)
    return dir_method_map


# Fill in passed in exper_status and stats dicts with status/summary stats info
def aggregate_for_method(method_name, method_path, exper_status_dict, stats_dict):
    method_stats = []
    sub_dirs = os.listdir(method_path)
    exper_status_dict[method_name] = (0,0)
    
    # get status and stats
    for d2 in sub_dirs:
        is_done, stats = check_running_done(
            os.path.join(method_path, d2), 'opt' in method_name)  # TODO check out this function
        (running, done) = exper_status_dict[method_name]
        if is_done:
            exper_status_dict[method_name] = (running, done + 1)
            method_stats.append(stats)
        else:
            print(os.path.join(method_path, d2))
            exper_status_dict[method_name] = (running + 1, done)
    
        # aggregate metrics (TODO accommodate both data saving cases)
        if len(method_stats) == 0:
            continue
        else:
            metrics = method_stats[0].keys()
            d = {}
            if 'opt' not in method_name:
                for metric in metrics:
                    d[metric] = get_mean_std_nets(method_stats, metric)
            else:
                for metric in metrics:
                    d[metric] = get_mean_std_opts(method_stats, metric)
            stats_dict[method_name] = d


# Check if experiment is running or done, and return stats if done
#   Note: Assumes experiments run for 1000 epochs, and that stats are saved for each epoch
#   for NN methods (i.e., saveAllStats flag is True for each run)
def check_running_done(path, is_opt=False):
    is_done = False
    stats = None
    
    if is_opt:
        if os.path.exists(os.path.join(path, 'results.dict')):
            with open(os.path.join(path, 'results.dict'), 'rb') as f:
                stats = pickle.load(f)
            is_done = True
    else:        
        try:   
            if os.path.exists(os.path.join(path, 'stats.dict')):
                with open(os.path.join(path, 'stats.dict'), 'rb') as f:
                    stats = pickle.load(f)
                is_done = (len(stats['valid_time']) == 1000)
                if not is_done:
                    print(len(stats['valid_time']))
        except Exception as e:
            print(str(e))
            is_done = False
            stats  = None
            
    return is_done, stats


# Compute summary stats for neural network methods (DC3 and baselines)
#   Note: Assumes stats are saved for each epoch (i.e., saveAllStats flag is True for each run)
def get_mean_std_nets(stats_dicts, metric):
    if 'train_time' in metric:
        results = [d[metric].sum() for d in stats_dicts]
    elif 'time' in metric:  
        # test and valid time: use time for latest epoch
        results = [d[metric][-1] for d in stats_dicts]
    else:
        # use mean across samples for latest epoch
        results = [d[metric][-1].mean() for d in stats_dicts]

    # return mean and stddev across replicates
    return np.mean(results), np.std(results)

# Compute summary stats for baseline optimizers
def get_mean_std_opts(stats_dicts, metric):
    if 'time' in metric:
        results = [d[metric] for d in stats_dicts]
    else:
        results = [d[metric].mean() for d in stats_dicts]
    return np.mean(results), np.std(results)


if __name__ == '__main__':
    main()
