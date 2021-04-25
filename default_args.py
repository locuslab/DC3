def baseline_opt_default_args(prob_type):
    defaults = {}
    defaults['simpleVar'] = 100
    defaults['simpleIneq'] = 50
    defaults['simpleEq'] = 50
    defaults['simpleEx'] = 10000
    defaults['nonconvexVar'] = 100
    defaults['nonconvexIneq'] = 50
    defaults['nonconvexEq'] = 50
    defaults['nonconvexEx'] = 10000

    if prob_type == 'simple':
        defaults['corrEps'] = 1e-4
    elif prob_type == 'nonconvex':
        defaults['corrEps'] = 1e-4
    elif 'acopf' in prob_type:
        defaults['corrEps'] = 1e-4

    return defaults

def baseline_nn_default_args(prob_type):
    defaults = {}
    defaults['simpleVar'] = 100
    defaults['simpleIneq'] = 50
    defaults['simpleEq'] = 50
    defaults['simpleEx'] = 10000
    defaults['nonconvexVar'] = 100
    defaults['nonconvexIneq'] = 50
    defaults['nonconvexEq'] = 50
    defaults['nonconvexEx'] = 10000
    defaults['saveAllStats'] = True
    defaults['resultsSaveFreq'] = 50

    if prob_type == 'simple':
        defaults['epochs'] = 1000
        defaults['batchSize'] = 200
        defaults['lr'] = 1e-4
        defaults['hiddenSize'] = 200
        defaults['softWeight'] = 100
        defaults['softWeightEqFrac'] = 0.5
        defaults['useTestCorr'] = True
        defaults['corrTestMaxSteps'] = 10
        defaults['corrEps'] = 1e-4
        defaults['corrLr'] = 1e-7
        defaults['corrMomentum'] = 0.5
    elif prob_type == 'nonconvex':
        defaults['epochs'] = 1000
        defaults['batchSize'] = 200
        defaults['lr'] = 1e-4
        defaults['hiddenSize'] = 200
        defaults['softWeight'] = 100
        defaults['softWeightEqFrac'] = 0.5
        defaults['useTestCorr'] = True
        defaults['corrTestMaxSteps'] = 10
        defaults['corrEps'] = 1e-4
        defaults['corrLr'] = 1e-7
        defaults['corrMomentum'] = 0.5
    elif 'acopf' in prob_type:
        defaults['epochs'] = 1000
        defaults['batchSize'] = 200
        defaults['lr'] = 1e-3
        defaults['hiddenSize'] = 200
        defaults['softWeight'] = 100
        defaults['softWeightEqFrac'] = 0.5
        defaults['useTestCorr'] = True
        defaults['corrTestMaxSteps'] = 5
        defaults['corrEps'] = 1e-4
        defaults['corrLr'] = 1e-5
        defaults['corrMomentum'] = 0.5
    else:
        raise NotImplementedError

    return defaults

def baseline_eq_nn_default_args(prob_type):
    defaults = {}
    defaults['simpleVar'] = 100
    defaults['simpleIneq'] = 50
    defaults['simpleEq'] = 50
    defaults['simpleEx'] = 10000
    defaults['nonconvexVar'] = 100
    defaults['nonconvexIneq'] = 50
    defaults['nonconvexEq'] = 50
    defaults['nonconvexEx'] = 10000
    defaults['saveAllStats'] = True
    defaults['resultsSaveFreq'] = 50

    if prob_type == 'simple':
        defaults['epochs'] = 1000
        defaults['batchSize'] = 200
        defaults['lr'] = 1e-4
        defaults['hiddenSize'] = 200
        defaults['softWeightEqFrac'] = 0.5
        defaults['useTestCorr'] = True
        defaults['corrMode'] = 'partial'
        defaults['corrTestMaxSteps'] = 10
        defaults['corrEps'] = 1e-4
        defaults['corrLr'] = 1e-7
        defaults['corrMomentum'] = 0.5
    elif prob_type == 'nonconvex':
        defaults['epochs'] = 1000
        defaults['batchSize'] = 200
        defaults['lr'] = 1e-4
        defaults['hiddenSize'] = 200
        defaults['softWeightEqFrac'] = 0.5
        defaults['useTestCorr'] = True
        defaults['corrMode'] = 'partial'
        defaults['corrTestMaxSteps'] = 10
        defaults['corrEps'] = 1e-4
        defaults['corrLr'] = 1e-7
        defaults['corrMomentum'] = 0.5
    elif 'acopf' in prob_type:
        defaults['epochs'] = 1000
        defaults['batchSize'] = 200
        defaults['lr'] = 1e-3
        defaults['hiddenSize'] = 200
        defaults['softWeightEqFrac'] = 0.5
        defaults['useTestCorr'] = True
        defaults['corrMode'] = 'full'
        defaults['corrTestMaxSteps'] = 5
        defaults['corrEps'] = 1e-4
        defaults['corrLr'] = 1e-5
        defaults['corrMomentum'] = 0.5
    else:
        raise NotImplementedError

    return defaults

def method_default_args(prob_type):
    defaults = {}
    defaults['simpleVar'] = 100
    defaults['simpleIneq'] = 50
    defaults['simpleEq'] = 50
    defaults['simpleEx'] = 10000
    defaults['nonconvexVar'] = 100
    defaults['nonconvexIneq'] = 50
    defaults['nonconvexEq'] = 50
    defaults['nonconvexEx'] = 10000
    defaults['saveAllStats'] = True
    defaults['resultsSaveFreq'] = 50

    if prob_type == 'simple':
        defaults['epochs'] = 1000
        defaults['batchSize'] = 200
        defaults['lr'] = 1e-4
        defaults['hiddenSize'] = 200
        defaults['softWeight'] = 10          # use 100 if useCompl=False
        defaults['softWeightEqFrac'] = 0.5
        defaults['useCompl'] = True
        defaults['useTrainCorr'] = True
        defaults['useTestCorr'] = True
        defaults['corrMode'] = 'partial'    # use 'full' if useCompl=False
        defaults['corrTrainSteps'] = 10
        defaults['corrTestMaxSteps'] = 10
        defaults['corrEps'] = 1e-4
        defaults['corrLr'] = 1e-7
        defaults['corrMomentum'] = 0.5
    elif prob_type == 'nonconvex':
        defaults['epochs'] = 1000
        defaults['batchSize'] = 200
        defaults['lr'] = 1e-4
        defaults['hiddenSize'] = 200
        defaults['softWeight'] = 10          # use 100 if useCompl=False
        defaults['softWeightEqFrac'] = 0.5
        defaults['useCompl'] = True
        defaults['useTrainCorr'] = True
        defaults['useTestCorr'] = True
        defaults['corrMode'] = 'partial'    # use 'full' if useCompl=False
        defaults['corrTrainSteps'] = 10
        defaults['corrTestMaxSteps'] = 10
        defaults['corrEps'] = 1e-4
        defaults['corrLr'] = 1e-7
        defaults['corrMomentum'] = 0.5
    elif 'acopf' in prob_type:
        defaults['epochs'] = 1000
        defaults['batchSize'] = 200
        defaults['lr'] = 1e-3
        defaults['hiddenSize'] = 200
        defaults['softWeight'] = 10             # use 100 if useCompl=False
        defaults['softWeightEqFrac'] = 0.5
        defaults['useCompl'] = True
        defaults['useTrainCorr'] = True
        defaults['useTestCorr'] = True
        defaults['corrMode'] = 'partial'    # use 'full' if useCompl=False
        defaults['corrTrainSteps'] = 5
        defaults['corrTestMaxSteps'] = 5
        defaults['corrEps'] = 1e-4
        defaults['corrLr'] = 1e-4           # use 1e-5 if useCompl=False
        defaults['corrMomentum'] = 0.5
    else:
        raise NotImplementedError

    return defaults