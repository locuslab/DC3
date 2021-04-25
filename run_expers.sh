#!/bin/bash

## Usage
## bash run_expers.sh > commands
## cat commands | xargs -n1 -P8 -I{} /bin/sh -c "{}" 

# some buffer for GPU scheduling at the start
echo sleep 20
echo sleep 40
echo sleep 60
echo sleep 80
echo sleep 100
echo sleep 120
echo sleep 140

for i in 1 2 3 4 5
do

    for probType in simple
    do

        echo python method.py --probType $probType
        echo python method.py --probType $probType --useCompl False --corrMode full --softWeight 100
        echo python method.py --probType $probType --useTrainCorr False
        echo python method.py --probType $probType --softWeight 0
        echo python baseline_opt.py --probType $probType
        echo python baseline_nn.py --probType $probType
        echo python baseline_eq_nn.py --probType $probType

        for numIneq in 10 30 70 90
        do
            echo python method.py --probType $probType --simpleIneq $numIneq
            echo python method.py --probType $probType --simpleIneq $numIneq --useCompl False --corrMode full --softWeight 100
            echo python method.py --probType $probType --simpleIneq $numIneq --useTrainCorr False
            echo python method.py --probType $probType --simpleIneq $numIneq --softWeight 0
            echo python baseline_opt.py --probType $probType --simpleIneq $numIneq
            echo python baseline_nn.py --probType $probType --simpleIneq $numIneq
            echo python baseline_eq_nn.py --probType $probType --simpleIneq $numIneq
        done

        for numEq in 10 30 70 90
        do
            echo python method.py --probType $probType --simpleEq $numEq
            echo python method.py --probType $probType --simpleEq $numEq --useCompl False --corrMode full --softWeight 100
            echo python method.py --probType $probType --simpleEq $numEq --useTrainCorr False
            echo python method.py --probType $probType --simpleEq $numEq --softWeight 0
            echo python baseline_opt.py --probType $probType --simpleEq $numEq
            echo python baseline_nn.py --probType $probType --simpleEq $numEq
            echo python baseline_eq_nn.py --probType $probType --simpleEq $numEq
        done

    done

    for probType in nonconvex
    do
        echo python method.py --probType $probType
        echo python method.py --probType $probType --useCompl False --corrMode full --softWeight 100
        echo python method.py --probType $probType --useTrainCorr False
        echo python method.py --probType $probType --softWeight 0
        echo python baseline_opt.py --probType $probType
        echo python baseline_nn.py --probType $probType
        echo python baseline_eq_nn.py --probType $probType
    done

    for probType in acopf57
    do
        echo python method.py --probType $probType
        echo python method.py --probType $probType --useCompl False --corrMode full --softWeight 100 --corrLr 1e-5
        echo python method.py --probType $probType --useTrainCorr False
        echo python method.py --probType $probType --softWeight 0
        echo python baseline_opt.py --probType $probType
        echo python baseline_nn.py --probType $probType --runBaselineOpt True
        echo python baseline_eq_nn.py --probType $probType
    done

done
