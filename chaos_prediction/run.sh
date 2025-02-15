#!/bin/bash

for seed in {0..9}
do
    for bond in 10 20 30 40 50 60 70 80 90 100
    do
        for std in `seq 0.01 0.01 0.1`
        do
            python3 chaos_prediction.py --d_bond $bond --std $std --rho 1 --seed $seed
        done

        for std in `seq 0.11 0.01 0.19`
        do
            python3 chaos_prediction.py --d_bond $bond --std $std --rho 1 --seed $seed
        done

        for std in `seq 0.2 0.1 0.9`
        do
            python3 chaos_prediction.py --d_bond $bond --std $std --rho 1 --seed $seed
        done
    done

    for bond in 10 20 30 40 50 60 70 80 90 100
    do
        python3 chaos_prediction.py --d_bond $bond --std 0.001 --rho 1 --seed $seed
    done
done

python3 run_esn_optimize.py
python3 run_mps_optimize.py
