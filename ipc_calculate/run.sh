# # #!/bin/bash

for seed in {0..9}
do
    for i in {10..100..10}
    do
        for j in `seq 0.01 0.01 0.1`
        do
            python3 calculate_ipc.py --d_bond $i --std $j --rho 1 --seed $seed
        done

        for j in `seq 0.2 0.1 0.9`
        do
            python3 calculate_ipc.py --d_bond $i --std $j --rho 1 --seed $seed
        done

        python3 calculate_ipc.py --d_bond $i --std 0.001 --rho 1 --seed $seed
    done
done

python3 entropy.py
