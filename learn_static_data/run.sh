# # #!/bin/bash
for seed in {0..9}
    do
    for i in {10..100..10}
    do
        for j in `seq 0.01 0.01 0.1`
        do
            python3 elm_tn.py  --h_size 1000 --d_bond $i --std $j --seed $seed
        done

        python3 elm_tn.py  --h_size 1000 --d_bond $i --std 0.001 --seed $seed
    done

    for i in {11..19..1}
    do
        for j in `seq 0.01 0.01 0.1`
        do
            python3 elm_tn.py  --h_size 1000 --d_bond $i --std $j --seed $seed
        done

        python3 elm_tn.py  --h_size 1000 --d_bond $i --std 0.001 --seed $seed
    done
done

python3 entropy.py
