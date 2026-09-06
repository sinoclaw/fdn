#!/bin/bash
# FDN-v0.5 批次1：多 seed 复现 w=0.5（并行跑 6 个 seed）
cd /data/fdn
SEEDS="0 1 2 3 4 5"
for s in $SEEDS; do
    (
        .venv/bin/python experiments/continual.py --seq core --seed $s \
            --n_train 1200 --epochs_per_task 40 --run C --profile v04lite \
            --soft_task_bias 0.5 --out results/summary_v05_seed$s.json
    ) &
done
wait
echo "ALL_SEEDS_DONE"
