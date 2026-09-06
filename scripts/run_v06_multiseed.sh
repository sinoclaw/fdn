#!/bin/bash
# FDN-v0.6 批次4：多 seed 验证 Node 自主专化（A→B→A' 最简序列，并行跑 6 个 seed）
# 依据 GPT 二审判据：A_end>0.4 / B>0.3 / A2>0.35 / forgetting<0.05 / cos(A,A')>0.7 / cos(A,B)<0.5 / disjoint>0
cd /data/fdn
SEEDS="0 1 2 3 4 5"
for s in $SEEDS; do
    (
        .venv/bin/python experiments/continual.py --seq mini --seed $s \
            --n_train 1200 --epochs_per_task 40 --run C --profile v06 \
            --out results/summary_v06_seed$s.json
    ) &
done
wait
echo "ALL_SEEDS_DONE"
