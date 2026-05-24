#!/bin/bash
# Run: bash check_training.sh
if ps aux | grep -E "python.*(retrain|atlas_patch|train_auto|train_stage)" | grep -v grep > /dev/null; then
    echo "训练在跑:"
    ps aux | grep -E "python.*(retrain|atlas_patch|train_auto|train_stage)" | grep -v grep
else
    echo "无训练进程"
fi
echo "---"
LATEST=$(find /home/ljq/code/PINN/SolvingTeukolskyEq_autoencoder/outputs/stage1_retrain/ -name "history.jsonl" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
echo "最新日志: ${LATEST:-无}"
[ -n "$LATEST" ] && tail -3 "$LATEST"
