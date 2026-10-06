#!/bin/bash
# Train the full 17-model pool. ~35 h on one 24 GB GPU; resumable (each model
# is cached by artifact tag, so re-running skips whatever already finished).
set -u
cd "$(dirname "$0")/.." || exit 1
GPU="${1:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
# bitsandbytes needs the CUDA libs that ship inside the torch wheel on the path
NV="$(python3 -c 'import site;print(site.getsitepackages()[0])')/nvidia"
[ -d "$NV" ] && export LD_LIBRARY_PATH="$(ls -d $NV/*/lib 2>/dev/null | tr '\n' ':')${LD_LIBRARY_PATH:-}"

COMMON="--folds 5 --max_len 192 --test data/test.csv --train_on_valid --num_workers 4 --artifacts artifacts"
run () { n=$1; shift; echo "=== [$(date +%H:%M)] $n ==="; python3 src/train.py $COMMON --out /tmp/sub_$n.csv "$@"; }

# plain DeBERTa backbone
run deb42    --models deberta --seeds 42   --epochs 5 --run_tag v2
run deb1337  --models deberta --seeds 1337 --epochs 5 --run_tag v2
run deb2024  --models deberta --seeds 2024 --epochs 5 --run_tag v2
run sw3      --models deberta --seeds 42   --epochs 5 --sarcasm_weight 3 --run_tag sw3
run focal    --models deberta --seeds 42   --epochs 5 --focal_gamma 2.0  --run_tag focal
run vaug     --models deberta --seeds 42   --epochs 5 --variety_aug 0.3  --run_tag vaug

# domain-adaptive pretraining on Reddit sarcasm (largest single gain)
run sarcbig42   --models deberta --seeds 42   --epochs 5 --pretrain --pretrain_epochs 1 --run_tag sarcbig
run sarcbig1337 --models deberta --seeds 1337 --epochs 5 --pretrain --pretrain_epochs 1 --run_tag sarcbig
run sarcbig2024 --models deberta --seeds 2024 --epochs 5 --pretrain --pretrain_epochs 1 --run_tag sarcbig
run bigsarc2    --models deberta --seeds 42   --epochs 5 --pretrain --pretrain_epochs 2 --run_tag bigsarc2
run sarcredd42   --models deberta --seeds 42   --epochs 5 --pretrain --pretrain_epochs 1 --run_tag sarcreddit
run sarcredd1337 --models deberta --seeds 1337 --epochs 5 --pretrain --pretrain_epochs 1 --run_tag sarcreddit

# news/tweet TAPT + RoBERTa for architecture diversity
run tapt42   --models deberta --seeds 42   --epochs 5 --pretrain --pretrain_epochs 2 --run_tag tapt2
run tapt1337 --models deberta --seeds 1337 --epochs 5 --pretrain --pretrain_epochs 2 --run_tag tapt2
run rob42    --models roberta --seeds 42   --epochs 5 --run_tag rob
run rob1337  --models roberta --seeds 1337 --epochs 5 --run_tag rob

# 8B decoder (strongest single member)
run llama8b  --models llama8b --seeds 42 --epochs 3 --run_tag l8b
echo "=== pool complete $(date) ==="
