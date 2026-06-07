#!/bin/bash
# Example / manual commands for Ghosted Layers reproduction.
# Naming convention used below:  <recovery>_<eval>_<criterion>_<model>_<nL>
# e.g. ghost_ppl_streamline_llama31_7L
set -e
cd "$(dirname "$0")/.."

TASKS="arc_easy,arc_challenge,hellaswag,winogrande,boolq,openbookqa,rte,copa,race"
COMMON="--calibration_data c4 --train_size 128 --ghost_max_batches 32 \
        --eval_ppl --eval_tasks $TASKS --eval_batch_size 8"

# ============================================================
# LLaMA-3.1-8B, LLM-Streamline, n=7  — recovery comparison
# ============================================================
MODEL="meta-llama/Llama-3.1-8B"

# dense reference
python main.py --model $MODEL --pruning_method none --insert_type none \
    --eval_ppl --eval_tasks $TASKS --outdir dense_llama31

# pruned, no recovery
python main.py --model $MODEL --pruning_method streamline --total_num_prune 7 \
    --insert_type none $COMMON --outdir pruned_streamline_llama31_7L

# LinearPatch-Diag / -Rotate
python main.py --model $MODEL --pruning_method streamline --total_num_prune 7 \
    --insert_type diag   $COMMON --outdir diag_streamline_llama31_7L
python main.py --model $MODEL --pruning_method streamline --total_num_prune 7 \
    --insert_type rotate $COMMON --outdir rotate_streamline_llama31_7L

# Ghosted Layers (ours)
python main.py --model $MODEL --pruning_method streamline --total_num_prune 7 \
    --insert_type ghost  $COMMON --outdir ghost_streamline_llama31_7L

# ============================================================
# LLaMA-3-8B, ShortGPT (non-contiguous), n=11
# ============================================================
MODEL="meta-llama/Meta-Llama-3-8B"

python main.py --model $MODEL --pruning_method shortgpt --total_num_prune 11 \
    --insert_type rotate $COMMON --outdir rotate_shortgpt_llama3_11L
python main.py --model $MODEL --pruning_method shortgpt --total_num_prune 11 \
    --insert_type ghost  $COMMON --outdir ghost_shortgpt_llama3_11L

echo "[ALL DONE]"
