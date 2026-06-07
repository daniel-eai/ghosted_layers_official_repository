# Ghosted Layers: Unconstrained Activation Alignment for Recovering Layer-Pruned LLMs

Training-free recovery for layer-pruned large language models. After removing
Transformer decoder blocks, **Ghosted Layers** inserts a single closed-form
linear operator `W* = I + M*` at each pruning boundary that reconstructs the
boundary-activation gap — the unconstrained optimum of the same alignment
objective that **LinearPatch** solves only over the symmetric subspace.

This repository is a minimal reproduction comparing **LinearPatch** (`diag` /
`rotate`) against **Ghosted Layers** (`ghost`) under two layer-selection
criteria (**LLM-Streamline** and **ShortGPT**), measuring perplexity and
zero-shot commonsense QA accuracy.

> Our code is built on the official LinearPatch repository:
> **https://github.com/chenxinrui-tsinghua/LinearPatch**

---

## Method in one paragraph

Layer pruning removes a block `B = {ℓ*, …, ℓ*+n-1}` of `n` decoder layers. In a
pre-norm residual network the post-boundary hidden state of the *unpruned* model
is `X_post = X_pre + Δ`, where `Δ = Σ f(X)` is the contribution of the removed
block. The pruned model simply forwards `X_pre` to the next surviving layer, so
every downstream layer receives the wrong input. We collect `X_pre, X_post` from
a small calibration set and solve, in closed form,

```
W* = argmin_W || X_pre · W − X_post ||_F²        (reparam. W = I + M)
M* = argmin_M || X_pre · M − Δ ||_F²  =  X_pre⁺ Δ
```

via the regularized normal equations `(XᵀX + εI) M* = Xᵀ Δ` (float64, ε=1e-6).
At inference the boundary computes `x_new = x W* = x + x M*` — a single `C×C`
matmul, identical in cost to LinearPatch's fused `H D Hᵀ`.

**Why it beats LinearPatch (Theorem 4.1).** LinearPatch's operator
`W_LP = H D Hᵀ` is symmetric by construction, so it lives in the
`C(C+1)/2`-dimensional symmetric subspace. The optimal `M*` has a *substantial
anti-symmetric component* (≈47% of its Frobenius norm), which is structurally
unreachable by any symmetric operator. Ghosted Layers attains the full-space
optimum at the same inference cost.

---

## What this repo supports

| Axis | Options |
|---|---|
| Pruning criterion | `streamline` (contiguous block), `shortgpt` (BI-score, non-contiguous), `none` (dense) |
| Recovery operator | `none`, `diag` (LinearPatch-D), `rotate` (LinearPatch-R), `ghost` (Ghosted Layers, ours) |
| Calibration corpus | `c4` (default), `wikitext2`, `ptb` |
| Perplexity eval | WikiText-2, PTB, C4 |
| Accuracy eval | 9 zero-shot commonsense QA tasks (lm-evaluation-harness) |

Recovery works for both contiguous pruning (one boundary operator) and
non-contiguous pruning (one operator per removed layer, inserted via forward
hooks after re-indexing).

---

## Installation

The environment matches the official **LinearPatch** repository — follow its
setup, then this repo runs as-is:

```bash
# 1) follow https://github.com/chenxinrui-tsinghua/LinearPatch for the base env
git clone https://github.com/chenxinrui-tsinghua/LinearPatch
# create the conda/pip environment as instructed there (PyTorch + transformers)

# 2) extra deps used here
pip install -r requirements.txt
```

Key versions (single NVIDIA A40 48GB used in the paper):
`torch`, `transformers`, `datasets`, `lm-eval` (lm-evaluation-harness),
`scikit-learn` is **not** required (clustering criteria are not part of this repo).

---

## Usage

```bash
python main.py \
    --model meta-llama/Llama-3.1-8B \
    --pruning_method streamline \
    --total_num_prune 7 \
    --insert_type ghost \
    --calibration_data c4 \
    --train_size 128 --ghost_max_batches 32 \
    --eval_ppl \
    --eval_tasks arc_easy,arc_challenge,hellaswag,winogrande,boolq,openbookqa,rte,copa,race \
    --outdir llama31_streamline_7L_ghost
```

Key flags:

| Flag | Meaning |
|---|---|
| `--pruning_method` | `streamline` \| `shortgpt` \| `none` |
| `--total_num_prune` | number of layers removed `n` (e.g. 7 or 11 for 32-layer models) |
| `--insert_type` | `none` \| `diag` \| `rotate` \| `ghost` |
| `--calibration_data` | `c4` \| `wikitext2` \| `ptb` (paper default: c4) |
| `--train_size` | # calibration sequences (paper default: 128; accuracy saturates at 32) |
| `--ghost_max_batches` | # batches used to estimate the operator (paper default: 32) |
| `--eval_ppl` | also report WikiText-2 / PTB / C4 perplexity |

### Manual example commands

```bash
# Ghosted Layers — ppl + acc, LLM-Streamline, LLaMA-3.1-8B, 7 layers
python main.py --model meta-llama/Llama-3.1-8B \
    --pruning_method streamline --total_num_prune 7 --insert_type ghost \
    --eval_ppl --outdir ghost_streamline_llama31_7L

# LinearPatch-Rotate — same setting, for comparison
python main.py --model meta-llama/Llama-3.1-8B \
    --pruning_method streamline --total_num_prune 7 --insert_type rotate \
    --eval_ppl --outdir rotate_streamline_llama31_7L

# Ghosted Layers — ShortGPT (non-contiguous), LLaMA-3-8B, 11 layers
python main.py --model meta-llama/Meta-Llama-3-8B \
    --pruning_method shortgpt --total_num_prune 11 --insert_type ghost \
    --eval_ppl --outdir ghost_shortgpt_llama3_11L

# Pruned baseline — no recovery (LLM-Streamline)
python main.py --model meta-llama/Llama-3.1-8B \
    --pruning_method streamline --total_num_prune 7 --insert_type none \
    --eval_ppl --outdir pruned_streamline_llama31_7L

# Dense reference
python main.py --model meta-llama/Llama-3.1-8B \
    --pruning_method none --insert_type none --eval_ppl --outdir dense_llama31
```

A ready-made batch of these is in [`scripts/run_examples.sh`](scripts/run_examples.sh).

---

## Reproduced results (from the paper)

### Zero-shot accuracy, LLM-Streamline, 7/11-layer pruning (Table 2, AVG over 9 tasks)

| Model | n/L | Pruned | LinearPatch-D | LinearPatch-R | **Ghost (ours)** |
|---|---|---|---|---|---|
| LLaMA-3-8B    | 7/32  | 40.70 | 48.65 | 50.55 | **60.10** |
| LLaMA-3-8B    | 11/32 | 44.24 | 51.40 | 51.37 | **53.66** |
| LLaMA-3.1-8B  | 7/32  | 42.60 | 50.46 | 53.27 | **60.01** |
| LLaMA-3.1-8B  | 11/32 | 44.68 | 52.39 | 52.39 | **53.92** |
| DeepSeek-R1-Distill-8B | 7/32 | 48.01 | 51.68 | 53.59 | **57.80** |
| DeepSeek-R1-Distill-8B | 11/32 | 46.52 | 50.40 | 50.37 | **52.11** |

### Perplexity, three pruning criteria, LLaMA-3.1-8B 7-layer (Table 3, PPL AVG ↓)

| Criterion | LinearPatch-D | LinearPatch-R | **Ghost (ours)** |
|---|---|---|---|
| LLM-Streamline | 127.04 | 66.06 | **27.81** |
| ShortGPT       |  32.71 | 37.31 | **20.82** |

Ghosted Layers attains the highest accuracy and lowest perplexity in every
setting, at **matched inference cost** with LinearPatch (a single `C×C` matmul;
identical prefill latency within measurement noise, see Table 5/A8/A9 in the
paper). Accuracy already saturates at **32 calibration sequences**.

---

## Repository layout

```
ghosted_layers/
├── main.py                 # entry point: load → select → recover → evaluate
├── lib/
│   ├── data.py             # WikiText-2 / PTB / C4 loaders + perplexity
│   ├── pruning.py          # LLM-Streamline + ShortGPT layer selection
│   ├── recovery.py         # LinearPatch (diag/rotate) + Ghosted Layers (ours)
│   └── hadamard_utils.py   # Walsh–Hadamard matrices (for LinearPatch-Rotate)
├── scripts/
│   └── run_examples.sh     # example / manual commands
└── requirements.txt
```

---

## Citation

```bibtex
@article{yun2026ghosted,
  title   = {Ghosted Layers: Unconstrained Activation Alignment for Recovering Layer-Pruned LLMs},
  author  = {Yun, Vincent-Daniel and Jo, Junhyuk and Karimireddy, Sai Praneeth and Lee, Sunwoo},
  journal = {arXiv preprint arXiv:2605.15491},
  year    = {2026}
}
```

### Acknowledgements / baselines

- **LinearPatch** (recovery baseline; our code builds on it): https://github.com/chenxinrui-tsinghua/LinearPatch
- **LLM-Streamline** (pruning criterion): https://github.com/RUCKBReasoning/LLM-Streamline
- **ShortGPT** (pruning criterion): https://github.com/sramshetty/ShortGPT
- **lm-evaluation-harness** (evaluation): https://github.com/EleutherAI/lm-evaluation-harness
