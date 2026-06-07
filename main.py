"""Ghosted Layers — reproduction entry point.

Pipeline:  load model → select layers to prune → (optionally) compute a
training-free recovery operator at the boundary → remove layers → evaluate
(perplexity + zero-shot commonsense QA via lm-evaluation-harness).

Supported pruning criteria : streamline (contiguous), shortgpt (non-contiguous).
Supported recovery operators: none, diag / rotate (LinearPatch), ghost (ours).

Code builds on the official LinearPatch repository:
    https://github.com/chenxinrui-tsinghua/LinearPatch
"""
import argparse
import os
import json
import random
import time
import copy

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version

from lib.data import get_loaders, test_ppl
from lib.pruning import select_layer
from lib.recovery import (
    register_linear_patch,
    register_linear_patch_multi,
    compute_linear_patch_multi,
    compute_ghosted_layer,
    register_ghosted_layer,
    compute_ghosted_layer_multi,
    register_ghosted_layer_multi,
)

os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"

print("torch", version("torch"))
print("transformers", version("transformers"))
print("# of gpus:", torch.cuda.device_count())


def get_llm(model_name, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="cuda",
        attn_implementation="sdpa",
    )
    return model


def json_serializer(obj):
    if hasattr(obj, "tolist"):
        return obj.tolist()
    try:
        return str(obj)
    except Exception:
        return "Unserializable Object"


@torch.no_grad()
def evaluate(model, tokenizer, args, device, save_folder_path=None, tag=None):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    model.config.return_dict = True
    model.to(device)
    model.eval()
    results = {}

    if args.eval_ppl:
        datasets = ["wikitext2", "ptb", "c4"]
        print("Start ppl evaluation.")
        ppl_results = test_ppl(model, tokenizer, datasets, args.ppl_seqlen)
        for dataset in ppl_results:
            print(f"{dataset} perplexity: {ppl_results[dataset]:.2f}")
            if save_folder_path is not None and tag is not None:
                with open(os.path.join(save_folder_path, f"ppl_{dataset}_{tag}.txt"), "w") as f:
                    f.write(f"{ppl_results[dataset]:.4f}\n")

    if args.eval_tasks != "":
        import lm_eval
        from lm_eval.models.huggingface import HFLM
        from lm_eval.utils import make_table
        task_list = [t.strip() for t in args.eval_tasks.split(",") if t.strip()]
        print(f"Evaluation downstream task list: {task_list}")
        lm = HFLM(pretrained=model, tokenizer=tokenizer,
                  batch_size=args.eval_batch_size, device="cuda")
        results = lm_eval.simple_evaluate(model=lm, tasks=task_list,
                                          batch_size=args.eval_batch_size,
                                          num_fewshot=0)
        print(make_table(results))
        total_acc = sum(results["results"][t]["acc,none"] for t in task_list)
        print(f"Average Acc: {total_acc/len(task_list)*100:.2f}%")

    model.config.use_cache = use_cache
    return results


def save_results(results, save_file_path, tag):
    with open(save_file_path, "a") as f:
        json.dump(f"\n[{tag}]\n", f, indent=4, ensure_ascii=False, default=json_serializer)
        json.dump(results, f, indent=4, ensure_ascii=False, default=json_serializer)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--cache_dir", default="llm_weights", type=str)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--ppl_seqlen", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_size", type=int, default=128,
                        help="number of calibration sequences (paper default: 128)")
    parser.add_argument("--val_size", type=int, default=16)

    parser.add_argument("--pruning_method", type=str, default="streamline",
                        choices=["none", "streamline", "shortgpt"],
                        help="none: dense eval | streamline: contiguous block | "
                             "shortgpt: BI-score (non-contiguous)")
    parser.add_argument("--total_num_prune", type=int, default=7,
                        help="number of layers to remove (n)")

    parser.add_argument("--insert_type", type=str, default="none",
                        choices=["none", "diag", "rotate", "ghost"],
                        help="none: no recovery | diag/rotate: LinearPatch | "
                             "ghost: Ghosted Layers (ours)")

    parser.add_argument("--calibration_data", type=str, default="c4",
                        choices=["wikitext2", "c4", "ptb"])
    parser.add_argument("--ghost_max_batches", type=int, default=32,
                        help="calibration batches for operator estimation (paper: 32)")

    parser.add_argument("--eval_ppl", action="store_true")
    parser.add_argument("--eval_tasks", type=str,
                        default="arc_easy,arc_challenge,hellaswag,winogrande,"
                                "boolq,openbookqa,rte,copa,race")
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--outdir", type=str, default="RESULTS")

    args = parser.parse_args()
    model_name = args.model.split("/")[-1]

    save_folder_path = (
        f"results/{args.outdir}/results_{model_name}"
        f"_pruning_{args.pruning_method}"
        f"_insert_type_{args.insert_type}"
    )
    os.makedirs(save_folder_path, exist_ok=True)
    save_file_path = os.path.join(
        save_folder_path, f"calibration_data_{args.calibration_data}_results.txt")
    print(f"Results path: {save_folder_path}")

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    print("=" * 77)
    print(f"model={args.model}  pruning={args.pruning_method}  recovery={args.insert_type}")
    print("=" * 77)

    model = get_llm(args.model, args.cache_dir)
    model.eval()
    device = next(model.parameters()).device
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False, legacy=False)
    for p in model.parameters():
        p.requires_grad = False

    trainloader, valloader = get_loaders(
        name=args.calibration_data, tokenizer=tokenizer,
        train_size=args.train_size, val_size=args.val_size,
        seed=args.seed, seqlen=args.seqlen)

    # ── Case 1: dense ────────────────────────────────────────────────────
    if args.pruning_method == "none":
        print("[Dense] Evaluating unpruned model...")
        result = evaluate(model, tokenizer, args, device, save_folder_path, tag="dense")
        save_results(result, save_file_path, tag="Dense")
        return

    init_num_layer = len(model.model.layers)
    before_params = sum(p.numel() for p in model.parameters())

    # ── layer selection ──────────────────────────────────────────────────
    _insert_for_select = args.insert_type if args.insert_type in ["diag", "rotate"] else "none"
    layer, start_l, end_l, _, scale_params, t1, t2, pruned_indices = select_layer(
        model, trainloader, args.total_num_prune, _insert_for_select, device,
        pruning_method=args.pruning_method)
    print(f"Layer select: {t1:.1f}s, scale calc: {t2:.1f}s")
    print(f"Pruning {layer}/{init_num_layer} layers [{start_l}, {end_l})")

    is_non_contiguous = pruned_indices is not None

    # ── precompute Ghosted Layers on the FULL model before removal ───────
    # (no teacher deepcopy needed for non-contiguous ghost)
    precomputed_recovery = None
    teacher = None
    if is_non_contiguous and args.insert_type == "ghost":
        print("[MemEff] precompute Ghosted Layers (multi) on full model before pruning")
        precomputed_recovery = compute_ghosted_layer_multi(
            model, pruned_indices, trainloader, device, max_batches=args.ghost_max_batches)
        torch.cuda.empty_cache()
    elif args.insert_type != "none":
        # contiguous recovery (and contiguous ghost) needs a pre-pruning copy
        teacher = copy.deepcopy(model).eval()

    # ── remove layers ────────────────────────────────────────────────────
    if pruned_indices is not None:
        keep = set(range(len(model.model.layers))) - set(pruned_indices)
        model.model.layers = torch.nn.ModuleList(
            [l for i, l in enumerate(model.model.layers) if i in keep])
    else:
        model.model.layers = torch.nn.ModuleList(
            [l for i, l in enumerate(model.model.layers) if i < start_l or i >= end_l])
    pmodel = model

    # ── Case 2: pruned, no recovery ──────────────────────────────────────
    if args.insert_type == "none":
        print("[Pruned] Evaluating pruned model (no recovery)...")
        result = evaluate(pmodel, tokenizer, args, device, save_folder_path, tag="pruned")
        save_results(result, save_file_path, tag="Pruned")
        return

    # ── Case 3: pruned + recovery ────────────────────────────────────────
    if args.insert_type == "ghost":
        if is_non_contiguous:
            print("[Recovery] Ghosted Layers (multi)")
            ghosted_list = precomputed_recovery
            register_ghosted_layer_multi(pmodel, pruned_indices, ghosted_list, device)
        else:
            print("[Recovery] Ghosted Layers")
            ghosted = compute_ghosted_layer(teacher, pmodel, start_l, end_l,
                                            trainloader, device,
                                            max_batches=args.ghost_max_batches)
            register_ghosted_layer(pmodel, start_l, ghosted, device)

    elif args.insert_type in ("diag", "rotate"):
        rotated = args.insert_type == "rotate"
        if is_non_contiguous:
            print(f"[Recovery] LinearPatch-{'R' if rotated else 'D'} (multi)")
            lp_list = compute_linear_patch_multi(teacher, pruned_indices, trainloader,
                                                 device, rotated=rotated,
                                                 max_batches=args.ghost_max_batches)
            register_linear_patch_multi(pmodel, pruned_indices, lp_list, device)
        else:
            print(f"[Recovery] LinearPatch-{'R' if rotated else 'D'}")
            register_linear_patch(pmodel, start_l, scale_params, device, rotated=rotated)

    after_params = sum(p.numel() for p in pmodel.parameters())
    print(f"#PruneLayer: {args.total_num_prune}  "
          f"#Param before: {before_params}  #Param after: {after_params}  "
          f"PruneRatio: {100 - 100.0*after_params/before_params:.4f}%")

    print(f"[{args.insert_type}] Evaluating pruned model with recovery...")
    results = evaluate(pmodel, tokenizer, args, device, save_folder_path, tag=args.insert_type)
    save_results(results, save_file_path, tag=args.insert_type)


if __name__ == "__main__":
    main()
