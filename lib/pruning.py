"""Layer selection criteria.

Two pruning criteria are supported, both reproduced from their official
implementations:

  * streamline : LLM-Streamline (https://github.com/RUCKBReasoning/LLM-Streamline)
                 removes a single *contiguous* block of `n` layers whose boundary
                 activations have the highest cosine similarity.
  * shortgpt   : ShortGPT (https://github.com/sramshetty/ShortGPT)
                 assigns each layer a Block-Influence (BI) score (1 - cosine
                 similarity between its input and output) and removes the
                 `n` lowest-BI layers one-shot (non-contiguous).

`select_layer` returns `pruned_indices=None` for the contiguous (streamline)
case and an explicit index list for the non-contiguous (shortgpt) case.
"""
import time
import torch
from tqdm import tqdm

from .hadamard_utils import get_hadamard_matrix


# ── scale params (LinearPatch diag/rotate) ───────────────────────────────
def get_scale_params(model, trainloader, i, j, rotate, device):
    model = model.to(device)
    num_layers = len(model.model.layers)
    d_clip = [torch.zeros(1).to(device) for _ in range(num_layers)]
    scale_params = torch.zeros(1).to(device)

    def hook(module, input, output, layer_name, rotate):
        if rotate:
            d_clip[layer_name] = torch.matmul(
                input[0],
                get_hadamard_matrix(input[0].shape[-1], input[0].device).half())
        else:
            d_clip[layer_name] = input[0]

    handles = []
    for l, layer in enumerate(model.model.layers):
        if l == i or l == j:
            handle = layer.register_forward_hook(
                lambda module, input, output, layer_name=l, rotate=rotate:
                    hook(module, input, output, layer_name, rotate))
            handles.append(handle)

    num_samples = num_sample = 128
    calibration_loop = tqdm(enumerate(trainloader, start=0),
                            desc="Calibrating", total=num_samples)
    for num, batch in calibration_loop:
        batch = batch[0].to(device)
        try:
            with torch.no_grad():
                model(batch)
        except IndexError:
            pass
        scale_param = (
            d_clip[j].abs().mean(dim=0, keepdim=True).mean(dim=1, keepdim=True) /
            d_clip[i].abs().mean(dim=0, keepdim=True).mean(dim=1, keepdim=True))
        scale_params = scale_params + scale_param
        num_sample -= 1
        if not num_sample:
            break

    scale_params = scale_params / num_samples
    for handle in handles:
        handle.remove()
    torch.cuda.empty_cache()
    return scale_params[0][0]


# ── LLM-Streamline (contiguous block) ────────────────────────────────────
def get_pruned_layer_streamline(model, trainloader, num_to_prune, device):
    model = model.to(device)
    num_layers = len(model.model.layers)
    max_start = num_layers - num_to_prune
    act = [torch.zeros(1).to(device) for _ in range(num_layers)]
    cosine_sim = [torch.zeros(1).to(device) for _ in range(max_start)]

    def hook(module, input, output, layer_name):
        act[layer_name] = input[0]

    handles = [
        layer.register_forward_hook(
            lambda module, input, output, l=l: hook(module, input, output, l))
        for l, layer in enumerate(model.model.layers)]

    num_samples = num_sample = 128
    for _, batch in tqdm(enumerate(trainloader), desc="Selecting (Streamline)",
                         total=num_samples):
        with torch.no_grad():
            try:
                model(batch[0].to(device))
            except IndexError:
                pass
        for i in range(1, max_start):
            cosine_sim[i] += torch.cosine_similarity(act[i], act[i + num_to_prune]).mean()
        num_sample -= 1
        if not num_sample:
            break

    for h in handles:
        h.remove()

    cosine_sim = [s.item() / num_samples for s in cosine_sim]
    start_l = cosine_sim.index(max(cosine_sim))
    end_l = start_l + num_to_prune

    torch.cuda.empty_cache()
    return num_to_prune, start_l, end_l, max(cosine_sim), None


# ── ShortGPT (Block-Influence, non-contiguous) ───────────────────────────
def get_pruned_layer_shortgpt(model, trainloader, num_to_prune, device):
    model = model.to(device)
    num_layers = len(model.model.layers)
    act = [torch.zeros(1).to(device) for _ in range(num_layers)]
    bi_score = [torch.zeros(1).to(device) for _ in range(num_layers)]

    def hook(module, input, output, l):
        act[l] = input[0]

    handles = [
        layer.register_forward_hook(
            lambda module, input, output, l=l: hook(module, input, output, l))
        for l, layer in enumerate(model.model.layers)]

    num_samples = num_sample = 128
    for _, batch in tqdm(enumerate(trainloader), desc="Selecting (ShortGPT)",
                         total=num_samples):
        with torch.no_grad():
            try:
                model(batch[0].to(device))
            except IndexError:
                pass
        for i in range(num_layers - 1):
            bi_score[i] += torch.cosine_similarity(act[i], act[i + 1]).mean()
        num_sample -= 1
        if not num_sample:
            break

    for h in handles:
        h.remove()

    # BI = 1 - cos(in, out); higher cosine ⇒ lower influence ⇒ prune first.
    bi_score = [s.item() / num_samples for s in bi_score[:-1]]
    pruned_indices = sorted(
        sorted(range(len(bi_score)), key=lambda i: bi_score[i],
               reverse=True)[:num_to_prune])
    start_l = pruned_indices[0]
    end_l = pruned_indices[-1] + 1
    avg_sim = sum(bi_score[i] for i in pruned_indices) / num_to_prune

    print(f"[ShortGPT] pruned_indices: {pruned_indices}")
    torch.cuda.empty_cache()
    return num_to_prune, start_l, end_l, avg_sim, pruned_indices


# ── dispatch ─────────────────────────────────────────────────────────────
def select_layer(model, trainloader, num_to_prune, insert_type, dev,
                 pruning_method="streamline"):
    tick = time.time()
    print("[select_layer] pruning_method=" + pruning_method)

    if pruning_method == "shortgpt":
        layer, start_l, end_l, score, pruned_indices = \
            get_pruned_layer_shortgpt(model, trainloader, num_to_prune, dev)
    elif pruning_method == "streamline":
        layer, start_l, end_l, score, pruned_indices = \
            get_pruned_layer_streamline(model, trainloader, num_to_prune, dev)
    else:
        raise NotImplementedError(
            f"pruning_method={pruning_method}; supported: streamline, shortgpt")

    print("Prune layer in #Layer[" + str(start_l) + ", " + str(end_l) + ")")
    t1 = time.time() - tick
    tick = time.time()

    if insert_type != "none":
        if pruned_indices is not None:
            # non-contiguous: per-boundary patches computed in recovery.py
            scale_params = torch.ones(model.config.hidden_size,
                                      dtype=torch.float32).to(dev)
        else:
            scale_params = get_scale_params(model, trainloader, start_l, end_l,
                                            insert_type == "rotate", dev)
    else:
        scale_params = torch.ones(model.config.hidden_size,
                                  dtype=torch.float32).to(dev)

    t2 = time.time() - tick
    return layer, start_l, end_l, score, scale_params, t1, t2, pruned_indices
