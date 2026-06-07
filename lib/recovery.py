"""Boundary recovery operators inserted at the pruning boundary.

Two training-free recovery families are supported:

  * LinearPatch (diag / rotate)  — symmetric operator W_LP = H D H^T
        Reproduced from https://github.com/chenxinrui-tsinghua/LinearPatch
        `diag`   : channel-wise scaling only (D)
        `rotate` : Hadamard-rotated channel-wise scaling (H D H^T)

  * Ghosted Layers (ours)        — unconstrained operator W* = I + M*
        M* = X_pre^+ Δ  is the closed-form minimum-norm least-squares solution
        to  min_M || X_pre M - Δ ||_F^2 , Δ = X_post - X_pre, solved via the
        regularized normal equations (X^T X + εI) M* = X^T Δ in float64.
        forward:  x_new = x W* = x + x M*.

Both operators reduce to a single C×C matmul at the boundary, so they share the
same inference cost; Ghosted Layers searches the full operator space whereas
LinearPatch is restricted to the symmetric subspace.

Insertion handles both contiguous pruning (single boundary, `*_layer`) and
non-contiguous pruning (one operator per removed layer, `*_multi`).
"""
import torch
import torch.nn as nn

from .hadamard_utils import get_hadamard_matrix


def generate_symmetric_matrix(diag):
    orth = get_hadamard_matrix(diag.shape[-1], diag.device).to(torch.float32)
    diag = torch.diag(diag)
    return torch.matmul(orth, torch.matmul(diag, orth.T)).to(diag.device)


# ── operators ────────────────────────────────────────────────────────────
class LinearPatch(nn.Module):
    """W_LP = H D H^T (rotate) or D (diag). Symmetric by construction."""
    def __init__(self, diag=None, rotated=True, weight=None, d_weight=None):
        super().__init__()
        if weight is not None:
            self.weight = nn.Parameter(weight)
        elif rotated and diag is not None:
            self.weight = nn.Parameter(generate_symmetric_matrix(diag))
        else:
            self.weight = nn.Parameter(torch.diag(diag))
        if d_weight is not None:
            self.register_buffer("d_weight", d_weight.to(torch.float32).view(1, 1, -1))
        else:
            self.d_weight = None

    def forward(self, x):
        x_fp32 = x.to(torch.float32)
        if self.d_weight is not None:
            x_fp32 = x_fp32 * self.d_weight
        return torch.matmul(x_fp32, self.weight.to(torch.float32)).to(torch.float16)

    def get_weight(self):
        return self.weight.to(torch.float32)


class GhostedLayer(nn.Module):
    """W* = I + M*,  forward: x_new = x @ W* = x + x @ M*."""
    def __init__(self, W: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(W.to(torch.float32))

    def forward(self, x):
        return torch.matmul(x.to(torch.float32), self.weight).to(x.dtype)

    def get_weight(self):
        return self.weight.to(torch.float32)


# ── boundary activation capture ──────────────────────────────────────────
def _collect_boundary_activations(teacher, start_l, end_l, trainloader, device,
                                  max_batches):
    """Collect X_pre (input of layer start_l) and X_post (input of layer end_l)
    from the unpruned model via forward pre-hooks."""
    X_pre_list, X_post_list = [], []

    def hook_pre(module, args):
        hs = args[0]
        if isinstance(hs, (tuple, list)):
            hs = hs[0]
        X_pre_list.append(hs.detach().float().cpu())

    def hook_post(module, args):
        hs = args[0]
        if isinstance(hs, (tuple, list)):
            hs = hs[0]
        X_post_list.append(hs.detach().float().cpu())

    h1 = teacher.model.layers[start_l].register_forward_pre_hook(hook_pre)
    h2 = teacher.model.layers[end_l].register_forward_pre_hook(hook_post)

    with torch.no_grad():
        for step, batch in enumerate(trainloader):
            if step >= max_batches:
                break
            input_ids = (batch[0].to(device) if isinstance(batch, (tuple, list))
                         else batch["input_ids"].to(device))
            teacher(input_ids=input_ids, use_cache=False)

    h1.remove()
    h2.remove()

    X_pre = torch.cat([x.reshape(-1, x.shape[-1]) for x in X_pre_list], dim=0)
    X_post = torch.cat([x.reshape(-1, x.shape[-1]) for x in X_post_list], dim=0)
    return X_pre, X_post


def _compute_ghosted_from_activations(X_pre, X_post, device):
    """Closed-form W* = I + M* via regularized normal equations (float64)."""
    C = X_pre.shape[1]
    X_pre = X_pre.to(device=device, dtype=torch.float64)
    X_post = X_post.to(device=device, dtype=torch.float64)
    Delta = X_post - X_pre

    XtX = X_pre.T @ X_pre
    XtD = X_pre.T @ Delta
    reg = 1e-6 * torch.eye(C, device=device, dtype=torch.float64)
    M_star = torch.linalg.solve(XtX + reg, XtD)
    W_star = torch.eye(C, device=device, dtype=torch.float64) + M_star

    W_star = W_star.to(torch.float32).cpu()
    torch.cuda.empty_cache()
    return GhostedLayer(W_star)


# ── Ghosted Layers: contiguous (single boundary) ─────────────────────────
def compute_ghosted_layer(teacher, pmodel, start_l, end_l, trainloader, device,
                          max_batches=128):
    teacher.eval()
    teacher.to(device)
    X_pre, X_post = _collect_boundary_activations(
        teacher, start_l, end_l, trainloader, device, max_batches)
    m_norm = (X_post.double() - X_pre.double()).norm().item()
    print("[GhostedLayer] ||Delta||_F = " + str(round(m_norm, 4)))
    return _compute_ghosted_from_activations(X_pre, X_post, device)


def register_ghosted_layer(model, start_l, ghosted_layer, dev):
    """Insert GhostedLayer as a forward hook on layer start_l-1."""
    def ghost_hook(module, input, output):
        if torch.is_tensor(output):
            return module.ghosted_layer(output)
        if isinstance(output, (tuple, list)):
            out = output[0]
            if isinstance(out, (tuple, list)):
                out = out[0]
            patched = module.ghosted_layer(out)
            if isinstance(output, tuple):
                return (patched,) + output[1:]
            output = list(output)
            output[0] = patched
            return output
        return output

    handles = []
    for i, layer in enumerate(model.model.layers):
        if i == start_l - 1:
            ghosted_layer = ghosted_layer.to(dev)
            setattr(layer, "ghosted_layer", ghosted_layer)
            handles.append(layer.register_forward_hook(ghost_hook))
            break
    return handles


# ── Ghosted Layers: non-contiguous (one per removed layer) ───────────────
def compute_ghosted_layer_multi(teacher, pruned_indices, trainloader, device,
                                max_batches=128):
    teacher.eval()
    teacher.to(device)
    ghosted_layers = []
    for pruned_idx in sorted(pruned_indices):
        X_pre, X_post = _collect_boundary_activations(
            teacher, pruned_idx, pruned_idx + 1, trainloader, device, max_batches)
        gl = _compute_ghosted_from_activations(X_pre, X_post, device)
        print("[GhostedLayer-multi] pruned_idx=" + str(pruned_idx))
        ghosted_layers.append(gl)
    return ghosted_layers


def register_ghosted_layer_multi(model, pruned_indices, ghosted_layers, dev):
    handles = []
    sorted_pairs = sorted(zip(pruned_indices, ghosted_layers), key=lambda x: x[0])
    for rank, (orig_idx, gl) in enumerate(sorted_pairs):
        new_idx = orig_idx - rank          # index after removing `rank` earlier layers
        insert_at = new_idx - 1

        def make_ghost_hook(ghosted):
            def ghost_hook(module, input, output):
                if torch.is_tensor(output):
                    return ghosted(output)
                if isinstance(output, (tuple, list)):
                    out = output[0]
                    if isinstance(out, (tuple, list)):
                        out = out[0]
                    patched = ghosted(out)
                    if isinstance(output, tuple):
                        return (patched,) + output[1:]
                    output = list(output)
                    output[0] = patched
                    return output
                return output
            return ghost_hook

        gl = gl.to(dev)
        if insert_at >= 0:
            layer = model.model.layers[insert_at]
            setattr(layer, "ghosted_layer_" + str(rank), gl)
            handles.append(layer.register_forward_hook(make_ghost_hook(gl)))
            print("[GhostedLayer-multi] orig=" + str(orig_idx) + " → insert_at=" + str(insert_at))
        else:
            def make_pre_hook(ghosted):
                def pre_hook(module, args):
                    hs = args[0]
                    if isinstance(hs, (tuple, list)):
                        hs = (ghosted(hs[0]),) + hs[1:]
                    else:
                        hs = ghosted(hs)
                    return (hs,)
                return pre_hook
            layer = model.model.layers[0]
            setattr(layer, "ghosted_layer_" + str(rank), gl)
            handles.append(layer.register_forward_pre_hook(make_pre_hook(gl)))
            print("[GhostedLayer-multi] orig=" + str(orig_idx) + " → pre_hook on layer 0")
    return handles


# ── LinearPatch: contiguous (single boundary) ────────────────────────────
def register_linear_patch(model, start_l, scale_param, dev, rotated=True,
                          weight=None, d_weight=None):
    def rotated_hook(module, input, output):
        if torch.is_tensor(output):
            return module.linear_patch(output)
        if isinstance(output, (tuple, list)):
            out = output[0]
            if isinstance(out, (tuple, list)):
                out = out[0]
            patched = module.linear_patch(out)
            if isinstance(output, tuple):
                return (patched,) + output[1:]
            output = list(output)
            output[0] = patched
            return output
        return output

    handles = []
    for i, layer in enumerate(model.model.layers):
        if i == start_l - 1:
            if scale_param is not None:
                linear_patch = LinearPatch(
                    diag=scale_param.to(torch.float32).to(dev), rotated=rotated,
                    weight=None,
                    d_weight=d_weight.to(torch.float32).to(dev) if d_weight is not None else None)
            else:
                linear_patch = LinearPatch(
                    diag=None, rotated=rotated, weight=weight.to(torch.float32).to(dev),
                    d_weight=d_weight.to(torch.float32).to(dev) if d_weight is not None else None)
            setattr(layer, "linear_patch", linear_patch)
            handles.append(layer.register_forward_hook(rotated_hook))
            break
    return handles


# ── LinearPatch: non-contiguous (one per removed layer) ──────────────────
def compute_linear_patch_multi(teacher, pruned_indices, trainloader, device,
                               rotated=True, max_batches=128):
    teacher.eval()
    teacher.to(device)
    linear_patches = []
    for pruned_idx in sorted(pruned_indices):
        X_pre, X_post = _collect_boundary_activations(
            teacher, pruned_idx, pruned_idx + 1, trainloader, device, max_batches)
        X_pre_dev = X_pre.to(device)
        X_post_dev = X_post.to(device)
        scale = (X_post_dev.abs().mean(dim=0) / (X_pre_dev.abs().mean(dim=0) + 1e-8))
        linear_patches.append(LinearPatch(diag=scale.to(torch.float32), rotated=rotated))
        print("[LinearPatch-multi] pruned_idx=" + str(pruned_idx))
        torch.cuda.empty_cache()
    return linear_patches


def register_linear_patch_multi(model, pruned_indices, linear_patches, dev):
    handles = []
    sorted_pairs = sorted(zip(pruned_indices, linear_patches), key=lambda x: x[0])
    for rank, (orig_idx, lp) in enumerate(sorted_pairs):
        new_idx = orig_idx - rank
        insert_at = new_idx - 1

        def make_lp_hook(patch):
            def lp_hook(module, input, output):
                if torch.is_tensor(output):
                    return patch(output)
                if isinstance(output, (tuple, list)):
                    out = output[0]
                    if isinstance(out, (tuple, list)):
                        out = out[0]
                    patched = patch(out)
                    if isinstance(output, tuple):
                        return (patched,) + output[1:]
                    output = list(output)
                    output[0] = patched
                    return output
                return output
            return lp_hook

        lp = lp.to(dev)
        if insert_at >= 0:
            layer = model.model.layers[insert_at]
            setattr(layer, "linear_patch_" + str(rank), lp)
            handles.append(layer.register_forward_hook(make_lp_hook(lp)))
            print("[LinearPatch-multi] orig=" + str(orig_idx) + " → insert_at=" + str(insert_at))
    return handles
