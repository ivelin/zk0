"""Shared FL machinery: the action-expert parameter subset, hashing,
and checkpoint rewriting.

Federated updates cover the ACTION EXPERT only — the SmolVLM
backbone stays frozen (spec section 5.3, matching zk0's published
configuration). In the checkpoint's own parameter names the subset
is everything that is not the VLM backbone:

  model.vlm_with_expert.lm_expert.*   (~98.2M params — the expert)
  model.action_in_proj / action_out_proj / action_time_mlp_* /
  model.state_proj                    (~1.6M params — the action I/O)

~99.9M of the model's ~450M parameters; the backbone's ~350M never
move in a round.
"""
import hashlib
import shutil

import torch
from pathlib import Path

import numpy as np

BACKBONE_PREFIX = "model.vlm_with_expert.vlm."


def subset_keys(state_dict) -> list:
    """The federated subset's keys, in canonical (sorted) order."""
    return sorted(k for k in state_dict if not k.startswith(BACKBONE_PREFIX))


def get_subset(policy) -> list:
    """The policy's action-expert parameters as fp32 numpy arrays."""
    sd = policy.state_dict()
    # The checkpoint file stores fp32, but the policy loads parts of
    # the expert as bfloat16 in memory (129 of 155 subset tensors).
    # Federation transports and averages fp32 — the file's format —
    # so cast torch-side before numpy (numpy has no bf16).
    return [sd[k].detach().to(torch.float32).cpu().numpy()
            for k in subset_keys(sd)]


def set_subset(policy, arrays) -> None:
    """Load action-expert arrays back into the policy (in place)."""
    import torch

    sd = policy.state_dict()
    keys = subset_keys(sd)
    if len(arrays) != len(keys):
        raise ValueError(f"subset size mismatch: {len(arrays)} arrays "
                         f"for {len(keys)} keys")
    with torch.no_grad():
        for k, arr in zip(keys, arrays):
            sd[k].copy_(torch.from_numpy(np.asarray(arr, dtype=np.float32)))


def arrays_hash(arrays) -> str:
    """SHA-256 over the subset arrays in canonical order."""
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a, dtype=np.float32)
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def full_model_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_updated_checkpoint(policy, src_dir, out_dir) -> Path:
    """Write the policy's full state as a new checkpoint directory:
    the source checkpoint's config/processor files, with the model
    weights replaced by the policy's current state."""
    from safetensors.torch import save_file

    src, out = Path(src_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file() and f.name != "model.safetensors":
            shutil.copy2(f, out / f.name)
    sd = {k: v.detach().to(torch.float32).cpu().contiguous()
          for k, v in policy.state_dict().items()}
    save_file(sd, str(out / "model.safetensors"),
              metadata={"format": "pt"})
    return out



def subset_from_checkpoint(ckpt_dir) -> list:
    """The action-expert subset read straight from a checkpoint
    directory's model.safetensors (no policy instantiation)."""
    from safetensors import safe_open

    path = Path(ckpt_dir) / "model.safetensors"
    with safe_open(str(path), framework="numpy") as f:
        keys = subset_keys({k: None for k in f.keys()})
        return [np.asarray(f.get_tensor(k), dtype=np.float32)
                for k in keys]


def subset_hash_of_checkpoint(ckpt_dir) -> str:
    return arrays_hash(subset_from_checkpoint(ckpt_dir))
