"""Model loading and parameter handling utilities for zk0."""

import torch
from collections import OrderedDict

from loguru import logger


def get_model(dataset_meta=None):
    """Load SmolVLA model using lerobot factory (like standalone train script)."""
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.factory import make_policy

    # Assert that dataset metadata is provided (from actual dataset)
    assert dataset_meta is not None, (
        "Dataset metadata must be provided from an actual dataset"
    )

    # Create SmolVLA config (like standalone script)
    cfg = SmolVLAConfig()
    cfg.pretrained_path = "lerobot/smolvla_base"

    # Use lerobot factory to create policy (like standalone train script)
    policy = make_policy(cfg=cfg, ds_meta=dataset_meta)

    # Log memory usage after model loading (for OOM debugging)
    if torch.cuda.is_available():
        allocated_gb = torch.cuda.memory_allocated() / 1e9
        reserved_gb = torch.cuda.memory_reserved() / 1e9
        logger.info(
            f"Model loading complete - VRAM allocated: {allocated_gb:.2f} GB, reserved: {reserved_gb:.2f} GB"
        )

    return policy


def compute_param_norms(model, trainable_only=True):
    """Compute parameter norms for model (trainable only by default)."""
    trainable_params = [
        p for p in model.parameters() if not trainable_only or p.requires_grad
    ]
    param_norms = [torch.norm(p) for p in trainable_params]
    total_norm = sum(param_norms)
    num_params = len(trainable_params)
    sum_squares = sum(n**2 for n in param_norms)
    return total_norm, num_params, sum_squares


def log_param_status(model, stage="pre"):
    """Log parameter norms and requires_grad status."""
    full_norm, full_num, full_sum_squares = compute_param_norms(
        model, trainable_only=False
    )
    train_norm, train_num, train_sum_squares = compute_param_norms(
        model, trainable_only=True
    )

    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in model.parameters())

    logger.info(
        f"{stage} params - Full: norm={full_norm:.4f}, num={full_num}, sum_squares={full_sum_squares:.4f}"
    )
    logger.info(
        f"{stage} params - Trainable: norm={train_norm:.4f}, num={train_num}, sum_squares={train_sum_squares:.4f} ({trainable_count}/{total_count} params trainable)"
    )
    logger.debug(
        f"DEBUG {stage} params - Full: norm={full_norm:.4f}, num={full_num}, sum_squares={full_sum_squares:.4f}"
    )
    logger.debug(
        f"DEBUG {stage} params - Trainable: norm={train_norm:.4f}, num={train_num}, sum_squares={train_sum_squares:.4f} ({trainable_count}/{total_count} params trainable)"
    )


def get_params(model):
    """Extract model parameters as numpy arrays to be sent to server."""
    # Log post-training norms before extraction
    log_param_status(
        model,
        "post-local training round: parameters prepared to be sent from client to server",
    )

    params = []
    for _, val in model.state_dict().items():
        # Convert BFloat16 and other unsupported dtypes to float32
        if val.dtype == torch.bfloat16:
            val = val.float()
        params.append(val.cpu().numpy())
    return params


def get_trainable_params(model):
    """Extract only trainable model parameters as numpy arrays."""
    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Convert BFloat16 to float32 for NumPy
            if param.dtype == torch.bfloat16:
                param_data = param.float().cpu().numpy()
            else:
                param_data = param.cpu().numpy()
            trainable_params.append(param_data)
    logger.info(f"Extracted {len(trainable_params)} trainable parameter tensors")
    return trainable_params


def set_trainable_params(model, trainable_parameters):
    """Set only trainable model parameters from numpy arrays, leaving frozen unchanged."""
    trainable_param_iter = iter(trainable_parameters)
    updated = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            try:
                param_data = next(trainable_param_iter)
                tensor = torch.from_numpy(param_data)
                # Restore original dtype (bfloat16 for efficiency)
                if param.dtype == torch.bfloat16:
                    tensor = tensor.bfloat16()
                param.data = tensor
                updated += 1
            except StopIteration:
                logger.warning(
                    f"More trainable params expected than provided for {name}"
                )
                break
    logger.info(f"Set {updated} trainable parameters; frozen VLM unchanged")


def extract_trainable_params(model) -> list:
    """Extract trainable parameters as numpy arrays for FedProx proximal term calculation."""
    trainable_params = []
    for name, val in model.state_dict().items():
        param = model.get_parameter(name)
        if param.requires_grad:  # Only include trainable params
            if val.dtype == torch.bfloat16:
                val = val.float()
            trainable_params.append(val.cpu().numpy())
    return trainable_params


def set_params(model, parameters) -> None:
    """Set model parameters from numpy arrays sent by server at the beginning of a round."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict()

    for k, v in params_dict:
        tensor = torch.from_numpy(v)
        # Convert back to the original dtype if it was BFloat16
        original_param = model.state_dict()[k]
        if original_param.dtype == torch.bfloat16:
            tensor = tensor.bfloat16()
        state_dict[k] = tensor

    model.load_state_dict(state_dict, strict=True)

    # Log after setting params (received from server)
    log_param_status(
        model, "pre-local training round: parameters sent from server to client"
    )