"""Learning rate scheduler and adjustment utilities for zk0."""

from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
)

from loguru import logger


def compute_adjustment_factor(eval_losses):
    """Compute adjustment factor for joint mu/LR tuning based on evaluation loss trends.

    Used by server-side strategy to determine if mu and LR should be adjusted together.

    Args:
        eval_losses: List of recent server evaluation loss values

    Returns:
        float: Adjustment factor (0.8 for stall, 1.1 for divergence, 1.0 for stable)
    """
    if len(eval_losses) < 3:
        return 1.0  # No adjustment with insufficient data

    # Calculate improvement over the last 3 rounds
    recent_3 = eval_losses[-3:]
    improvement = (recent_3[0] - recent_3[-1]) / max(
        recent_3[0], 1e-8
    )  # Avoid division by zero

    # Check for divergence first (most severe condition)
    if improvement < -0.05:
        return 1.1  # Increase both mu and LR by 10%

    # Check for stalling (less than 1% improvement over 3 rounds)
    elif improvement < 0.01:
        return 0.8  # Reduce both mu and LR by 20%

    # Stable progress
    return 1.0  # No adjustment needed


def create_scheduler(optimizer, cfg, epochs):
    """Factory function for creating different types of learning rate schedulers.

    Args:
        optimizer: PyTorch optimizer with param_groups
        cfg: Configuration object with scheduler parameters
        epochs: Number of epochs for this round

    Returns:
        PyTorch learning rate scheduler or None
    """
    scheduler_type = cfg.get("scheduler_type", "cosine")
    eta_min = cfg.get("eta_min", optimizer.param_groups[0]["lr"] * 0.1)

    if scheduler_type == "cosine_warm_restarts":
        T_0 = cfg.get("cosine_warm_restarts_T_0", 15)
        T_mult = int(cfg.get("cosine_warm_restarts_T_mult", 2))  # Must be integer >= 1
        return CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min
        )
    elif scheduler_type == "reduce_on_plateau":
        patience = cfg.get("stall_patience", 5)
        return ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=patience, min_lr=eta_min
        )
    else:  # "cosine" default
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)


def compute_adaptive_lr_factor(client_history, cfg):
    """Compute adaptive learning rate boost factor for hard clients.

    Args:
        client_history: Dict with client training history (avg_loss, current_loss)
        cfg: Configuration object with adaptive LR parameters

    Returns:
        float: LR boost factor (1.0 for no boost, >1.0 for boost)
    """
    if not cfg.get("adaptive_lr_enabled", False) or not client_history:
        return 1.0

    avg_prior_loss = client_history.get("avg_loss", 1.0)
    current_loss = client_history.get("current_loss", avg_prior_loss)
    threshold = cfg.get("high_loss_multiplier", 2.0)

    if current_loss > avg_prior_loss * threshold:
        return cfg.get("lr_boost_factor", 1.15)
    return 1.0


def reset_scheduler_adaptive(
    optimizer, lr_scheduler, initial_lr, epochs, client_history, cfg
):
    """Reset learning rate scheduler with adaptive LR boosts for federated learning rounds.

    This function ensures each FL round starts with correct initial LR, with adaptive boosts
    for hard clients based on training history.

    Args:
        optimizer: PyTorch optimizer with param_groups
        lr_scheduler: PyTorch learning rate scheduler (or None)
        initial_lr: Base initial learning rate to reset to
        epochs: Number of epochs for this round
        client_history: Dict with client training history for adaptive boosts
        cfg: Configuration object with scheduler and adaptive LR parameters

    Returns:
        Updated lr_scheduler (may be recreated if needed)
    """
    adaptive_factor = compute_adaptive_lr_factor(client_history, cfg)
    adjusted_lr = initial_lr * adaptive_factor

    for group in optimizer.param_groups:
        group["lr"] = adjusted_lr

    if lr_scheduler is None:
        lr_scheduler = create_scheduler(optimizer, cfg, epochs)
    else:
        lr_scheduler = create_scheduler(
            optimizer, cfg, epochs
        )  # Recreate for type safety

    logger.info(
        f"Adaptive reset: LR={adjusted_lr:.6f} (factor={adaptive_factor}), scheduler={type(lr_scheduler).__name__}"
    )
    return lr_scheduler


def compute_joint_adjustment(
    eval_losses, current_mu, current_lr, min_lr=1e-5, max_lr=1e-3
):
    """Compute joint adjustment for both mu and LR based on evaluation loss trends.

    Used by server-side strategy to compute synchronized mu/LR adjustments.

    Args:
        eval_losses: List of recent server evaluation loss values
        current_mu: Current FedProx mu value
        current_lr: Current learning rate
        min_lr: Minimum allowed learning rate (default: 1e-5)
        max_lr: Maximum allowed learning rate (default: 1e-3)

    Returns:
        tuple: (new_mu, new_lr, adjustment_reason) where adjustment_reason explains the change
    """
    factor = compute_adjustment_factor(eval_losses)
    if factor == 1.0:
        return current_mu, current_lr, "convergence_progress"
    elif factor < 1.0:
        new_mu = current_mu * factor
        new_lr = max(current_lr * factor, min_lr)
        return new_mu, new_lr, "stall_detected"
    elif factor > 1.0:
        new_mu = current_mu * factor
        new_lr = min(current_lr * factor, max_lr)
        return new_mu, new_lr, "divergence_detected"
    else:
        return current_mu, current_lr, "unknown_factor"