"""Training setup and configuration utilities for zk0."""

from torch.optim.lr_scheduler import (
    LinearLR,
    CosineAnnealingLR,
)

from loguru import logger


def setup_training_components(
    policy,
    trainloader,
    epochs,
    batch_size,
    device,
    initial_lr,
    partition_id=None,
):
    """Setup training components: optimizer, scheduler, metrics, and configuration.

    Used by client-side training to initialize components.
    """
    from lerobot.optim.factory import make_optimizer_and_scheduler
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.configs.default import (
        WandBConfig,
    )  # + Import WandBConfig for logging enablement
    from lerobot.utils.logging_utils import MetricsTracker
    from torch.amp import GradScaler

    # Create client-specific WandB configuration to prevent metric overlap
    client_id = partition_id if partition_id is not None else "unknown"
    dataset_name = (
        trainloader.dataset.meta.repo_id
        if hasattr(trainloader.dataset, "meta")
        else "unknown"
    )

    # Create minimal config for lerobot factories (like standalone script)
    from lerobot.configs.default import DatasetConfig

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=trainloader.dataset.meta.repo_id),
        policy=policy.config,
        use_policy_training_preset=False,
        optimizer=policy.config.get_optimizer_preset(),
        scheduler=policy.config.get_scheduler_preset(),
        batch_size=batch_size,
        num_workers=0,
        log_freq=250,
        steps=epochs,
        wandb=WandBConfig(  # Disable WandB in LeRobot (we handle it at Flower level)
            project="zk0",
            enable=False,  # Always disable to prevent duplicate WandB runs
            mode="online",
            run_id=f"client_{client_id}_{dataset_name}",
            notes=f"Federated Learning Client {client_id} - Dataset: {dataset_name}",
        ),
    )

    # Use lerobot's optimizer factory
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    # Override to use linear scheduler for FL partial rounds
    if initial_lr is None:
        initial_lr = 1e-3
    for group in optimizer.param_groups:
        group["lr"] = initial_lr

    # Replace cosine with CosineAnnealingLR for smoother decay
    from torch.optim.lr_scheduler import CosineAnnealingLR

    if lr_scheduler is not None:
        eta_min = initial_lr * 0.1  # Decay to 10% of initial LR
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)
        logger.info(
            f"FL scheduler: CosineAnnealingLR with initial_lr={initial_lr}, eta_min={eta_min} over {epochs} steps"
        )
    else:
        logger.warning(f"No scheduler found; using constant LR={initial_lr}")

    # Log optimizer details
    num_groups = len(optimizer.param_groups)
    total_opt_params = sum(len(group["params"]) for group in optimizer.param_groups)
    logger.info(
        f"Optimizer: {type(optimizer).__name__}, {num_groups} groups, {total_opt_params} params optimized"
    )
    for i, group in enumerate(optimizer.param_groups):
        logger.info(
            f"  Group {i}: {len(group['params'])} params, lr={group.get('lr')}, weight_decay={group.get('weight_decay')}"
        )

    # Create gradient scaler
    grad_scaler = GradScaler(
        device.type if hasattr(device, "type") else "cuda", enabled=cfg.policy.use_amp
    )

    # Setup metrics tracking
    train_metrics = create_train_metrics()

    logger.info(
        f"Train start: Initial metrics setup - loss_avg={train_metrics['loss'].avg:.4f}, grad_norm_avg={train_metrics['grad_norm'].avg:.4f}, lr_avg={train_metrics['lr'].avg:.4f}"
    )

    # Create MetricsTracker
    train_tracker = MetricsTracker(
        cfg.batch_size, 1000, 10, train_metrics, initial_step=0
    )

    return cfg, optimizer, lr_scheduler, grad_scaler, train_metrics, train_tracker


def reset_learning_rate_scheduler(optimizer, lr_scheduler, initial_lr, epochs):
    """Reset learning rate scheduler for federated learning rounds.

    This function ensures that each federated learning round starts with the
    correct initial learning rate, preventing decay across rounds.

    Args:
        optimizer: PyTorch optimizer with param_groups
        lr_scheduler: PyTorch learning rate scheduler (or None)
        initial_lr: Initial learning rate to reset to
        epochs: Number of epochs for this round (for scheduler setup)

    Returns:
        Updated lr_scheduler (may be recreated if needed)
    """
    # Reset optimizer learning rates to initial value
    for group in optimizer.param_groups:
        group["lr"] = initial_lr

    # Reset or recreate scheduler to start fresh
    if lr_scheduler is not None:
        # Reset LinearLR scheduler to start from initial_lr again
        if hasattr(lr_scheduler, "start_factor"):
            # This is a LinearLR scheduler, reset it
            lr_scheduler = LinearLR(
                optimizer, start_factor=1.0, end_factor=0.5, total_iters=epochs
            )
        else:
            # For other scheduler types, try to reset last_epoch
            lr_scheduler.last_epoch = -1
        logger.info(
            f"FL scheduler reset: Set initial LR={initial_lr}, scheduler reset for {epochs} epochs"
        )
    else:
        logger.warning(f"No scheduler to reset; using constant LR={initial_lr}")

    return lr_scheduler


def create_train_metrics():
    """Create the train_metrics dictionary with all required AverageMeter instances."""
    from lerobot.utils.logging_utils import AverageMeter

    # 'loss' is total_loss (policy_loss + fedprox_loss) for Flower compatibility
    # 'policy_loss' is pure model forward loss
    # 'fedprox_loss' is FedProx regularization (separate for analysis)
    return {
        "loss": AverageMeter(
            "loss", ":.3f"
        ),  # Total loss (policy + fedprox) for compatibility
        "policy_loss": AverageMeter(
            "policy_loss", ":.3f"
        ),  # Pure SmolVLA flow-matching loss
        "fedprox_loss": AverageMeter("fedprox_loss", ":.3f"),  # FedProx proximal term
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }