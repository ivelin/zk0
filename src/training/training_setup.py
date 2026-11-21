"""Training setup and configuration utilities for zk0."""

from torch.optim.lr_scheduler import (
    LinearLR,
    CosineAnnealingLR,
)

from loguru import logger


# Study 001 (D-11/G-6): the registered camera rename of the study
# driver's construction (front->camera1, top->camera2,
# wrist->camera3), used in the pooled-preprocessor factory call.
_STUDY_RENAME_MAP = {
    "observation.images.front": "observation.images.camera1",
    "observation.images.top": "observation.images.camera2",
    "observation.images.wrist": "observation.images.camera3",
}


def _construct_pooled_preprocessor(pooled_dir):
    """Reconstruct the D-9 pooled preprocessor from its serialized
    normalizer statistics, through the same factory call the study
    driver's build_pooled_preprocessor uses (same base config
    handling, same rename override), so the constructed pipeline is
    the gated object itself, not a serialization of it."""
    import os

    import torch
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from safetensors.torch import load_file

    stats_file = os.path.join(pooled_dir, "pooled_stats.safetensors")
    flat = load_file(stats_file)
    stats = {}
    for flat_key, tensor in flat.items():
        key, _, sub = flat_key.rpartition("|")
        stats.setdefault(key, {})[sub] = tensor

    pc = SmolVLAConfig.from_pretrained("lerobot/smolvla_base")
    pc.pretrained_path = "lerobot/smolvla_base"
    pc.empty_cameras = 0
    preprocessor, _postprocessor = make_pre_post_processors(
        policy_cfg=pc,
        dataset_stats=stats,
        overrides={
            "rename_observations_processor": {"rename_map": _STUDY_RENAME_MAP}
        },
    )
    return preprocessor



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
    )

    # Use lerobot's optimizer factory
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    # Override to use linear scheduler for FL partial rounds
    if initial_lr is None:
        initial_lr = 1e-3
    for group in optimizer.param_groups:
        group["lr"] = initial_lr

    # Replace cosine with CosineAnnealingLR for smoother decay

    if lr_scheduler is not None:
        eta_min = initial_lr * 0.1  # Decay to 10% of initial LR
        # Study 001 adaptation #7 (D-11/G-4): the registered schedule
        # holds the LR constant WITHIN a round (the server sends
        # lr_at at the mid-round offset per round). When the app
        # config carries lr_within_round = "constant", a flat cosine
        # (eta_min == initial_lr) realizes it with the stock
        # scheduler type.
        try:
            from src.common.utils import get_tool_config

            _app_cfg = (
                get_tool_config("flwr", "pyproject.toml")
                .get("app", {})
                .get("config", {})
            )
        except Exception:
            _app_cfg = {}
        if _app_cfg.get("lr_within_round") == "constant":
            eta_min = initial_lr
            logger.info(
                "FL scheduler: flat within-round LR (adaptation #7, lr_within_round=constant)"
            )
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


def build_batch_preprocessor(policy, dataset):
    """Build the canonical lerobot 0.7.0 SmolVLA batch preprocessor.

    Adaptation #3 (founder-approved 2026-11-21): era lerobot tokenized
    the language task inside its own training path; lerobot 0.7.0
    externalized batch preparation into the policy processor pipeline.
    SmolVLA.forward requires `observation.language.tokens` in the
    batch, which only that pipeline produces (alongside the rename,
    device, and normalize steps). This mirrors lerobot's own training
    pipeline (lerobot_train.py): processors built from the policy
    config plus the training dataset's own statistics, with the
    default (empty) rename map used throughout this codebase.

    Study 001 adaptation #9 (D-11/G-6): when the app config carries
    pooled_preprocessor_path, the ONE pooled union-250 preprocessor
    of deviation D-9 is constructed instead of building from this
    client's own dataset statistics - per-node stats would put the
    nodes in incommensurable normalization frames, which is exactly
    what D-9 forbids. The pooled preprocessor was built once by the
    study driver's own build_pooled_preprocessor construction; its
    normalizer statistics are serialized bit-exactly (float64
    safetensors, study_assets/pooled_preprocessor/pooled_stats.
    safetensors) and every client reconstructs the pipeline from
    them through the driver's own factory call, so the object each
    client trains with hashes (the driver's preprocessor_stats_
    hash) to the D-9 value a0455b5b... - lerobot's pipeline-level
    save_pretrained/from_pretrained round-trip was measured to
    degrade the stats to float32 (max diff 1.8e-6) and squeeze
    scalar shapes, so it is used only for checkpoint packaging
    (G-8), never for the training-time object.
    """
    pooled_path = None
    try:
        from src.common.utils import get_tool_config

        pooled_path = (
            get_tool_config("flwr", "pyproject.toml")
            .get("app", {})
            .get("config", {})
            .get("pooled_preprocessor_path")
        )
    except Exception:
        pooled_path = None
    if pooled_path:
        logger.info(
            f"Study 001 D-9: constructing pooled union-250 preprocessor from serialized stats at {pooled_path} (adaptation #9)"
        )
        return _construct_pooled_preprocessor(pooled_path)

    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.scripts.lerobot_train import rename_stats

    preprocessor, _postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        dataset_stats=rename_stats(dataset.meta.stats, {}),
    )
    return preprocessor
