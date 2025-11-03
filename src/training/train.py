"""Main training function for zk0."""

import logging

from loguru import logger

from src.training.model_utils import log_param_status
from src.training.training_setup import setup_training_components, reset_learning_rate_scheduler
from src.training.training_loop import run_training_loop


def train(
    net=None,
    trainloader=None,
    epochs=None,
    batch_size=64,
    device=None,
    global_params=None,
    fedprox_mu=0.01,
    initial_lr=None,
    partition_id=None,
    round_num=None,
) -> dict[str, float]:
    """Train SmolVLA model using lerobot's training loop (reusing the provided model instance)."""
    logging.debug(
        f"Starting train for {epochs} epochs on device {device}, len(trainloader)={len(trainloader)}"
    )

    # Use the provided model (already updated with server parameters)
    policy = net
    policy.train()

    # Log pre-training param status
    log_param_status(policy, "pre-training")

    # Setup training components
    cfg, optimizer, lr_scheduler, grad_scaler, train_metrics, train_tracker = (
        setup_training_components(
            policy,
            trainloader,
            epochs,
            batch_size,
            device,
            initial_lr,
            partition_id,
        )
    )

    # CRITICAL FIX: Reset learning rate scheduler for federated learning rounds
    # This ensures each FL round starts with the correct initial_lr, not decayed values
    lr_scheduler = reset_learning_rate_scheduler(
        optimizer, lr_scheduler, initial_lr, epochs
    )

    # Run training loop
    final_step = run_training_loop(
        policy,
        trainloader,
        epochs,
        device,
        cfg,
        optimizer,
        lr_scheduler,
        grad_scaler,
        train_metrics,
        train_tracker,
        global_params,
        fedprox_mu,
        partition_id,
        round_num,
    )

    # Log post-training param status
    log_param_status(policy, "post-training")

    # Log final metrics
    total_final = train_metrics["policy_loss"].avg + train_metrics["fedprox_loss"].avg
    logger.info(
        f"Train end: Final metrics - policy_loss={train_metrics['policy_loss'].avg:.4f}, fedprox_loss={train_metrics['fedprox_loss'].avg:.4f}, total_loss={total_final:.4f}, grad_norm={train_metrics['grad_norm'].avg:.4f}, lr={train_metrics['lr'].avg:.4f}, steps_completed={final_step}"
    )

    # Collect final metrics for return
    # 'loss' is total (policy + fedprox) for Flower compatibility
    final_metrics = {
        "loss": train_metrics[
            "loss"
        ].avg,  # Total loss (policy + fedprox) for compatibility
        "policy_loss": train_metrics["policy_loss"].avg,
        "fedprox_loss": train_metrics["fedprox_loss"].avg,
        "grad_norm": train_metrics["grad_norm"].avg,
        "lr": train_metrics["lr"].avg,
        "update_s": train_metrics["update_s"].avg,
        "dataloading_s": train_metrics["dataloading_s"].avg,
        "steps_completed": final_step,
    }

    logging.info("End of client training")
    logging.debug(f"Completed train for {epochs} epochs, actual steps: {final_step}")

    return final_metrics