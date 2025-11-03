"""Training loop implementation for zk0."""

import time
import torch
from contextlib import nullcontext

from loguru import logger


def run_training_step(
    step,
    policy,
    batch,
    device,
    train_metrics,
    train_tracker,
    optimizer,
    grad_scaler,
    lr_scheduler,
    cfg,
    global_params,
    fedprox_mu,
):
    """Run a single training step with FedProx regularization."""
    from src.training.fedprox_utils import compute_fedprox_proximal_loss

    # Move batch to device
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device, non_blocking=device.type == "cuda")

    # Log VRAM before forward (matching LeRobot)
    if torch.cuda.is_available():
        pre_update_allocated = torch.cuda.memory_allocated(device) / 1e9
        pre_update_reserved = torch.cuda.memory_reserved(device) / 1e9
        free_memory, total_memory = torch.cuda.mem_get_info(device)
        free_memory_gb = free_memory / 1e9
        total_memory_gb = total_memory / 1e9
        logger.info(
            f"Step {step}: VRAM before update_policy - Allocated: {pre_update_allocated:.2f} GB, Reserved: {pre_update_reserved:.2f} GB, Free: {free_memory_gb:.2f} GB / {total_memory_gb:.2f} GB"
        )

    # Manual replication of LeRobot's update_policy for FedProx integration
    start_time = time.perf_counter()  # For update_s metric

    policy.train()
    with (
        torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext()
    ):
        main_loss, output_dict = policy.forward(batch)  # Main loss from LeRobot forward

    # Compute proximal loss BEFORE backprop (FedProx core)
    proximal_loss = torch.tensor(
        0.0, device=policy.parameters().__next__().device, dtype=torch.float32
    )
    if global_params is not None:
        trainable_params = [p for p in policy.parameters() if p.requires_grad]
        proximal_loss = compute_fedprox_proximal_loss(
            trainable_params, global_params, fedprox_mu
        )
        logger.debug(
            f"Step {step}: FedProx proximal_loss={proximal_loss.item():.6f} (mu={fedprox_mu}, trainable_params={len(trainable_params)})"
        )

    # Total loss for backprop: main_loss + proximal_loss (both tensors for consistent dtype)
    total_loss = main_loss + proximal_loss
    logger.debug(
        f"Step {step}: FedProx total_loss={total_loss.item():.6f} (main={main_loss.item():.6f} + proximal={proximal_loss.item():.6f})"
    )

    # Update separate metrics for policy and fedprox losses
    train_metrics["policy_loss"].update(main_loss.item())
    train_metrics["fedprox_loss"].update(proximal_loss.item())
    # Update 'loss' as total for compatibility (no separate total_loss meter)
    train_metrics["loss"].update(total_loss.item())

    # Backprop on total_loss (replicate LeRobot's flow)
    grad_scaler.scale(total_loss).backward()

    # Unscale gradients (as in LeRobot)
    grad_scaler.unscale_(optimizer)

    # Clip gradients (as in LeRobot)
    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(),
        max_norm=cfg.optimizer.grad_clip_norm,
        error_if_nonfinite=False,
    )

    # Step optimizer (as in LeRobot)
    grad_scaler.step(optimizer)
    grad_scaler.update()
    optimizer.zero_grad()

    # Step scheduler if present (as in LeRobot)
    if lr_scheduler is not None:
        lr_scheduler.step()

    # Update metrics (as in LeRobot, but with separate losses)
    # 'loss' is total (policy + fedprox) for compatibility
    train_metrics["loss"].update(total_loss.item())
    train_metrics["grad_norm"].update(grad_norm.item())
    train_metrics["lr"].update(
        optimizer.param_groups[0]["lr"] if optimizer.param_groups else 0.0
    )
    train_metrics["update_s"].update(time.perf_counter() - start_time)

    # Log VRAM after update
    if torch.cuda.is_available():
        post_update_allocated = torch.cuda.memory_allocated(device) / 1e9
        post_update_reserved = torch.cuda.memory_reserved(device) / 1e9
        delta_allocated = post_update_allocated - pre_update_allocated
        delta_reserved = post_update_reserved - pre_update_reserved
        free_memory, total_memory = torch.cuda.mem_get_info(device)
        free_memory_gb = free_memory / 1e9
        total_memory_gb = total_memory / 1e9
        logger.info(
            f"Step {step}: VRAM after update_policy - Allocated: {post_update_allocated:.2f} GB (delta: {delta_allocated:+.2f} GB), Reserved: {post_update_reserved:.2f} GB (delta: {delta_reserved:+.2f} GB), Free: {free_memory_gb:.2f} GB / {total_memory_gb:.2f} GB"
        )

        # Clear cache after each step
        torch.cuda.empty_cache()
        cleared_allocated = torch.cuda.memory_allocated(device) / 1e9
        cleared_reserved = torch.cuda.memory_reserved(device) / 1e9
        cleared_free_gb = free_memory / 1e9
        logger.debug(
            f"Step {step}: VRAM after empty_cache - Allocated: {cleared_allocated:.2f} GB, Reserved: {cleared_reserved:.2f} GB, Free: {cleared_free_gb:.2f} GB / {total_memory_gb:.2f} GB"
        )

    logger.debug(
        f"Step {step}: Training step completed successfully. Total loss: {total_loss.item():.6f}"
    )
    return train_tracker, output_dict, main_loss.item(), proximal_loss


def run_training_loop(
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
    partition_id=None,
    round_num=None,
):
    """Run the main training loop."""
    from lerobot.datasets.utils import cycle

    step = 0
    logger.info(f"Entering training loop: target steps={epochs}, initial step={step}")
    loop_start_time = time.perf_counter()

    # Use cycle iterator like standalone script for infinite looping
    dl_iter = cycle(trainloader)
    skipped_episodes = 0
    current_episode = None

    while step < epochs:
        start_time = time.perf_counter()
        batch = None
        max_attempts = 100  # Increased for more robustness against corrupt data
        attempts = 0
        while batch is None and attempts < max_attempts:
            try:
                batch = next(dl_iter)
                # Check for episode change to track skips
                if batch is not None and "episode_index" in batch:
                    batch_episode = int(batch["episode_index"][0].item())
                else:
                    batch_episode = -1
                if current_episode != batch_episode:
                    current_episode = batch_episode
                    if skipped_episodes > 0:
                        logger.warning(
                            f"Resumed from new episode {current_episode} after {skipped_episodes} skipped episodes"
                        )
                    skipped_episodes = 0
            except StopIteration:
                # Dataloader exhausted - this shouldn't happen with cycle, but log if it does
                logger.error(
                    "Dataloader iterator exhausted unexpectedly - restarting cycle"
                )
                dl_iter = cycle(trainloader)
                attempts += 1
                continue
            except Exception as e:
                attempts += 1
                # Fix: Check if batch is not None before accessing its keys
                if batch is not None and "episode_index" in batch:
                    batch_episode = int(batch["episode_index"][0].item())
                else:
                    batch_episode = current_episode
                if "Invalid data found when processing input" in str(
                    e
                ) or "Could not push packet to decoder" in str(e):
                    logger.warning(
                        f"Skipping corrupt batch in episode {batch_episode} due to decoding error: {e} (attempt {attempts}/{max_attempts}, total skipped batches this episode: {skipped_episodes})"
                    )
                    skipped_episodes += 1
                    if (
                        skipped_episodes >= 5
                    ):  # Skip entire episode after 5 consecutive bad batches
                        logger.warning(
                            f"Skipping entire episode {batch_episode} due to excessive corruption (5+ bad batches)"
                        )
                        # Advance iterator to next episode (approximate by skipping attempts)
                        for _ in range(10):  # Arbitrary skip to next episode
                            try:
                                next(dl_iter)
                            except Exception as e:
                                logger.warning(f"Failed to skip to next episode: {e}")
                        current_episode = None
                        skipped_episodes = 0
                    continue
                else:
                    logger.error(
                        f"Unexpected error in data loader for episode {batch_episode}: {e} (attempt {attempts}/{max_attempts})"
                    )
                    # Don't raise - continue trying
                    continue

        if batch is None:
            logger.warning(
                f"Could not fetch valid batch after {max_attempts} attempts in step {step}. Skipping step to avoid interruption."
            )
            # Log but continue to next step without incrementing
            train_metrics["dataloading_s"].update(time.perf_counter() - start_time)
            continue  # Skip this step gracefully

        train_metrics["dataloading_s"].update(time.perf_counter() - start_time)
        # Safe logging that handles None batch
        if batch is not None:
            episode_info = (
                int(batch["episode_index"][0].item())
                if "episode_index" in batch
                else "unknown"
            )
            keys_info = list(batch.keys())
            shapes_info = {
                k: v.shape if hasattr(v, "shape") else type(v) for k, v in batch.items()
            }
            logger.debug(
                f"Step {step}: Batch fetched successfully from episode {episode_info}. Keys: {keys_info}, Sample shapes: {shapes_info}"
            )
        else:
            logger.debug(f"Step {step}: Batch is None - skipping detailed logging")

        # Run training step
        try:
            train_tracker, output_dict, main_loss_val, proximal_loss_val = (
                run_training_step(
                    step,
                    policy,
                    batch,
                    device,
                    train_metrics,
                    train_tracker,
                    optimizer,
                    grad_scaler,
                    lr_scheduler,
                    cfg,
                    global_params,
                    fedprox_mu,
                )
            )
        except Exception as step_error:
            logger.error(
                f"Error in training step {step}: {step_error}. Skipping step gracefully."
            )
            import traceback

            logger.error(traceback.format_exc())
            continue  # Skip this step without failing the round

        step += 1
        train_tracker.step()

        # Log progress every 10 steps
        if step % 10 == 0:
            logger.info(
                f"DIAG: Step {step}/{epochs} completed, loss_avg={train_metrics['loss'].avg:.4f}, time_elapsed={time.perf_counter() - loop_start_time:.2f}s, total_skipped_episodes={skipped_episodes}"
            )

            # Client-side WandB logging removed - clients get recycled between rounds

        # Log progress (like lerobot train.py)
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        if is_log_step:
            import logging

            logging.info(train_tracker)
            logger.info(
                f"Step {step}: loss={train_metrics['loss'].avg:.4f}, grad_norm={train_metrics['grad_norm'].avg:.4f}, lr={train_metrics['lr'].avg:.4f}, update_s={train_metrics['update_s'].avg:.4f}"
            )
            train_tracker.reset_averages()

    actual_steps = step
    logger.info(
        f"Training completed after {actual_steps} steps (target: {epochs}), total time: {time.perf_counter() - loop_start_time:.2f}s, skipped_episodes={skipped_episodes}"
    )
    return actual_steps