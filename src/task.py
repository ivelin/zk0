"""zk0: A Flower / Hugging Face LeRobot app."""

import time
from collections import OrderedDict
from typing import Any

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR

from datasets.utils.logging import disable_progress_bar
from loguru import logger

from .utils import load_lerobot_dataset


def get_model(dataset_meta=None):
    """Load SmolVLA model using lerobot factory (like standalone train script)."""
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.factory import make_policy

    # Assert that dataset metadata is provided (from actual dataset)
    assert dataset_meta is not None, "Dataset metadata must be provided from an actual dataset"

    # Create SmolVLA config (like standalone script)
    cfg = SmolVLAConfig()
    cfg.pretrained_path = "lerobot/smolvla_base"

    # Use lerobot factory to create policy (like standalone train script)
    policy = make_policy(cfg=cfg, ds_meta=dataset_meta)

    # Log memory usage after model loading (for OOM debugging)
    if torch.cuda.is_available():
        allocated_gb = torch.cuda.memory_allocated() / 1e9
        reserved_gb = torch.cuda.memory_reserved() / 1e9
        logger.info(f"Model loading complete - VRAM allocated: {allocated_gb:.2f} GB, reserved: {reserved_gb:.2f} GB")

    return policy


def compute_param_norms(model, trainable_only=True):
    """Compute parameter norms for model (trainable only by default)."""
    trainable_params = [p for p in model.parameters() if not trainable_only or p.requires_grad]
    param_norms = [torch.norm(p) for p in trainable_params]
    total_norm = sum(param_norms)
    num_params = len(trainable_params)
    sum_squares = sum(n**2 for n in param_norms)
    return total_norm, num_params, sum_squares




def log_param_status(model, stage="pre"):
    """Log parameter norms and requires_grad status."""
    full_norm, full_num, full_sum_squares = compute_param_norms(model, trainable_only=False)
    train_norm, train_num, train_sum_squares = compute_param_norms(model, trainable_only=True)

    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in model.parameters())

    logger.info(f"{stage} params - Full: norm={full_norm:.4f}, num={full_num}, sum_squares={full_sum_squares:.4f}")
    logger.info(f"{stage} params - Trainable: norm={train_norm:.4f}, num={train_num}, sum_squares={train_sum_squares:.4f} ({trainable_count}/{total_count} params trainable)")
    logger.debug(f"DEBUG {stage} params - Full: norm={full_norm:.4f}, num={full_num}, sum_squares={full_sum_squares:.4f}")
    logger.debug(f"DEBUG {stage} params - Trainable: norm={train_norm:.4f}, num={train_num}, sum_squares={train_sum_squares:.4f} ({trainable_count}/{total_count} params trainable)")


def get_params(model):
    """Extract model parameters as numpy arrays to be sent to server."""
    # Log post-training norms before extraction
    log_param_status(model, "post-local training round: parameters prepared to be sent from client to server")
    
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
                logger.warning(f"More trainable params expected than provided for {name}")
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
    log_param_status(model, "pre-local training round: parameters sent from server to client")


def setup_training_components(policy, trainloader, epochs, batch_size, device, initial_lr, use_wandb=False, partition_id=None):
    """Setup training components: optimizer, scheduler, metrics, and configuration."""
    from lerobot.optim.factory import make_optimizer_and_scheduler
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.configs.default import WandBConfig  # + Import WandBConfig for logging enablement
    from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
    from torch.amp import GradScaler

    # Create client-specific WandB configuration to prevent metric overlap
    client_id = partition_id if partition_id is not None else "unknown"
    dataset_name = trainloader.dataset.meta.repo_id if hasattr(trainloader.dataset, 'meta') else "unknown"

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
        group['lr'] = initial_lr

    # Replace cosine with linear scheduler
    if lr_scheduler is not None:
        lr_scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=epochs)
        logger.info(f"FL scheduler: LinearLR with initial_lr={initial_lr}, decay to 0.5 over {epochs} steps")
    else:
        logger.warning(f"No scheduler found; using constant LR={initial_lr}")

    # Log optimizer details
    num_groups = len(optimizer.param_groups)
    total_opt_params = sum(len(group['params']) for group in optimizer.param_groups)
    logger.info(f"Optimizer: {type(optimizer).__name__}, {num_groups} groups, {total_opt_params} params optimized")
    for i, group in enumerate(optimizer.param_groups):
        logger.info(f"  Group {i}: {len(group['params'])} params, lr={group.get('lr')}, weight_decay={group.get('weight_decay')}")

    # Create gradient scaler
    grad_scaler = GradScaler(device.type if hasattr(device, 'type') else 'cuda', enabled=cfg.policy.use_amp)

    # Setup metrics tracking
    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    logger.info(f"Train start: Initial metrics setup - loss_avg={train_metrics['loss'].avg:.4f}, grad_norm_avg={train_metrics['grad_norm'].avg:.4f}, lr_avg={train_metrics['lr'].avg:.4f}")

    # Create MetricsTracker
    train_tracker = MetricsTracker(cfg.batch_size, 1000, 10, train_metrics, initial_step=0)

    return cfg, optimizer, lr_scheduler, grad_scaler, train_metrics, train_tracker


def run_training_step(step, policy, batch, device, train_metrics, train_tracker, optimizer, grad_scaler, lr_scheduler, cfg, global_params, fedprox_mu):
    """Run a single training step with FedProx regularization."""
    from contextlib import nullcontext

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
        logger.info(f"Step {step}: VRAM before update_policy - Allocated: {pre_update_allocated:.2f} GB, Reserved: {pre_update_reserved:.2f} GB, Free: {free_memory_gb:.2f} GB / {total_memory_gb:.2f} GB")

    # Manual replication of LeRobot's update_policy for FedProx integration
    start_time = time.perf_counter()  # For update_s metric

    policy.train()
    with torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        main_loss, output_dict = policy.forward(batch)  # Main loss from LeRobot forward

    # Compute proximal loss BEFORE backprop (FedProx core)
    proximal_loss = 0.0
    if global_params is not None:
        trainable_params = [p for p in policy.parameters() if p.requires_grad]
        for param, global_param in zip(trainable_params, global_params):
            global_param_tensor = torch.from_numpy(global_param).to(device)
            param_diff = torch.sum((param - global_param_tensor) ** 2)
            proximal_loss += param_diff
        proximal_loss = (fedprox_mu / 2.0) * proximal_loss
        logger.debug(f"Step {step}: FedProx proximal_loss={proximal_loss:.6f} (mu={fedprox_mu}, trainable_params={len(trainable_params)})")

    # Total loss for backprop: main_loss + proximal_loss
    total_loss = main_loss + proximal_loss
    logger.debug(f"Step {step}: FedProx total_loss={total_loss:.6f} (main={main_loss.item():.6f} + proximal={proximal_loss:.6f})")

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

    # Update metrics (as in LeRobot, but with total_loss)
    train_metrics["loss"].update(total_loss.item())
    train_metrics["grad_norm"].update(grad_norm.item())
    train_metrics["lr"].update(optimizer.param_groups[0]["lr"] if optimizer.param_groups else 0.0)
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
        logger.info(f"Step {step}: VRAM after update_policy - Allocated: {post_update_allocated:.2f} GB (delta: {delta_allocated:+.2f} GB), Reserved: {post_update_reserved:.2f} GB (delta: {delta_reserved:+.2f} GB), Free: {free_memory_gb:.2f} GB / {total_memory_gb:.2f} GB")

        # Clear cache after each step
        torch.cuda.empty_cache()
        cleared_allocated = torch.cuda.memory_allocated(device) / 1e9
        cleared_reserved = torch.cuda.memory_reserved(device) / 1e9
        cleared_free_gb = free_memory / 1e9
        logger.debug(f"Step {step}: VRAM after empty_cache - Allocated: {cleared_allocated:.2f} GB, Reserved: {cleared_reserved:.2f} GB, Free: {cleared_free_gb:.2f} GB / {total_memory_gb:.2f} GB")

    logger.debug(f"Step {step}: Training step completed successfully. Total loss: {total_loss.item():.6f}")
    return train_tracker, output_dict, main_loss.item(), proximal_loss.item()


def run_training_loop(policy, trainloader, epochs, device, cfg, optimizer, lr_scheduler, grad_scaler, train_metrics, train_tracker, global_params, fedprox_mu, use_wandb=False, partition_id=None, round_num=None):
    """Run the main training loop."""
    from lerobot.datasets.utils import cycle

    step = 0
    skipped_batches = 0
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
                batch_episode = int(batch['episode_index'][0].item()) if 'episode_index' in batch else -1
                if current_episode != batch_episode:
                    current_episode = batch_episode
                    if skipped_episodes > 0:
                        logger.warning(f"Resumed from new episode {current_episode} after {skipped_episodes} skipped episodes")
                    skipped_episodes = 0
            except StopIteration:
                # Dataloader exhausted - this shouldn't happen with cycle, but log if it does
                logger.error("Dataloader iterator exhausted unexpectedly - restarting cycle")
                dl_iter = cycle(trainloader)
                attempts += 1
                continue
            except Exception as e:
                attempts += 1
                batch_episode = int(batch['episode_index'][0].item()) if 'episode_index' in batch and batch else current_episode
                if "Invalid data found when processing input" in str(e) or "Could not push packet to decoder" in str(e):
                    logger.warning(f"Skipping corrupt batch in episode {batch_episode} due to decoding error: {e} (attempt {attempts}/{max_attempts}, total skipped batches this episode: {skipped_episodes})")
                    skipped_episodes += 1
                    if skipped_episodes >= 5:  # Skip entire episode after 5 consecutive bad batches
                        logger.warning(f"Skipping entire episode {batch_episode} due to excessive corruption (5+ bad batches)")
                        # Advance iterator to next episode (approximate by skipping attempts)
                        for _ in range(10):  # Arbitrary skip to next episode
                            try:
                                next(dl_iter)
                            except:
                                pass
                        current_episode = None
                        skipped_episodes = 0
                    continue
                else:
                    logger.error(f"Unexpected error in data loader for episode {batch_episode}: {e} (attempt {attempts}/{max_attempts})")
                    # Don't raise - continue trying
                    continue

        if batch is None:
            logger.warning(f"Could not fetch valid batch after {max_attempts} attempts in step {step}. Skipping step to avoid interruption.")
            # Log but continue to next step without incrementing
            train_metrics["dataloading_s"].update(time.perf_counter() - start_time)
            continue  # Skip this step gracefully

        train_metrics["dataloading_s"].update(time.perf_counter() - start_time)
        logger.debug(f"Step {step}: Batch fetched successfully from episode {int(batch['episode_index'][0].item()) if 'episode_index' in batch else 'unknown'}. Keys: {list(batch.keys())}, Sample shapes: {{k: v.shape if hasattr(v, 'shape') else type(v) for k,v in batch.items()}}")

        # Run training step
        try:
            train_tracker, output_dict, main_loss_val, proximal_loss_val = run_training_step(
                step, policy, batch, device, train_metrics, train_tracker, optimizer, grad_scaler,
                lr_scheduler, cfg, global_params, fedprox_mu
            )
        except Exception as step_error:
            logger.error(f"Error in training step {step}: {step_error}. Skipping step gracefully.")
            import traceback
            logger.error(traceback.format_exc())
            continue  # Skip this step without failing the round

        step += 1
        train_tracker.step()

        # Log progress every 10 steps
        if step % 10 == 0:
            logger.info(f"DIAG: Step {step}/{epochs} completed, loss_avg={train_metrics['loss'].avg:.4f}, time_elapsed={time.perf_counter() - loop_start_time:.2f}s, total_skipped_episodes={skipped_episodes}")

            # Log to WandB with client prefix
            if use_wandb and partition_id is not None:
                from src.wandb_utils import log_wandb_metrics
                client_prefix = f"client_{partition_id}"
                log_wandb_metrics({
                    f"{client_prefix}_federated_client_id": partition_id,
                    f"{client_prefix}_federated_round_number": round_num or 0,
                    f"{client_prefix}_training_step": step,
                    f"{client_prefix}_model_forward_loss": main_loss_val,
                    f"{client_prefix}_fedprox_regularization_loss": proximal_loss_val,
                    f"{client_prefix}_total_training_loss": train_metrics['loss'].avg,
                    f"{client_prefix}_learning_rate": train_metrics['lr'].avg,
                    f"{client_prefix}_gradient_norm": train_metrics['grad_norm'].avg,
                    f"{client_prefix}_steps_completed": step,
                    f"{client_prefix}_skipped_batches": skipped_batches,
                    f"{client_prefix}_skipped_episodes": skipped_episodes
                }, step=step)

        # Log progress (like lerobot train.py)
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        if is_log_step:
            import logging
            logging.info(train_tracker)
            logger.info(f"Step {step}: loss={train_metrics['loss'].avg:.4f}, grad_norm={train_metrics['grad_norm'].avg:.4f}, lr={train_metrics['lr'].avg:.4f}, update_s={train_metrics['update_s'].avg:.4f}")
            train_tracker.reset_averages()

    actual_steps = step
    logger.info(f"Training completed after {actual_steps} steps (target: {epochs}), total time: {time.perf_counter() - loop_start_time:.2f}s, skipped_episodes={skipped_episodes}")
    return actual_steps


def train(net=None, trainloader=None, epochs=None, batch_size=64, device=None, global_params=None, fedprox_mu=0.01, initial_lr=None, use_wandb=False, partition_id=None, round_num=None) -> dict[str, float]:
    """Train SmolVLA model using lerobot's training loop (reusing the provided model instance)."""
    import logging

    logging.debug(f"Starting train for {epochs} epochs on device {device}, len(trainloader)={len(trainloader)}")

    # Use the provided model (already updated with server parameters)
    policy = net
    policy.train()

    # Log pre-training param status
    log_param_status(policy, "pre-training")

    # Setup training components
    cfg, optimizer, lr_scheduler, grad_scaler, train_metrics, train_tracker = setup_training_components(
        policy, trainloader, epochs, batch_size, device, initial_lr, use_wandb, partition_id
    )

    # Run training loop
    final_step = run_training_loop(
        policy, trainloader, epochs, device, cfg, optimizer, lr_scheduler,
        grad_scaler, train_metrics, train_tracker, global_params, fedprox_mu,
        use_wandb, partition_id, round_num
    )

    # Log post-training param status
    log_param_status(policy, "post-training")

    # Log final metrics
    logger.info(f"Train end: Final metrics - loss={train_metrics['loss'].avg:.4f}, grad_norm={train_metrics['grad_norm'].avg:.4f}, lr={train_metrics['lr'].avg:.4f}, steps_completed={final_step}")

    # Collect final metrics for return
    final_metrics = {
        "loss": train_metrics["loss"].avg,
        "grad_norm": train_metrics["grad_norm"].avg,
        "lr": train_metrics["lr"].avg,
        "update_s": train_metrics["update_s"].avg,
        "dataloading_s": train_metrics["dataloading_s"].avg,
        "steps_completed": final_step,
    }

    logging.info("End of client training")
    logging.debug(f"Completed train for {epochs} epochs, actual steps: {final_step}")

    return final_metrics


def test(net, device, batch_size=64, eval_mode: str = "quick") -> tuple[float, int, dict[str, float]]:
    """Evaluate SmolVLA model using server evaluation dataset."""
    import logging
    from .utils import load_lerobot_dataset

    # Convert device string to torch.device object if needed
    if isinstance(device, str):
        device = torch.device(device)

    # In SmolVLA terminology policy is the neural network
    policy = net
    policy.eval()

    logging.info("Evaluating on server dataset")

    # Load server evaluation dataset
    from .configs import DatasetConfig
    config = DatasetConfig.load()

    if not config.server:
        raise ValueError("No server evaluation dataset configured")

    server_config = config.server[0]  # Use first server dataset
    dataset_name = server_config.name

    dataset = load_lerobot_dataset(dataset_name)

    # Create evaluation dataloader (will limit to first N episodes in the loop)
    eval_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
        batch_size=batch_size,
        shuffle=False,  # No shuffle for evaluation
        pin_memory=False,
        drop_last=False,
    )

    # Track episodes processed
    episodes_processed = 0
    max_episodes = server_config.first_n_episodes_for_eval
    logger.info(f"Server evaluation using first {max_episodes} episodes from {dataset_name} (mode: {eval_mode})")

    total_loss = 0.0
    total_samples = 0
    successful_batches = 0
    total_batches_processed = 0

    # Set evaluation limit based on mode
    max_batches_for_eval = 32 if eval_mode == "quick" else None
    if eval_mode == "full":
        logger.info(f"Full mode: processing all {max_episodes} episodes")
    else:
        logger.info(f"Mode {eval_mode}: limiting to {max_batches_for_eval} batches max")

    # Evaluate batches, limiting to first N episodes
    current_episode = None
    for batch in eval_loader:
        total_batches_processed += 1

        # Check episode limit
        batch_episode = int(batch['episode_index'][0].item()) if 'episode_index' in batch else 0
        if current_episode != batch_episode:
            current_episode = batch_episode
            episodes_processed += 1

        # Stop if we've processed enough episodes
        if episodes_processed > max_episodes:
            logger.info(f"Reached episode limit ({max_episodes}), stopping evaluation")
            break

        try:
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device, non_blocking=device.type == "cuda")

            # Compute policy loss (primary metric for SmolVLA flow-matching)
            with torch.no_grad():
                eval_loss, output_dict = policy.forward(batch)
                logger.debug(f"DEBUG: policy.forward() returned eval_loss={eval_loss.item():.6f}, type={type(eval_loss)}, shape={eval_loss.shape if hasattr(eval_loss, 'shape') else 'scalar'}")

                # For SmolVLA, policy loss is the primary evaluation metric
                # (MSE computation removed for standardization on policy_loss)
                target_actions = batch.get('action')
                batch_loss = eval_loss
                total_loss += batch_loss.item()
                total_samples += len(target_actions) if target_actions is not None else batch_size
                successful_batches += 1
            
                action_dim = target_actions.shape[-1] if target_actions is not None and len(target_actions.shape) > 1 else 7
                # Log batch-level stats
                logging.debug(f"Batch {successful_batches}: policy_loss={batch_loss.item():.4f}, samples={total_samples}, action_dim={action_dim}")

        except Exception as e:
            logging.error(f"Failed to evaluate batch {total_batches_processed}: {e} - this indicates a serious evaluation issue that needs fixing")
            # Do not increment successful_batches for failed batches
            continue

        # Limit batches for quick mode (based on total batches processed, not just successful ones)
        if max_batches_for_eval and total_batches_processed >= max_batches_for_eval:
            logging.info(f"Quick mode: stopping evaluation after {total_batches_processed} batches (limit: {max_batches_for_eval})")
            break

    if total_samples > 0:
        raw_policy_loss = total_loss / successful_batches  # Average policy loss per batch (matches client averaging)
        logging.info(f"Successfully evaluated {successful_batches} batches with {total_samples} total samples, policy_loss={raw_policy_loss:.4f} (action_dim={action_dim})")
    else:
        logging.warning("No batches successfully evaluated")
        raw_policy_loss = 1.0
        action_dim = 7

    # Clear GPU cache after evaluation to prevent VRAM accumulation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("Cleared GPU cache after evaluation")

    # Return metrics in Flower format: loss, num_examples, metrics_dict
    # For SmolVLA, policy loss is the primary evaluation metric (flow-matching loss)
    loss = raw_policy_loss
    num_examples = total_samples
    metrics = {
        "policy_loss": raw_policy_loss,  # Average policy forward loss per batch (primary metric for SmolVLA)
        "action_dim": action_dim,  # Number of action dimensions detected from batch (default 7 for SO-100 joints + gripper)
        "successful_batches": successful_batches,  # Number of batches successfully processed during evaluation
        "total_batches_processed": total_batches_processed,  # Total batches attempted (including failed)
        "total_samples": total_samples,  # Total number of action samples evaluated
    }

    logging.info(f"Server evaluation: loss={raw_policy_loss:.4f}, policy_loss={raw_policy_loss:.4f}, successful_batches={successful_batches}, total_batches={total_batches_processed}, samples={total_samples}")

    return float(loss), num_examples, metrics
