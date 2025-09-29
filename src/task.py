"""zk0: A Flower / Hugging Face LeRobot app."""

import time
from collections import OrderedDict
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.utils.logging import disable_progress_bar
from loguru import logger

from .utils import load_lerobot_dataset

disable_progress_bar()

def load_data(
    partition_id: int, num_partitions: int, model_name: str, batch_size=64, device=None
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    """Load SO-100 data (training and eval)"""
    # Load dataset configuration
    from .configs import DatasetConfig
    config = DatasetConfig.load()

    # Get dataset name for this partition
    dataset_name = config.clients[partition_id % len(config.clients)].name
    dataset = load_lerobot_dataset(dataset_name)

    # Create dataloader for offline training.
    trainloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,  # Match standalone train script (no multiprocessing overhead)
        batch_size=batch_size,  # Fixed to 64 to match standalone for stable gradients
        shuffle=True,
        pin_memory=False,  # Match standalone train script (no VRAM pinning)
        drop_last=True,
    )

    # Log memory usage after dataset loading (for OOM debugging)
    if torch.cuda.is_available():
        allocated_gb = torch.cuda.memory_allocated() / 1e9
        reserved_gb = torch.cuda.memory_reserved() / 1e9
        logger.info(f"Dataset loading complete - VRAM allocated: {allocated_gb:.2f} GB, reserved: {reserved_gb:.2f} GB")

    # SmolVLA doesn't use gym evaluation like PushT
    # Return None for testloader to match Flower's expected interface
    return trainloader, None


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


def train(net=None, trainloader=None, epochs=None, batch_size=64, device=None) -> dict[str, float]:
    """Train SmolVLA model using lerobot's training loop (reusing the provided model instance)."""
    import logging

    logging.debug(f"Starting train for {epochs} epochs on device {device}, len(trainloader)={len(trainloader)}")

    # Use the provided model (already updated with server parameters)
    policy = net
    policy.train()

    # Log pre-training param status
    log_param_status(policy, "pre-training")

    # Log optimizer creation details
    from lerobot.optim.factory import make_optimizer_and_scheduler
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
    from torch.amp import GradScaler
    import logging

    # Create minimal config for lerobot factories (like standalone script)
    from lerobot.configs.default import DatasetConfig
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=trainloader.dataset.meta.repo_id),  # Use client's dataset for consistency
        policy=policy.config,  # Use the actual policy config
        use_policy_training_preset=False,  # Set presets manually for FL
        optimizer=policy.config.get_optimizer_preset(),  # Get SmolVLA's optimizer preset
        scheduler=policy.config.get_scheduler_preset(),  # Get SmolVLA's scheduler preset
        batch_size=batch_size,
        num_workers=0,  # Avoid multiprocessing in FL
        log_freq=250,  # Log every 250 steps
        steps=epochs,  # Set total steps
    )

    # Use lerobot's optimizer factory (like standalone script)
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    
    # Override scheduler for FL partial rounds: Reset to initial LR each round
    initial_lr = 1e-4  # Standard for SmolVLA finetuning
    for group in optimizer.param_groups:
        group['lr'] = initial_lr
    if lr_scheduler is not None:
        for group in lr_scheduler.optimizer.param_groups:
            group['lr'] = initial_lr
        if hasattr(lr_scheduler, 'last_epoch'):
            lr_scheduler.last_epoch = -1  # Reset to start of schedule
        logger.info(f"FL scheduler reset: Set initial LR={initial_lr}, last_epoch=-1 for partial round (overrode decay)")
        logger.debug(f"DEBUG FL reset: LR={initial_lr}, scheduler={type(lr_scheduler).__name__}, last_epoch={getattr(lr_scheduler, 'last_epoch', 'N/A')}")
    
    # Log optimizer details (post-override)
    num_groups = len(optimizer.param_groups)
    total_opt_params = sum(len(group['params']) for group in optimizer.param_groups)
    logger.info(f"Optimizer: {type(optimizer).__name__}, {num_groups} groups, {total_opt_params} params optimized")
    logger.debug(f"DEBUG Optimizer: {type(optimizer).__name__}, {num_groups} groups, {total_opt_params} params optimized")
    for i, group in enumerate(optimizer.param_groups):
        group_size = len(group['params'])
        logger.info(f"  Group {i}: {group_size} params, lr={group.get('lr', 'N/A')}, weight_decay={group.get('weight_decay', 'N/A')}")
        logger.debug(f"DEBUG Group {i}: {group_size} params, lr={group.get('lr', 'N/A')}, weight_decay={group.get('weight_decay', 'N/A')}")

    # Create gradient scaler (like standalone script)
    grad_scaler = GradScaler(device.type if hasattr(device, 'type') else 'cuda', enabled=cfg.policy.use_amp)

    # Setup metrics tracking (like lerobot train.py)
    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    # Log initial metrics setup (after train_metrics defined)
    logger.info(f"Train start: Initial metrics setup - loss_avg={train_metrics['loss'].avg:.4f}, grad_norm_avg={train_metrics['grad_norm'].avg:.4f}, lr_avg={train_metrics['lr'].avg:.4f}")

    try:
        logger.info("Creating MetricsTracker...")
        train_tracker = MetricsTracker(
            cfg.batch_size, 1000, 10, train_metrics, initial_step=0  # Dummy values for FL
        )
        logger.info("MetricsTracker created successfully.")
    except Exception as e:
        logger.error(f"Failed to create MetricsTracker: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

    # Training loop (exactly like standalone lerobot train script)
    from lerobot.scripts.train import update_policy
    from lerobot.datasets.utils import cycle
    step = 0
    logger.info(f"Entering training loop: target steps={epochs}, initial step={step}")
    loop_start_time = time.perf_counter()

    # Use cycle iterator like standalone script
    dl_iter = cycle(trainloader)
    for _ in range(step, epochs):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        train_tracker.dataloading_s = time.perf_counter() - start_time
        logger.debug(f"Step {step}: Batch fetched successfully. Keys: {list(batch.keys()) if isinstance(batch, dict) else 'Not a dict'}. Sample shapes: { {k: v.shape if hasattr(v, 'shape') else type(v) for k,v in batch.items()} if isinstance(batch, dict) else 'N/A' }")

        try:
            logger.debug(f"Step {step}: Moving batch to device...")
            # Move batch to device (like lerobot train.py)
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device, non_blocking=device.type == "cuda")
            logger.debug(f"Step {step}: Batch moved to device successfully.")
            if torch.cuda.is_available():
                pre_update_allocated = torch.cuda.memory_allocated(device) / 1e9
                pre_update_reserved = torch.cuda.memory_reserved(device) / 1e9
                free_memory, total_memory = torch.cuda.mem_get_info(device)
                free_memory_gb = free_memory / 1e9
                total_memory_gb = total_memory / 1e9
                logger.info(f"Step {step}: VRAM before update_policy - Allocated: {pre_update_allocated:.2f} GB, Reserved: {pre_update_reserved:.2f} GB, Free: {free_memory_gb:.2f} GB / {total_memory_gb:.2f} GB")
        except Exception as e:
            logger.error(f"Step {step}: Failed to move batch to device: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

        try:
            logger.debug(f"Step {step}: Calling update_policy (policy.training={policy.training})...")
            train_tracker, output_dict = update_policy(
                train_tracker,
                policy,  # Use the provided model (already has server parameters)
                batch,
                optimizer,
                grad_clip_norm=cfg.optimizer.grad_clip_norm,
                grad_scaler=grad_scaler,
                lr_scheduler=lr_scheduler,
                use_amp=cfg.policy.use_amp,
            )
            if torch.cuda.is_available():
                post_update_allocated = torch.cuda.memory_allocated(device) / 1e9
                post_update_reserved = torch.cuda.memory_reserved(device) / 1e9
                delta_allocated = post_update_allocated - pre_update_allocated
                delta_reserved = post_update_reserved - pre_update_reserved
                free_memory, total_memory = torch.cuda.mem_get_info(device)
                free_memory_gb = free_memory / 1e9
                total_memory_gb = total_memory / 1e9
                logger.info(f"Step {step}: VRAM after update_policy - Allocated: {post_update_allocated:.2f} GB (delta: {delta_allocated:+.2f} GB), Reserved: {post_update_reserved:.2f} GB (delta: {delta_reserved:+.2f} GB), Free: {free_memory_gb:.2f} GB / {total_memory_gb:.2f} GB")
                # Clear cache after each step to prevent accumulation
                torch.cuda.empty_cache()
                cleared_allocated = torch.cuda.memory_allocated(device) / 1e9
                cleared_reserved = torch.cuda.memory_reserved(device) / 1e9
                free_memory, total_memory = torch.cuda.mem_get_info(device)
                cleared_free_gb = free_memory / 1e9
                logger.debug(f"Step {step}: VRAM after empty_cache - Allocated: {cleared_allocated:.2f} GB, Reserved: {cleared_reserved:.2f} GB, Free: {cleared_free_gb:.2f} GB / {total_memory_gb:.2f} GB")
            logger.debug(f"Step {step}: update_policy completed successfully. Loss from output: {output_dict.get('loss', 'N/A')}")
        except Exception as e:
            logger.error(f"Step {step}: Failed in update_policy: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

        step += 1
        train_tracker.step()

        # Log progress every 10 steps for diagnosis (in addition to cfg.log_freq)
        if step % 10 == 0:
            logger.info(f"DIAG: Step {step}/{epochs} completed, loss_avg={train_metrics['loss'].avg:.4f}, time_elapsed={time.perf_counter() - loop_start_time:.2f}s")

        # Log progress (like lerobot train.py)
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        if is_log_step:
            logging.info(train_tracker)
            logger.info(f"Step {step}: loss={train_metrics['loss'].avg:.4f}, grad_norm={train_metrics['grad_norm'].avg:.4f}, lr={train_metrics['lr'].avg:.4f}, update_s={train_metrics['update_s'].avg:.4f}")
            train_tracker.reset_averages()

    logger.info(f"Training completed after {step} steps (target: {epochs}), total time: {time.perf_counter() - loop_start_time:.2f}s")

    # Log post-training param status
    log_param_status(policy, "post-training")

    # Log final train_tracker state
    logger.info(f"Train end: Final metrics - loss={train_metrics['loss'].avg:.4f}, grad_norm={train_metrics['grad_norm'].avg:.4f}, lr={train_metrics['lr'].avg:.4f}, steps_completed={step}")

    # Collect final metrics for return
    final_metrics = {
        "loss": train_metrics["loss"].avg,
        "grad_norm": train_metrics["grad_norm"].avg,
        "lr": train_metrics["lr"].avg,
        "update_s": train_metrics["update_s"].avg,
        "dataloading_s": train_metrics["dataloading_s"].avg,
        "steps_completed": step,  # Add actual steps for diagnosis
    }

    logging.info("End of client training")
    logging.debug(f"Completed train for {epochs} epochs, actual steps: {step}")

    return final_metrics


def test(partition_id: int, net, device, batch_size=64, eval_mode: str = "quick") -> tuple[float, int, dict[str, float]]:
    """Evaluate SmolVLA model using LeRobot's evaluation approach (aligned with standalone train script)."""
    import logging
    from .utils import load_lerobot_dataset

    # In SmolVLA terminology policy is the neural network
    policy = net
    policy.eval()

    logging.info(f"Evaluating client {partition_id} policy")

    # Load client's dataset for evaluation using LeRobot's approach
    from .configs import DatasetConfig
    config = DatasetConfig.load()
    dataset_name = config.clients[partition_id % len(config.clients)].name

    # Use the same dataset loading as train (which works) instead of make_dataset
    dataset = load_lerobot_dataset(dataset_name)

    # Get number of eval episodes based on mode
    config = DatasetConfig.load()
    client_config = config.clients[partition_id % len(config.clients)]
    last_n_episodes_for_eval = client_config.last_n_episodes_for_eval
    # Quick mode uses batch limit instead of episode limit

    # Filter to only eval episodes if episodes metadata is available
    logger.debug(f"Client {partition_id}: Dataset episodes type/len: {type(dataset.episodes)}, {len(dataset.episodes) if dataset.episodes is not None else 'None'}")
    if dataset.episodes is not None:
        total_episodes = len(dataset.episodes)
        if last_n_episodes_for_eval < total_episodes:
            episode_indices = list(range(total_episodes - last_n_episodes_for_eval, total_episodes))
            from lerobot.datasets.utils import FilteredLeRobotDataset
            dataset = FilteredLeRobotDataset(dataset, episode_indices)
            logging.info(f"Filtered dataset to last {last_n_episodes_for_eval} episodes for evaluation")
    else:
        logging.warning(f"Client {partition_id}: Episodes metadata is None (likely TorchCodec/FFmpeg missing). Falling back to batch-based evaluation without episode filtering.")

    # Create evaluation dataloader with proper batching (like standalone script)
    eval_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,  # Use same batch size as training for consistency
        shuffle=False,
        num_workers=0,  # Reduce memory overhead by avoiding multiprocessing
        pin_memory=False,  # Disable pin_memory to reduce VRAM pinning overhead
        drop_last=False,  # Don't drop last batch for evaluation
    )

    total_loss = 0.0
    total_samples = 0
    successful_batches = 0
    total_batches_processed = 0

    # Set evaluation limit based on mode
    max_batches_for_eval = 10 if eval_mode == "quick" else None

    # Evaluate all batches in the dataloader
    for batch in eval_loader:
        total_batches_processed += 1

        try:
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device, non_blocking=device.type == "cuda")

            # Simple forward pass for evaluation
            with torch.no_grad():
                outputs = policy(batch)
                if isinstance(outputs, tuple):
                    pred_actions = outputs[0] if len(outputs) > 0 else None
                else:
                    pred_actions = outputs.get('action', outputs.get('predicted_actions', None))
                target_actions = batch.get('action')

                if pred_actions is not None and target_actions is not None:
                    batch_loss = F.mse_loss(pred_actions, target_actions)
                    total_loss += batch_loss.item() * len(target_actions)
                    total_samples += len(target_actions)
                    successful_batches += 1
                    # Log batch-level stats
                    pred_mean = pred_actions.mean().item()
                    target_mean = target_actions.mean().item()
                    logging.debug(f"Batch {successful_batches}: MSE={batch_loss.item():.4f}, pred_mean={pred_mean:.4f}, target_mean={target_mean:.4f}, samples={len(target_actions)}")
                else:
                    logging.error(f"Batch {total_batches_processed}: Missing action keys for loss computation - this indicates a serious data or model issue that needs fixing")
                    # Do not increment successful_batches for failed batches
                    continue

        except Exception as e:
            logging.error(f"Failed to evaluate batch {total_batches_processed} for client {partition_id}: {e} - this indicates a serious evaluation issue that needs fixing")
            # Do not increment successful_batches for failed batches
            continue

        # Limit batches for quick mode (based on total batches processed, not just successful ones)
        if max_batches_for_eval and total_batches_processed >= max_batches_for_eval:
            logging.info(f"Quick mode: stopping evaluation after {total_batches_processed} batches (limit: {max_batches_for_eval})")
            break

    if total_samples > 0:
        avg_loss = total_loss / total_samples
        logging.info(f"Successfully evaluated {successful_batches} batches with {total_samples} total samples, avg_MSE={avg_loss:.4f}")
    else:
        logging.warning(f"No batches successfully evaluated for client {partition_id}")
        avg_loss = 1.0

    # Clear GPU cache after evaluation to prevent VRAM accumulation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("Cleared GPU cache after evaluation")

    # Return metrics in Flower format: loss, num_examples, metrics_dict
    # For SmolVLA, the primary metric is the flow matching loss (MSE)
    loss = avg_loss
    num_examples = total_samples
    metrics = {
        "action_mse": avg_loss,  # Primary SmolVLA evaluation metric (flow matching loss)
        "successful_batches": successful_batches,  # Number of batches successfully evaluated
        "total_batches_processed": total_batches_processed,  # Total number of batches processed
        "total_samples": total_samples,  # Total number of samples evaluated
    }

    logging.info(f"Client {partition_id} evaluation: loss={avg_loss:.4f}, successful_batches={successful_batches}, total_batches={total_batches_processed}, samples={total_samples}")

    return float(loss), num_examples, metrics