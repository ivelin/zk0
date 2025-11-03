"""Evaluation utilities for zk0."""

import torch
import logging


def test(
    net, device, batch_size=64, eval_batches: int = 0, dataset=None
) -> tuple[float, int, dict[str, float]]:
    """Evaluate SmolVLA model using server evaluation dataset."""
    # Assert dataset is provided (required for server evaluation)
    assert dataset is not None, "dataset parameter is required for test() function"

    # Convert device string to torch.device object if needed
    if isinstance(device, str):
        device = torch.device(device)

    # In SmolVLA terminology policy is the neural network
    policy = net
    policy = policy.to(device)
    policy.eval()

    logging.info("Evaluating on server dataset")

    # Use the provided dataset
    dataset_name = (
        getattr(dataset, "meta", {}).get("repo_id", "provided_dataset")
        if hasattr(dataset, "meta") and hasattr(dataset.meta, "get")
        else "provided_dataset"
    )
    logging.info(f"Using provided dataset: {dataset_name}")

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
    # Use eval_batches to control evaluation scope instead of episodes
    max_episodes = float("inf")  # No episode limit when using eval_batches
    logging.info(
        f"Server evaluation using {dataset_name} (eval_batches: {eval_batches})"
    )

    total_loss = 0.0
    total_samples = 0
    successful_batches = 0
    total_batches_processed = 0

    # Set evaluation limit based on eval_batches
    max_batches_for_eval = eval_batches if eval_batches > 0 else None
    if eval_batches == 0:
        logging.info(f"Full evaluation: processing all {max_episodes} episodes")
    else:
        logging.info(
            f"Limited evaluation: processing up to {max_batches_for_eval} batches"
        )

    # Evaluate batches, limiting to first N episodes
    current_episode = None
    for batch in eval_loader:
        total_batches_processed += 1

        # Check episode limit
        batch_episode = (
            int(batch["episode_index"][0].item()) if "episode_index" in batch else 0
        )
        if current_episode != batch_episode:
            current_episode = batch_episode
            episodes_processed += 1

        # Stop if we've processed enough episodes
        if episodes_processed > max_episodes:
            logging.info(f"Reached episode limit ({max_episodes}), stopping evaluation")
            break

        try:
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(
                        device, non_blocking=device.type == "cuda"
                    )

            # Compute policy loss (primary metric for SmolVLA flow-matching)
            with torch.no_grad():
                eval_loss, output_dict = policy.forward(batch)
                logging.debug(
                    f"DEBUG: policy.forward() returned eval_loss={eval_loss.item():.6f}, type={type(eval_loss)}, shape={eval_loss.shape if hasattr(eval_loss, 'shape') else 'scalar'}"
                )

                # For SmolVLA, policy loss is the primary evaluation metric
                target_actions = batch.get("action")
                batch_loss = eval_loss
                total_loss += batch_loss.item()
                total_samples += (
                    len(target_actions) if target_actions is not None else batch_size
                )
                successful_batches += 1

                action_dim = (
                    target_actions.shape[-1]
                    if target_actions is not None and len(target_actions.shape) > 1
                    else 7
                )
                # Log batch-level stats
                logging.debug(
                    f"Batch {successful_batches}: policy_loss={batch_loss.item():.4f}, samples={total_samples}, action_dim={action_dim}"
                )

        except Exception as e:
            logging.error(
                f"Failed to evaluate batch {total_batches_processed}: {e} - this indicates a serious evaluation issue that needs fixing"
            )
            # Do not increment successful_batches for failed batches
            continue

        # Limit batches for quick mode (based on total batches processed, not just successful ones)
        if max_batches_for_eval and total_batches_processed >= max_batches_for_eval:
            logging.info(
                f"Quick mode: stopping evaluation after {total_batches_processed} batches (limit: {max_batches_for_eval})"
            )
            break

    if total_samples > 0:
        raw_policy_loss = (
            total_loss / successful_batches
        )  # Average policy loss per batch (matches client averaging)
        logging.info(
            f"Successfully evaluated {successful_batches} batches with {total_samples} total samples, policy_loss={raw_policy_loss:.4f} (action_dim={action_dim})"
        )
    else:
        logging.warning("No batches successfully evaluated")
        raw_policy_loss = 1.0
        action_dim = 7

    # Clear GPU cache after evaluation to prevent VRAM accumulation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logging.debug("Cleared GPU cache after evaluation")

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

    logging.info(
        f"Server evaluation: loss={raw_policy_loss:.4f}, policy_loss={raw_policy_loss:.4f}, successful_batches={successful_batches}, total_batches={total_batches_processed}, samples={total_samples}"
    )

    return float(loss), num_examples, metrics