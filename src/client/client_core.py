"""zk0: Core client functionality for SmolVLA federated learning."""

from pathlib import Path

import psutil

import torch
from src.training.model_utils import (
    compute_param_norms,
    get_model,
    set_params,
    extract_trainable_params,
    get_params,
)
from src.training.train import train

from src.common.utils import (
    compute_param_update_norm,
    save_client_round_metrics,
)

from loguru import logger

from flwr.client import NumPyClient



# Flower client
class SmolVLAClient(NumPyClient):
    def __init__(
        self,
        client_id,
        local_epochs,
        trainloader,
        nn_device=None,
        batch_size=64,
        dataset_repo_id=None,
        is_simulation=True,
    ) -> None:
        self.client_id = client_id
        self.trainloader = trainloader
        self.local_epochs = local_epochs
        self.device = nn_device
        self.dataset_name = dataset_repo_id  # Cache dataset name to avoid repeated loading
        self.is_simulation = is_simulation

        # Log CUDA availability on instantiation
        logger.info(
            f"Client {self.client_id}: Instantiated - CUDA available: {torch.cuda.is_available()}, using device: {self.device}"
        )

        # Validate required parameters
        assert self.dataset_name is not None, (
            f"dataset_repo_id must be provided for client {self.client_id}"
        )

        # Load dataset metadata for model creation (like lerobot train script)
        # Get dataset metadata from the trainloader's dataset
        dataset_meta = (
            trainloader.dataset.meta if hasattr(trainloader.dataset, "meta") else None
        )

        # Load model using global function
        self.net = get_model(dataset_meta)

        # Store data
        self.trainloader = trainloader

        # Round-specific state
        self.round_num = None
        self.global_params = None
        self.fedprox_mu = 0.01
        self.initial_lr = None

        # Dataset tracking
        self.dataset_id = dataset_repo_id  # For verification

        policy = self.net
        # SmolVLA uses flow matching, not diffusion, so no diffusion.num_inference_steps to set
        logger.info(f"Client {self.client_id}: Moving model to device {self.device}")
        policy.to(self.device)
        logger.info(
            f"Client {self.client_id}: Model moved to {self.device} - VRAM allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB"
        )

        # Match standalone train script initialization (no extra cache clearing)

    def _validate_and_set_parameters(self, parameters, config, logger):
        """Validate incoming parameters against server hash and set them on the model."""
        # 🔐 VALIDATE: Client incoming parameters against server hash - with detailed logging, hash before set_params
        expected_hash = config.get("param_hash")
        if expected_hash:
            from src.common.parameter_utils import compute_parameter_hash

            # Handle both Parameters object and list of ndarrays
            if isinstance(parameters, list):
                received_ndarrays = parameters
            else:
                from flwr.common import parameters_to_ndarrays
                received_ndarrays = parameters_to_ndarrays(parameters)
            logger.debug(f"Client {self.client_id}: Received {len(received_ndarrays)} parameter arrays")

            # Compute hash on received ndarrays directly (before any model modification)
            received_hash = compute_parameter_hash(received_ndarrays)
            logger.debug(
                f"Client {self.client_id}: Computed hash on raw received ndarrays: {received_hash}"
            )

            if received_hash != expected_hash:
                error_msg = f"Parameter hash mismatch! Expected: {expected_hash}, Received: {received_hash}"
                logger.error(f"❌ Client {self.client_id}: {error_msg}")
                logger.error(
                    f"  Server sent hash (pre-serialization?): {expected_hash}"
                )
                logger.error(f"  Client computed hash (raw ndarrays): {received_hash}")
                # Additional debug: Compare sample values (no model load needed)
                if len(received_ndarrays) > 0:
                    sample_param = received_ndarrays[0]
                    logger.error(
                        f"  Sample received param (first 10 elems): {sample_param.flatten()[:10]}"
                    )
                raise RuntimeError(error_msg)
            else:
                logger.info(
                    f"✅ Client {self.client_id}: Parameter hash validated: {received_hash[:8]}... (matches server expected)"
                )
                # Now safe to load into model
                set_params(self.net, parameters)
        else:
            logger.warning(
                f"⚠️ Client {self.client_id}: No param_hash provided by server, skipping validation"
            )
            set_params(self.net, parameters)

    def fit(self, parameters, config) -> tuple[list, int, dict]:
        client_id_from_config = config.get("partition-id", self.client_id)
        if client_id_from_config != self.client_id:
            logger.error(
                f"Client ID from client_fn mismatch with client ID in fit(): {client_id_from_config} != {self.client_id}"
            )

        # Setup logging in the actor process
        from src.logger import setup_client_logging

        # this is necessary due to the way flower / ray recycle actors in the same python process
        log_file_path = config.get("log_file_path")
        if log_file_path:
            setup_client_logging(
                Path(log_file_path), self.client_id
            )  # Now handles cleanup internally
            logger.info(
                f"Client {self.client_id}: Dynamic logging setup (client_id from config: {client_id_from_config})"
            )

        # Sync instance if mismatch (for safety, though rare post-client_fn)
        if client_id_from_config != self.client_id:
            logger.error(
                f"Client sync: Updating self.client_id from {self.client_id} to {client_id_from_config}"
            )
            self.client_id = client_id_from_config

        batch_size = config.get("batch_size", 64)
        logger.info(
            f"Client {self.client_id}: Starting fit operation (epochs={self.local_epochs}, batch_size={batch_size}, len(trainloader)={len(self.trainloader)})"
        )
        logger.info(
            f"Client {self.client_id}: Loading dataset '{self.dataset_name}' for training"
        )
        logger.debug(f"Client {self.client_id}: Received config: {config}")

        logger.debug(f"Client {self.client_id}: Starting fit operation")

        logger.debug(f"Client {self.client_id}: Setting model parameters")

        # Validate and set parameters
        self._validate_and_set_parameters(parameters, config, logger)

        # Log pre-training norms (separate trainable vs frozen)

        full_norm, full_num, _ = compute_param_norms(self.net, trainable_only=False)
        train_norm, train_num, _ = compute_param_norms(self.net, trainable_only=True)
        total_params = sum(p.numel() for p in self.net.parameters())
        trainable_params = sum(
            p.numel() for p in self.net.parameters() if p.requires_grad
        )
        logger.info(
            f"Client {self.client_id} R{self.round_num} PRE-TRAIN: Full norm={full_norm:.4f} ({full_num} tensors, {total_params} elems), Trainable norm={train_norm:.4f} ({train_num} tensors, {trainable_params} elems)"
        )

        # FedProx: Extract global params for proximal term calculation (only trainable params)
        global_params = extract_trainable_params(self.net)

        # FedProx: Get proximal_mu from server config (default to 0.01 if not provided)
        proximal_mu = config.get("proximal_mu", 0.01)
        logger.info(
            f"Client {self.client_id}: Using proximal_mu={proximal_mu} for FedProx regularization"
        )

        # Get initial_lr from server config (default to 1e-3 if not provided)
        initial_lr = config.get("initial_lr", 1e-3)
        logger.info(
            f"Client {self.client_id}: Using initial_lr={initial_lr} for training"
        )

        # Client-side WandB removed - clients get recycled between rounds

        # Set round config
        self.round_num = config.get("round", 0)
        self.global_params = global_params
        self.fedprox_mu = proximal_mu
        self.initial_lr = initial_lr

        # Log CUDA before training
        logger.info(
            f"Client {self.client_id}: Starting training - CUDA available: {torch.cuda.is_available()}, device: {self.device}"
        )

        logger.info(
            f"Client {self.client_id}: About to call train() with epochs={self.local_epochs}"
        )
        try:
            training_metrics = train(
                net=self.net,
                trainloader=self.trainloader,
                epochs=self.local_epochs,
                batch_size=batch_size,
                device=self.device,
                global_params=self.global_params,
                fedprox_mu=self.fedprox_mu,
                initial_lr=self.initial_lr,
                partition_id=self.client_id,
                round_num=self.round_num,
            )
            logger.info(
                f"Client {self.client_id}: train() returned successfully with metrics: {training_metrics}"
            )
        except Exception as e:
            logger.error(f"Client {self.client_id}: Exception in train(): {e}")
            import traceback

            logger.error(traceback.format_exc())
            raise  # Re-raise to fail the client properly

        # Add client_id to metrics for server aggregation
        training_metrics["client_id"] = self.client_id

        # Anonymize dataset name in production mode for privacy
        training_metrics["dataset_name"] = self.dataset_name if self.is_simulation else f"{self.dataset_name}-{self.client_id}"

        logger.info(
            f"Client {self.client_id}: Training completed ({self.local_epochs} epochs, batch_size={batch_size})"
        )

        logger.debug(f"Client {self.client_id}: Extracting updated parameters")
        updated_params = get_params(self.net)

        # Log post-training norms (separate trainable vs frozen)
        full_norm, full_num, _ = compute_param_norms(self.net, trainable_only=False)
        train_norm, train_num, _ = compute_param_norms(self.net, trainable_only=True)
        total_params = sum(p.numel() for p in self.net.parameters())
        trainable_params = sum(
            p.numel() for p in self.net.parameters() if p.requires_grad
        )
        logger.info(
            f"Client {self.client_id} R{self.round_num} POST-TRAIN: Full norm={full_norm:.4f} ({full_num} tensors, {total_params} elems), Trainable norm={train_norm:.4f} ({train_num} tensors, {trainable_params} elems)"
        )

        # 🔐 VALIDATE: Client outgoing parameters (compute hash for server validation) - with detailed logging

        logger.debug(f"Client {self.client_id}: Extracted {len(updated_params)} parameter arrays for upload")

        # 🔐 ADD: Include rounded parameter hash for drift-resistant validation
        # Use float32 precision for hash (matches transmission dtype, minimal overhead)
        from src.common.parameter_utils import compute_rounded_hash

        rounded_hash = compute_rounded_hash(updated_params, precision="float32")
        logger.info(
            f"✅ Client {self.client_id}: Updated parameters hash: {rounded_hash[:8]}..."
        )

        # Compute parameter update norm: L2 distance between pre-training and post-training parameters
        param_update_norm = 0.0
        if self.global_params is not None:
            # Get post-training trainable parameters
            post_train_params = extract_trainable_params(self.net)
            # Compute L2 norm of parameter differences
            param_update_norm = compute_param_update_norm(
                self.global_params, post_train_params
            )
            logger.info(
                f"Client {self.client_id}: Computed param_update_norm={param_update_norm:.6f}"
            )

        # Build full metrics for local logging (includes strings like param_hash)
        full_metrics = training_metrics.copy()
        full_metrics["param_update_norm"] = param_update_norm
        full_metrics["param_hash"] = rounded_hash  # String for validation
        # dataset_name already added above with anonymization

        # Strip to numerics only for Flower (avoids aggregation issues with strings)
        import numbers
        flower_metrics = {
            k: v for k, v in full_metrics.items()
            if isinstance(v, numbers.Number)
        }

        logger.info(f"Client {self.client_id}: Returning {len(flower_metrics)} numeric metrics to Flower, full {len(full_metrics)} for local")

        # Save full metrics locally
        save_client_round_metrics(
            config, full_metrics, self.round_num, self.client_id, logger
        )

        logger.debug(f"Client {self.client_id}: Fit operation completed")

        logger.info(
            f"Client {self.client_id}: Fit operation completed, returning {len(updated_params)} parameter arrays"
        )
        return updated_params, len(self.trainloader), flower_metrics