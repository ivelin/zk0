"""zk0: A Flower / Hugging Face LeRobot app."""

from pathlib import Path
import json
import os

import psutil

import torch
from src.task import (
    compute_param_norms,
    get_model,
    set_params,
    extract_trainable_params,
    get_params,
    test,
    train,
)


def compute_param_update_norm(pre_params, post_params):
    """Compute L2 norm of parameter differences between pre and post training.

    Args:
        pre_params: List of numpy arrays (pre-training parameters)
        post_params: List of numpy arrays (post-training parameters)

    Returns:
        float: L2 norm of parameter differences
    """
    if pre_params is None or post_params is None:
        return 0.0

    if len(pre_params) != len(post_params):
        return 0.0

    import numpy as np

    param_diff_norm = np.sqrt(
        sum(np.sum((post - pre) ** 2) for post, pre in zip(post_params, pre_params))
    )
    return float(param_diff_norm)


def save_client_round_metrics(
    config, training_metrics, round_num, partition_id, logger
):
    """Save per-round client metrics to JSON file.

    Args:
        config: Flower config dict containing timestamp
        training_metrics: Dict of training metrics
        round_num: Current round number
        partition_id: Client partition ID
        logger: Logger instance for logging

    Returns:
        None
    """
    try:
        from src.utils import create_client_metrics_dict

        timestamp = config.get("timestamp", "unknown")
        output_dir = f"outputs/{timestamp}/clients/client_{partition_id}"
        os.makedirs(output_dir, exist_ok=True)

        json_data = create_client_metrics_dict(
            round_num=round_num,
            client_id=partition_id,
            dataset_name=training_metrics.get("dataset_name", ""),
            policy_loss=training_metrics.get("policy_loss", 0.0),
            fedprox_loss=training_metrics.get("fedprox_loss", 0.0),
            grad_norm=training_metrics.get("grad_norm", 0.0),
            param_hash=training_metrics.get("param_hash", ""),
            num_steps=training_metrics.get("steps_completed", 0),
            param_update_norm=training_metrics.get("param_update_norm", 0.0),
        )
        with open(f"{output_dir}/round_{round_num}.json", "w") as f:
            json.dump(json_data, f, indent=2)
        logger.info(
            f"Client {partition_id}: Saved per-round metrics to {output_dir}/round_{round_num}.json"
        )
    except Exception as e:
        logger.warning(f"Client {partition_id}: Failed to save per-round metrics: {e}")


from src.utils import load_lerobot_dataset

from loguru import logger

from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Context


# Flower client
class SmolVLAClient(NumPyClient):
    def __init__(
        self,
        partition_id,
        local_epochs,
        trainloader,
        nn_device=None,
        batch_size=64,
        dataset_repo_id=None,
    ) -> None:
        self.partition_id = partition_id
        self.trainloader = trainloader
        self.local_epochs = local_epochs
        self.device = nn_device
        self.dataset_repo_id = (
            dataset_repo_id  # Cache dataset name to avoid repeated loading
        )

        # Validate required parameters
        assert self.dataset_repo_id is not None, (
            f"dataset_repo_id must be provided for client {self.partition_id}"
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
        policy.to(self.device)

        # Match standalone train script initialization (no extra cache clearing)

    def fit(self, parameters, config) -> tuple[list, int, dict]:
        # Setup logging in the actor process
        from src.logger import setup_logging, setup_client_logging

        log_file_path = config.get("log_file_path")
        if log_file_path:
            setup_logging(Path(log_file_path), client_id=f"client_{self.partition_id}")
            setup_client_logging(Path(log_file_path), self.partition_id)

        batch_size = config.get("batch_size", 64)
        logger.info(
            f"Client {self.partition_id}: Starting fit operation (epochs={self.local_epochs}, batch_size={batch_size}, len(trainloader)={len(self.trainloader)})"
        )
        logger.info(
            f"Client {self.partition_id}: Loading dataset '{self.dataset_repo_id}' for training"
        )
        logger.debug(f"Client {self.partition_id}: Received config: {config}")

        if torch.cuda.is_available():
            vram_gb = torch.cuda.max_memory_allocated() / 1e9
            logger.bind(vram_gb=f"{vram_gb:.2f}").info("Fit start - VRAM allocated")
        ram_percent = psutil.virtual_memory().percent
        logger.bind(ram_percent=f"{ram_percent:.1f}").info("Fit start - Host RAM used")

        logger.debug(f"Client {self.partition_id}: Setting model parameters")

        # 🔐 VALIDATE: Client incoming parameters against server hash - with detailed logging, hash before set_params
        expected_hash = config.get("param_hash")
        if expected_hash:
            from src.utils import compute_parameter_hash

            # Log received parameters details before hashing
            from flwr.common import parameters_to_ndarrays

            # Handle both Parameters object and list of ndarrays
            if isinstance(parameters, list):
                received_ndarrays = parameters
            else:
                received_ndarrays = parameters_to_ndarrays(parameters)
            logger.debug(
                f"Client {self.partition_id}: Received {len(received_ndarrays)} parameter arrays"
            )
            for i, ndarray in enumerate(
                received_ndarrays[:5]
            ):  # Log first 5 for brevity
                logger.debug(
                    f"  Received param {i}: shape={ndarray.shape}, dtype={ndarray.dtype}, min={ndarray.min():.4f}, max={ndarray.max():.4f}"
                )
            if len(received_ndarrays) > 5:
                logger.debug(f"  ... and {len(received_ndarrays) - 5} more parameters")

            # Compute hash on received ndarrays directly (before any model modification)
            received_hash = compute_parameter_hash(received_ndarrays)
            logger.debug(
                f"Client {self.partition_id}: Computed hash on raw received ndarrays: {received_hash}"
            )

            if received_hash != expected_hash:
                error_msg = f"Parameter hash mismatch! Expected: {expected_hash}, Received: {received_hash}"
                logger.error(f"❌ Client {self.partition_id}: {error_msg}")
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
                    f"✅ Client {self.partition_id}: Parameter hash validated: {received_hash[:8]}... (matches server expected)"
                )
                # Now safe to load into model
                set_params(self.net, parameters)
        else:
            logger.warning(
                f"⚠️ Client {self.partition_id}: No param_hash provided by server, skipping validation"
            )
            set_params(self.net, parameters)

        # Log pre-training norms (separate trainable vs frozen)
        from src.task import compute_param_norms

        full_norm, full_num, _ = compute_param_norms(self.net, trainable_only=False)
        train_norm, train_num, _ = compute_param_norms(self.net, trainable_only=True)
        total_params = sum(p.numel() for p in self.net.parameters())
        trainable_params = sum(
            p.numel() for p in self.net.parameters() if p.requires_grad
        )
        logger.info(
            f"Client {self.partition_id} R{self.round_num} PRE-TRAIN: Full norm={full_norm:.4f} ({full_num} tensors, {total_params} elems), Trainable norm={train_norm:.4f} ({train_num} tensors, {trainable_params} elems)"
        )

        # FedProx: Extract global params for proximal term calculation (only trainable params)
        global_params = extract_trainable_params(self.net)

        # FedProx: Get proximal_mu from server config (default to 0.01 if not provided)
        proximal_mu = config.get("proximal_mu", 0.01)
        logger.info(
            f"Client {self.partition_id}: Using proximal_mu={proximal_mu} for FedProx regularization"
        )

        # Get initial_lr from server config (default to 1e-3 if not provided)
        initial_lr = config.get("initial_lr", 1e-3)
        logger.info(
            f"Client {self.partition_id}: Using initial_lr={initial_lr} for training"
        )

        # Client-side WandB removed - clients get recycled between rounds

        # Set round config
        self.round_num = config.get("round", 0)
        self.global_params = global_params
        self.fedprox_mu = proximal_mu
        self.initial_lr = initial_lr

        logger.info(
            f"Client {self.partition_id}: About to call train() with epochs={self.local_epochs}"
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
                partition_id=self.partition_id,
                round_num=self.round_num,
            )
            logger.info(
                f"Client {self.partition_id}: train() returned successfully with metrics: {training_metrics}"
            )
        except Exception as e:
            logger.error(f"Client {self.partition_id}: Exception in train(): {e}")
            import traceback

            logger.error(traceback.format_exc())
            raise  # Re-raise to fail the client properly

        # Add client_id to metrics for server aggregation
        training_metrics["client_id"] = self.partition_id

        logger.info(
            f"Client {self.partition_id}: Training completed ({self.local_epochs} epochs, batch_size={batch_size})"
        )

        logger.debug(f"Client {self.partition_id}: Extracting updated parameters")
        updated_params = get_params(self.net)

        # Log post-training norms (separate trainable vs frozen)
        full_norm, full_num, _ = compute_param_norms(self.net, trainable_only=False)
        train_norm, train_num, _ = compute_param_norms(self.net, trainable_only=True)
        total_params = sum(p.numel() for p in self.net.parameters())
        trainable_params = sum(
            p.numel() for p in self.net.parameters() if p.requires_grad
        )
        logger.info(
            f"Client {self.partition_id} R{self.round_num} POST-TRAIN: Full norm={full_norm:.4f} ({full_num} tensors, {total_params} elems), Trainable norm={train_norm:.4f} ({train_num} tensors, {trainable_params} elems)"
        )

        # 🔐 VALIDATE: Client outgoing parameters (compute hash for server validation) - with detailed logging
        from src.utils import compute_parameter_hash

        # Log extracted params details before hashing
        logger.debug(
            f"Client {self.partition_id}: Extracted {len(updated_params)} parameter arrays for upload"
        )
        for i, ndarray in enumerate(updated_params[:5]):  # Log first 5
            logger.debug(
                f"  Upload param {i}: shape={ndarray.shape}, dtype={ndarray.dtype}, min={ndarray.min():.4f}, max={ndarray.max():.4f}"
            )
        if len(updated_params) > 5:
            logger.debug(f"  ... and {len(updated_params) - 5} more parameters")

        client_param_hash = compute_parameter_hash(updated_params)
        logger.debug(
            f"Client {self.partition_id}: Computed hash on extracted params: {client_param_hash}"
        )
        logger.info(
            f"✅ Client {self.partition_id}: Updated parameters hash: {client_param_hash[:8]}..."
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
                f"Client {self.partition_id}: Computed param_update_norm={param_update_norm:.6f}"
            )

        # Add param_update_norm to training_metrics for server aggregation
        training_metrics["param_update_norm"] = param_update_norm

        # Add param_hash and dataset_name to training_metrics for server aggregation
        training_metrics["param_hash"] = client_param_hash
        training_metrics["dataset_name"] = self.dataset_repo_id

        # Save per-round client metrics to JSON
        save_client_round_metrics(
            config, training_metrics, self.round_num, self.partition_id, logger
        )

        # Add hash to metrics for server validation (already added above for JSON)
        # Add dataset name to metrics for server aggregation (already added above for JSON)

        if torch.cuda.is_available():
            vram_gb = torch.cuda.max_memory_allocated() / 1e9
            logger.bind(vram_gb=f"{vram_gb:.2f}").info("Fit end - VRAM allocated")
        ram_percent = psutil.virtual_memory().percent
        logger.bind(ram_percent=f"{ram_percent:.1f}").info("Fit end - Host RAM used")

        logger.info(
            f"Client {self.partition_id}: Fit operation completed, returning {len(updated_params)} parameter arrays"
        )
        return updated_params, len(self.trainloader), training_metrics


def client_fn(context: Context) -> Client:
    """Construct a Client that will be run in a ClientApp."""
    import logging

    # Load environment variables from .env file (excluding WANDB_API_KEY for clients)
    try:
        from dotenv import load_dotenv

        load_dotenv()
        logging.debug("Environment variables loaded from .env file in client")
    except ImportError:
        logging.debug("python-dotenv not available in client, skipping .env loading")

    # Setup logging for client (DEBUG level for console and propagation)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Ensure console handler for DEBUG
    if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    partition_id = context.node_config.get("partition-id", "unknown")
    logging.info(f"🚀 Client function STARTED for partition {partition_id}")
    logging.debug(
        f"Client {partition_id}: Full context.node_config: {context.node_config}"
    )
    logging.debug(
        f"Client {partition_id}: Full context.run_config: {context.run_config}"
    )

    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    logging.info(
        f"✅ Client {partition_id}: Extracted partition_id={partition_id}, num_partitions={num_partitions}"
    )

    # Extract save_path for WandB dir isolation
    save_path = context.run_config.get("save_path")
    wandb_dir = f"{save_path}/clients/client_{partition_id}" if save_path else None
    logging.info(f"✅ Client {partition_id}: WandB dir set to {wandb_dir}")

    # Discover device
    nn_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"✅ Client {partition_id}: Device set to {nn_device}")

    # Read the run config to get settings to configure the Client
    model_name = context.run_config["model-name"]
    local_epochs = int(context.run_config["local-epochs"])
    logging.info(
        f"✅ Client {partition_id}: Config loaded - model_name={model_name}, local_epochs={local_epochs}"
    )

    # Setup client logging
    from src.logger import setup_logging, setup_client_logging

    log_file_path = context.run_config.get("log_file_path")
    if log_file_path:
        setup_logging(Path(log_file_path), client_id=f"client_{partition_id}")
        setup_client_logging(Path(log_file_path), partition_id)
        logger.info(f"✅ Client {partition_id}: Logging setup complete")
    else:
        logging.warning(f"⚠️ Client {partition_id}: No log_file_path provided in config")

    batch_size = context.run_config.get("batch_size", 64)
    logging.info(f"✅ Client {partition_id}: Batch size set to {batch_size}")

    # Load dataset first to get metadata for model creation
    logging.info(
        f"📊 Client {partition_id}: Loading dataset (partition_id={partition_id}, num_partitions={num_partitions})"
    )
    try:
        # Load dataset configuration
        from src.configs import DatasetConfig

        config = DatasetConfig.load()
        client_config = config.clients[partition_id % len(config.clients)]
        dataset_name = client_config.name

        # Load dataset directly
        dataset = load_lerobot_dataset(dataset_name)

        # Create dataloader (clients use full dataset for training)
        trainloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=0,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=False,
            drop_last=True,
        )

        train_episodes = len(dataset)
        logging.info(
            f"✅ Client {partition_id}: Dataset loaded successfully - training episodes: {train_episodes}, trainloader length: {len(trainloader)}"
        )
    except Exception as e:
        logging.error(f"❌ Client {partition_id}: Failed to load dataset: {e}")
        import traceback

        logging.error(
            f"❌ Client {partition_id}: Dataset loading traceback: {traceback.format_exc()}"
        )
        raise

    # Client-side WandB removed - clients get recycled between rounds
    wandb_run = None

    logging.info(
        f"🏗️ Client {partition_id}: Creating SmolVLAClient with {local_epochs} epochs"
    )
    try:
        client = SmolVLAClient(
            partition_id=partition_id,
            local_epochs=local_epochs,
            trainloader=trainloader,
            nn_device=nn_device,
            batch_size=batch_size,
            dataset_repo_id=dataset_name,
        )
        logging.info(f"✅ Client {partition_id}: SmolVLAClient created successfully")
        logging.info(f"🚀 Client {partition_id}: Converting to Flower client")
        flower_client = client.to_client()
        logging.info(
            f"✅ Client {partition_id}: Client initialization COMPLETE - returning to Flower"
        )
        return flower_client
    except Exception as e:
        logging.error(f"❌ Client {partition_id}: Failed during client creation: {e}")
        import traceback

        logging.error(
            f"❌ Client {partition_id}: Client creation traceback: {traceback.format_exc()}"
        )
        raise


app = ClientApp(client_fn=client_fn)
