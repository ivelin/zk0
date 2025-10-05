"""zk0: A Flower / Hugging Face LeRobot app."""

from pathlib import Path

import psutil

import torch
from src.task import (
    compute_param_norms,
    SmolVLATrainer,
)

from loguru import logger

from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Context

# Flower client
class SmolVLAClient(NumPyClient):
    def __init__(self, partition_id, local_epochs, trainloader, nn_device=None, use_wandb=False, wandb_group=None, batch_size=64) -> None:
        self.partition_id = partition_id
        self.trainloader = trainloader
        self.local_epochs = local_epochs
        self.device = nn_device

        # Load dataset metadata for model creation (like lerobot train script)
        # Get dataset metadata from the trainloader's dataset
        dataset_meta = trainloader.dataset.meta if hasattr(trainloader.dataset, 'meta') else None

        # Create SmolVLATrainer instance for training logic
        self.trainer = SmolVLATrainer(
            client_id=partition_id,
            device=nn_device,
            use_wandb=use_wandb,
            dataset_meta=dataset_meta,
            local_epochs=local_epochs,
            batch_size=batch_size  # Use the batch_size passed from server config
        )

        # Initialize WandB if enabled
        if use_wandb:
            from src.wandb_utils import init_wandb
            dataset_name = dataset_meta.repo_id if hasattr(dataset_meta, 'repo_id') else "unknown"
            if wandb_group:
                # Group under server's run
                self.wandb_run = init_wandb(
                    project="zk0",
                    name=f"client_{partition_id}",
                    group=wandb_group,
                    config={
                        "client_id": partition_id,
                        "dataset": dataset_name,
                        "local_epochs": local_epochs,
                        "batch_size": batch_size,
                    },
                    notes=f"Federated Learning Client {partition_id} - Dataset: {dataset_name}"
                )
            else:
                # Fallback: create separate run (legacy behavior)
                self.wandb_run = init_wandb(
                    project="zk0",
                    name=f"client_{partition_id}_{dataset_name}",
                    config={
                        "client_id": partition_id,
                        "dataset": dataset_name,
                        "local_epochs": local_epochs,
                        "batch_size": batch_size,
                    },
                    notes=f"Federated Learning Client {partition_id} - Dataset: {dataset_name}"
                )

        # Load model using trainer
        self.net = self.trainer.get_model()

        # Set the model and data in the trainer
        self.trainer.policy = self.net
        self.trainer.trainloader = trainloader

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
        dataset_repo_id = config.get("dataset_repo_id", "unknown")
        logger.info(f"Client {self.partition_id}: Starting fit operation (epochs={self.local_epochs}, batch_size={batch_size}, len(trainloader)={len(self.trainloader)})")
        logger.info(f"Client {self.partition_id}: Loading dataset '{dataset_repo_id}' for training")
        logger.debug(f"Client {self.partition_id}: Received config: {config}")

        if torch.cuda.is_available():
            vram_gb = torch.cuda.max_memory_allocated() / 1e9
            logger.bind(vram_gb=f"{vram_gb:.2f}").info("Fit start - VRAM allocated")
        ram_percent = psutil.virtual_memory().percent
        logger.bind(ram_percent=f"{ram_percent:.1f}").info("Fit start - Host RAM used")

        logger.debug(f"Client {self.partition_id}: Setting model parameters")

        # 🔐 VALIDATE: Client incoming parameters against server hash
        expected_hash = config.get("param_hash")
        if expected_hash:
            from src.utils import compute_parameter_hash
            received_hash = compute_parameter_hash(parameters)
            if received_hash != expected_hash:
                error_msg = f"Parameter hash mismatch! Expected: {expected_hash}, Received: {received_hash}"
                logger.error(f"❌ Client {self.partition_id}: {error_msg}")
                raise RuntimeError(error_msg)
            else:
                logger.info(f"✅ Client {self.partition_id}: Parameter hash validated: {received_hash[:8]}...")
        else:
            logger.warning(f"⚠️ Client {self.partition_id}: No param_hash provided by server, skipping validation")

        SmolVLATrainer.set_params(self.net, parameters)

        # FedProx: Extract global params for proximal term calculation (only trainable params)
        global_params = SmolVLATrainer.extract_trainable_params(self.net)

        # FedProx: Get proximal_mu from server config (default to 0.01 if not provided)
        proximal_mu = config.get("proximal_mu", 0.01)
        logger.info(f"Client {self.partition_id}: Using proximal_mu={proximal_mu} for FedProx regularization")

        # Get initial_lr from server config (default to 1e-3 if not provided)
        initial_lr = config.get("initial_lr", 1e-3)
        logger.info(f"Client {self.partition_id}: Using initial_lr={initial_lr} for training")

        # Get use_wandb from server config (default to False)
        use_wandb = config.get("use_wandb", False)
        logger.info(f"Client {self.partition_id}: Using use_wandb={use_wandb} for training")

        # Set round config in trainer
        self.trainer.set_round_config(
            round_num=config.get("round", 0),
            global_params=global_params,
            fedprox_mu=proximal_mu,
            initial_lr=initial_lr
        )
        # Note: use_wandb is already set during client initialization

        # Update batch_size in trainer if different from default
        self.trainer.batch_size = batch_size

        logger.info(f"Client {self.partition_id}: About to call trainer.train() with epochs={self.local_epochs}")
        try:
            training_metrics = self.trainer.train()
            logger.info(f"Client {self.partition_id}: trainer.train() returned successfully with metrics: {training_metrics}")
        except Exception as e:
            logger.error(f"Client {self.partition_id}: Exception in trainer.train(): {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise  # Re-raise to fail the client properly

        logger.info(f"Client {self.partition_id}: Training completed ({self.local_epochs} epochs, batch_size={batch_size})")

        logger.debug(f"Client {self.partition_id}: Extracting updated parameters")
        updated_params = SmolVLATrainer.get_params(self.net)

        # 🔐 VALIDATE: Client outgoing parameters (compute hash for server validation)
        from src.utils import compute_parameter_hash
        client_param_hash = compute_parameter_hash(updated_params)
        logger.info(f"✅ Client {self.partition_id}: Updated parameters hash: {client_param_hash[:8]}...")

        # Add hash to metrics for server validation
        training_metrics["param_hash"] = client_param_hash

        if torch.cuda.is_available():
            vram_gb = torch.cuda.max_memory_allocated() / 1e9
            logger.bind(vram_gb=f"{vram_gb:.2f}").info("Fit end - VRAM allocated")
        ram_percent = psutil.virtual_memory().percent
        logger.bind(ram_percent=f"{ram_percent:.1f}").info("Fit end - Host RAM used")

        logger.info(f"Client {self.partition_id}: Fit operation completed, returning {len(updated_params)} parameter arrays")
        return updated_params, len(self.trainloader), training_metrics

    def evaluate(self, parameters, config) -> tuple[float, int, dict[str, float]]:
        # Setup logging in the actor process
        from src.logger import setup_logging, setup_client_logging
        log_file_path = config.get("log_file_path")
        if log_file_path:
            setup_logging(Path(log_file_path), client_id=f"client_{self.partition_id}")
            setup_client_logging(Path(log_file_path), self.partition_id)

        dataset_repo_id = config.get("dataset_repo_id", "unknown")
        logger.info(f"Client {self.partition_id}: Starting evaluate operation")
        logger.info(f"Client {self.partition_id}: Loading dataset '{dataset_repo_id}' for evaluation")
        logger.debug(f"Client {self.partition_id}: Evaluate config: {config}")

        if torch.cuda.is_available():
            vram_gb = torch.cuda.max_memory_allocated() / 1e9
            logger.bind(vram_gb=f"{vram_gb:.2f}").info("Evaluate start - VRAM allocated")
        ram_percent = psutil.virtual_memory().percent
        logger.bind(ram_percent=f"{ram_percent:.1f}").info("Evaluate start - Host RAM used")

        logger.debug(f"Client {self.partition_id}: Setting evaluation parameters")

        # 🔐 VALIDATE: Client incoming parameters against server hash
        expected_hash = config.get("param_hash")
        if expected_hash:
            from src.utils import compute_parameter_hash
            received_hash = compute_parameter_hash(parameters)
            if received_hash != expected_hash:
                error_msg = f"Parameter hash mismatch! Expected: {expected_hash}, Received: {received_hash}"
                logger.error(f"❌ Client {self.partition_id}: {error_msg}")
                raise RuntimeError(error_msg)
            else:
                logger.info(f"✅ Client {self.partition_id}: Parameter hash validated: {received_hash[:8]}...")
        else:
            logger.warning(f"⚠️ Client {self.partition_id}: No param_hash provided by server, skipping validation")

        SmolVLATrainer.set_params(self.net, parameters)

        # Handle case where save_path might not be in config (evaluation rounds)
        try:
            # test() returns (loss, num_examples, metrics)
            eval_mode = config.get("eval_mode", "quick")
            eval_batch_size = config.get("eval_batch_size", config.get("batch_size", 64))
            loss, num_examples, metrics = SmolVLATrainer.test(
                partition_id=self.partition_id,
                net=self.net,
                device=self.device,
                eval_mode=eval_mode,
                batch_size=eval_batch_size
            )
            accuracy = metrics.get("action_mse", 0.0)  # Use action_mse as accuracy for Flower compatibility

            # Save client metrics to file
            try:
                import json
                from datetime import datetime

                # Use timestamp from config for consistent base path
                save_path = Path(config.get("save_path"))
                client_dir = save_path / "clients" / f"client_{self.partition_id}"
                client_dir.mkdir(parents=True, exist_ok=True)

                # Get round number from config (sent by server)
                if "round" not in config:
                    raise ValueError(f"Client {self.partition_id}: 'round' not found in config, cannot save evaluation file")
                round_num = config["round"]
                logger.info(f"Client {self.partition_id}: Using round number {round_num} for evaluation file")

                client_file = client_dir / f"round_{round_num}.json"
                data = {
                    "client_id": self.partition_id,
                    "round": round_num,
                    "timestamp": datetime.now().isoformat(),
                    "loss": float(loss),
                    "num_examples": num_examples,
                    "metrics": metrics
                }

                with open(client_file, 'w') as f:
                    json.dump(data, f, indent=2, default=str)

            except Exception as e:
                logger.info(f"Failed to save client metrics to file: {e}")
                # Continue execution even if file saving fails

        except Exception as e:
            logger.info(f"Client {self.partition_id} evaluation failed: {e}")
            # Return default values on evaluation failure
            loss = 1.0  # High loss indicates failure
            accuracy = 0.0
            num_examples = 0
            metrics = {"evaluation_error": str(e)}

        if torch.cuda.is_available():
            vram_gb = torch.cuda.max_memory_allocated() / 1e9
            logger.bind(vram_gb=f"{vram_gb:.2f}").info("Evaluate end - VRAM allocated")
        ram_percent = psutil.virtual_memory().percent
        logger.bind(ram_percent=f"{ram_percent:.1f}").info("Evaluate end - Host RAM used")

        testset_len = 1  # we test on one gym generated task
        post_eval_norm, _, _ = compute_param_norms(self.net)
        logger.info(f"Client {self.partition_id}: Evaluate completed - loss: {loss:.4f}, action_mse: {accuracy:.4f}, post-eval norm: {post_eval_norm:.4f}")
        return float(loss), testset_len, {"action_mse": accuracy, "partition_id": self.partition_id}


def client_fn(context: Context) -> Client:
    """Construct a Client that will be run in a ClientApp."""
    import logging

    # Load environment variables from .env file (for WANDB_API_KEY, etc.)
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
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    partition_id = context.node_config.get('partition-id', 'unknown')
    logging.info(f"🚀 Client function STARTED for partition {partition_id}")
    logging.debug(f"Client {partition_id}: Full context.node_config: {context.node_config}")
    logging.debug(f"Client {partition_id}: Full context.run_config: {context.run_config}")

    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    logging.info(f"✅ Client {partition_id}: Extracted partition_id={partition_id}, num_partitions={num_partitions}")

    # Discover device
    nn_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"✅ Client {partition_id}: Device set to {nn_device}")

    # Read the run config to get settings to configure the Client
    model_name = context.run_config["model-name"]
    local_epochs = int(context.run_config["local-epochs"])
    logging.info(f"✅ Client {partition_id}: Config loaded - model_name={model_name}, local_epochs={local_epochs}")

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
    use_wandb = context.run_config.get("use-wandb", False)
    wandb_group = context.run_config.get("wandb_group")
    logging.info(f"✅ Client {partition_id}: Batch size set to {batch_size}, use_wandb={use_wandb}, wandb_group={wandb_group}")

    # Load dataset first to get metadata for model creation
    logging.info(f"📊 Client {partition_id}: Loading dataset (partition_id={partition_id}, num_partitions={num_partitions})")
    try:
        trainloader, _ = SmolVLATrainer.load_data(partition_id, num_partitions, model_name, batch_size=batch_size, device=nn_device)
        total_episodes = len(trainloader.dataset)
        train_episodes = total_episodes - 3  # Exclude last 3 episodes for validation
        logging.info(f"✅ Client {partition_id}: Dataset loaded successfully - total episodes: {total_episodes}, training episodes: {train_episodes}, trainloader length: {len(trainloader)}")
    except Exception as e:
        logging.error(f"❌ Client {partition_id}: Failed to load dataset: {e}")
        import traceback
        logging.error(f"❌ Client {partition_id}: Dataset loading traceback: {traceback.format_exc()}")
        raise

    logging.info(f"🏗️ Client {partition_id}: Creating SmolVLAClient with {local_epochs} epochs")
    try:
        client = SmolVLAClient(
            partition_id=partition_id,
            local_epochs=local_epochs,
            trainloader=trainloader,
            nn_device=nn_device,
            use_wandb=use_wandb,
            wandb_group=wandb_group,
            batch_size=batch_size,
        )
        logging.info(f"✅ Client {partition_id}: SmolVLAClient created successfully")
        logging.info(f"🚀 Client {partition_id}: Converting to Flower client")
        flower_client = client.to_client()
        logging.info(f"✅ Client {partition_id}: Client initialization COMPLETE - returning to Flower")
        return flower_client
    except Exception as e:
        logging.error(f"❌ Client {partition_id}: Failed during client creation: {e}")
        import traceback
        logging.error(f"❌ Client {partition_id}: Client creation traceback: {traceback.format_exc()}")
        raise


app = ClientApp(client_fn=client_fn)