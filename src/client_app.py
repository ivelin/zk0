"""zk0: A Flower / Hugging Face LeRobot app."""

from pathlib import Path

import psutil

import torch
from src.task import (
    compute_param_norms,
    get_model,
    get_params,
    load_data,
    set_params,
    test,
    train,
)

from loguru import logger

from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Context

# Flower client
class SmolVLAClient(NumPyClient):
    def __init__(self, partition_id, local_epochs, trainloader, nn_device=None) -> None:
        self.partition_id = partition_id
        self.trainloader = trainloader
        self.local_epochs = local_epochs
        self.device = nn_device

        # Load dataset metadata for model creation (like lerobot train script)
        # Get dataset metadata from the trainloader's dataset
        dataset_meta = trainloader.dataset.meta if hasattr(trainloader.dataset, 'meta') else None
        self.net = get_model(dataset_meta)

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
        logger.info(f"Client {self.partition_id}: Starting fit operation (epochs={self.local_epochs}, batch_size={batch_size}, len(trainloader)={len(self.trainloader)})")
        logger.debug(f"Client {self.partition_id}: Received config: {config}")

        if torch.cuda.is_available():
            vram_gb = torch.cuda.max_memory_allocated() / 1e9
            logger.bind(vram_gb=f"{vram_gb:.2f}").info("Fit start - VRAM allocated")
        ram_percent = psutil.virtual_memory().percent
        logger.bind(ram_percent=f"{ram_percent:.1f}").info("Fit start - Host RAM used")

        logger.debug(f"Client {self.partition_id}: Setting model parameters")
        set_params(self.net, parameters)

        # FedProx: Extract global params for proximal term calculation (only trainable params)
        from src.task import extract_trainable_params
        global_params = extract_trainable_params(self.net)

        # FedProx: Get proximal_mu from server config (default to 0.01 if not provided)
        proximal_mu = config.get("proximal_mu", 0.01)
        logger.info(f"Client {self.partition_id}: Using proximal_mu={proximal_mu} for FedProx regularization")

        # Get initial_lr from server config (default to 1e-3 if not provided)
        initial_lr = config.get("initial_lr", 1e-3)
        logger.info(f"Client {self.partition_id}: Using initial_lr={initial_lr} for training")

        # + Get use_wandb from server config (default to False)
        use_wandb = config.get("use_wandb", False)
        logger.info(f"Client {self.partition_id}: Using use_wandb={use_wandb} for training")

        logger.info(f"Client {self.partition_id}: About to call train() with epochs={self.local_epochs}")
        try:
            training_metrics = train(
                net=self.net,
                trainloader=self.trainloader,
                epochs=self.local_epochs,
                device=self.device,
                batch_size=batch_size,
                global_params=global_params,
                fedprox_mu=proximal_mu,  # Use value from server config
                initial_lr=initial_lr,  # Use value from server config
                use_wandb=use_wandb  # + Pass use_wandb to enable/disable WandB logging
            )
            logger.info(f"Client {self.partition_id}: train() returned successfully with metrics: {training_metrics}")
        except Exception as e:
            logger.error(f"Client {self.partition_id}: Exception in train(): {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise  # Re-raise to fail the client properly

        logger.info(f"Client {self.partition_id}: Training completed ({self.local_epochs} epochs, batch_size={batch_size})")

        logger.debug(f"Client {self.partition_id}: Extracting updated parameters")
        updated_params = get_params(self.net)

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

        logger.info(f"Client {self.partition_id}: Starting evaluate operation")
        logger.debug(f"Client {self.partition_id}: Evaluate config: {config}")

        if torch.cuda.is_available():
            vram_gb = torch.cuda.max_memory_allocated() / 1e9
            logger.bind(vram_gb=f"{vram_gb:.2f}").info("Evaluate start - VRAM allocated")
        ram_percent = psutil.virtual_memory().percent
        logger.bind(ram_percent=f"{ram_percent:.1f}").info("Evaluate start - Host RAM used")

        logger.debug(f"Client {self.partition_id}: Setting evaluation parameters")
        set_params(self.net, parameters)

        # Handle case where save_path might not be in config (evaluation rounds)
        if config.get("skip", False):
            logger.info("Skipping evaluation")
            accuracy, loss = 0.0, 0.0
        else:
            try:
                # test() returns (loss, num_examples, metrics)
                eval_mode = config.get("eval_mode", "quick")
                eval_batch_size = config.get("eval_batch_size", config.get("batch_size", 64))
                loss, num_examples, metrics = test(
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
    logging.info(f"✅ Client {partition_id}: Batch size set to {batch_size}")

    # Load dataset first to get metadata for model creation
    logging.info(f"📊 Client {partition_id}: Loading dataset (partition_id={partition_id}, num_partitions={num_partitions})")
    try:
        trainloader, _ = load_data(partition_id, num_partitions, model_name, batch_size=batch_size, device=nn_device)
        logging.info(f"✅ Client {partition_id}: Dataset loaded successfully - trainloader length: {len(trainloader)}")
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