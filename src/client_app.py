"""zk0: A Flower / Hugging Face LeRobot app."""

from pathlib import Path

import torch
from src.core.utils import load_lerobot_dataset

from loguru import logger

from flwr.client import Client, ClientApp
from flwr.common import Context

from src.server_app import get_runtime_mode
from src.client_core import SmolVLAClient


def client_fn(context: Context) -> Client:
    """Construct a Client that will be run in a ClientApp."""

    # DEBUG: Log types and values before conversion for diagnosis
    raw_partition_id = context.node_config["partition-id"]
    logger.debug(
        f"DEBUG client_fn: raw_partition_id='{raw_partition_id}' (type: {type(raw_partition_id)}), context.node_id={context.node_id} (type: {type(context.node_id)})"
    )

    # Convert to int early for type consistency and assertion
    try:
        partition_id = int(raw_partition_id)
    except ValueError:
        logger.error(
            f"❌ client_fn: Invalid partition-id '{raw_partition_id}' - must be integer"
        )
        raise ValueError(f"Invalid partition-id in node_config: {raw_partition_id}")

    # Setup client logging once using partition_id
    from src.logger import setup_client_logging

    log_file_path = context.run_config.get("log_file_path")

    if log_file_path:
        setup_client_logging(Path(log_file_path), partition_id)
        logger.info(
            f"✅ Client {partition_id}: Logging setup complete (fixed CID for invariant client directories)"
        )
    else:
        logger.warning(f"⚠️ Client {partition_id}: No log_file_path provided in config")

    # Load environment variables from .env file (excluding WANDB_API_KEY for clients)
    try:
        from dotenv import load_dotenv

        load_dotenv()
        logger.debug("Environment variables loaded from .env file in client")
    except ImportError:
        logger.debug("python-dotenv not available in client, skipping .env loading")

    # DEBUG: Log types and values before conversion for diagnosis
    raw_partition_id = context.node_config["partition-id"]
    logger.debug(
        f"DEBUG client_fn: raw_partition_id='{raw_partition_id}' (type: {type(raw_partition_id)}), context.node_id={context.node_id} (type: {type(context.node_id)})"
    )

    # Convert to int early for type consistency and assertion
    try:
        partition_id = int(raw_partition_id)
    except ValueError:
        logger.error(
            f"❌ client_fn: Invalid partition-id '{raw_partition_id}' - must be integer"
        )
        raise ValueError(f"Invalid partition-id in node_config: {raw_partition_id}")

    # Determine runtime mode
    mode = get_runtime_mode(context)
    logger.info(f"Client {partition_id}: Running in {mode} mode")

    # Runtime mode guards and validation
    if mode == "simulation":
        logger.info("🧪 Simulation mode: Ensuring local-only execution")
        # Guard: Warn if external dataset config is provided (will be ignored)
        if context.run_config.get("dataset.repo_id") or context.run_config.get("dataset.root"):
            logger.warning("⚠️ Simulation mode: dataset.repo_id or dataset.root in run_config will be ignored; using DatasetConfig partitions instead")
    elif mode == "production":
        logger.info("🔒 Production mode: Configuring for external networking")
        # Guard: Ensure dataset is provided for production
        if not (context.run_config.get("dataset.repo_id") or context.run_config.get("dataset.root")):
            error_msg = "❌ Production mode requires dataset.repo_id or dataset.root in run_config"
            logger.error(error_msg)
            raise ValueError(error_msg)
    else:
        logger.warning(f"⚠️ Unknown mode '{mode}' - proceeding with defaults")

    logger.info(
        f"� client_fn: Client function STARTED with node_id={context.node_id} and partition {partition_id}"
    )
    logger.debug(
        f"Client {partition_id}: Full context.node_config: {context.node_config}"
    )
    logger.debug(
        f"Client {partition_id}: Full context.run_config: {context.run_config}"
    )

    # Read the node_config to fetch data partition associated to this node
    # (partition_id already converted above; reuse it)
    num_partitions = context.node_config["num-partitions"]
    logger.info(
        f"✅ Client {partition_id}: Extracted partition_id={partition_id}, num_partitions={num_partitions}"
    )

    # Extract save_path for WandB dir isolation
    save_path = context.run_config.get("save_path")
    wandb_dir = f"{save_path}/clients/client_{partition_id}" if save_path else None
    logger.info(f"✅ Client {partition_id}: WandB dir set to {wandb_dir}")

    # Discover device
    nn_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"✅ Client {partition_id}: Device set to {nn_device}")

    # Read the run config to get settings to configure the Client
    model_name = context.run_config["model-name"]
    local_epochs = int(context.run_config["local-epochs"])
    logger.info(
        f"✅ Client {partition_id}: Config loaded - model_name={model_name}, local_epochs={local_epochs}"
    )

    batch_size = context.run_config.get("batch_size", 64)
    logger.info(f"✅ Client {partition_id}: Batch size set to {batch_size}")

    # Determine client identifier for production mode (persistent CID for privacy)
    if mode == "production":
        client_id = context.cid
    else:
        client_id = partition_id

    # Load dataset first to get metadata for model creation
    try:
        if mode == "production":
            # Production mode: load from run_config (matches LeRobot CLI args)
            dataset_repo_id = context.run_config.get("dataset.repo_id")
            dataset_root = context.run_config.get("dataset.root")

            if dataset_repo_id:
                logger.info(f"Client {client_id}: Loading HF dataset: {dataset_repo_id}")
                dataset = load_lerobot_dataset(dataset_repo_id)
                dataset_name = dataset_repo_id
            elif dataset_root:
                logger.info(f"Client {client_id}: Loading local dataset: {dataset_root}")
                from lerobot.datasets.lerobot_dataset import LeRobotDataset
                dataset = LeRobotDataset(dataset_root)
                dataset_name = dataset_root
            else:
                raise ValueError("dataset.repo_id or dataset.root required in production mode run_config")

            logger.info(f"✅ Client {client_id}: Dataset loaded - episodes: {len(dataset)}")
        else:
            # Simulation mode: load from partitions
            logger.info(
                f"📊 Client {partition_id}: Loading dataset (partition_id={partition_id}, num_partitions={num_partitions})"
            )
            # Load dataset configuration
            from src.configs import DatasetConfig

            logger.info(f"🔍 Client {partition_id}: Loading DatasetConfig")
            config = DatasetConfig.load()
            client_config = config.clients[partition_id % len(config.clients)]
            dataset_name = client_config.name
            logger.info(f"🔍 Client {partition_id}: Selected dataset: {dataset_name}")

            # Load dataset directly
            logger.info(
                f"📥 Client {partition_id}: Calling load_lerobot_dataset({dataset_name})"
            )
            dataset = load_lerobot_dataset(dataset_name)
            logger.info(
                f"✅ Client {partition_id}: Dataset loaded - episodes: {len(dataset)}"
            )

        # Create dataloader (clients use full dataset for training) - common for both modes
        logger.info(
            f"🔄 Client {client_id}: Creating DataLoader (batch_size={batch_size}, num_workers=0)"
        )
        trainloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=0,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=False,
            drop_last=True,
        )
        logger.info(
            f"✅ Client {client_id}: DataLoader created - length: {len(trainloader)}"
        )

        train_episodes = len(dataset)
        logger.info(
            f"✅ Client {client_id}: Dataset loaded successfully - training episodes: {train_episodes}, trainloader length: {len(trainloader)}"
        )
    except Exception as e:
        logger.error(f"❌ Client {client_id}: Failed to load dataset: {e}")
        import traceback

        logger.error(
            f"❌ Client {client_id}: Dataset loading traceback: {traceback.format_exc()}"
        )
        raise

    # Client-side WandB removed - clients get recycled between rounds

    logger.info(
        f"🏗️ Client {client_id}: Creating SmolVLAClient with {local_epochs} epochs"
    )
    try:
        logger.info(f"🔧 Client {client_id}: Instantiating SmolVLAClient")
        client = SmolVLAClient(
            client_identifier=client_id,
            local_epochs=local_epochs,
            trainloader=trainloader,
            nn_device=nn_device,
            batch_size=batch_size,
            dataset_repo_id=dataset_name,
            mode=mode,
        )
        logger.info(f"✅ Client {client_id}: SmolVLAClient created successfully")
        logger.info(f"🚀 Client {client_id}: Converting to Flower client")
        flower_client = client.to_client()
        logger.info(
            f"✅ Client {client_id}: Client initialization COMPLETE - returning to Flower"
        )
        return flower_client
    except Exception as e:
        logger.error(f"❌ Client {client_id}: Failed during client creation: {e}")
        import traceback

        logger.error(
            f"❌ Client {client_id}: Client creation traceback: {traceback.format_exc()}"
        )
        raise


app = ClientApp(client_fn=client_fn)
