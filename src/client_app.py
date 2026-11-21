"""zk0: A Flower / Hugging Face LeRobot app."""

import sys
import os

import torch
from src.common.utils import load_lerobot_dataset

from loguru import logger

from flwr.client import Client, ClientApp
from flwr.common import Context

from src.client.client_core import SmolVLAClient


def _load_episode_list(path: str) -> list:
    """Load a P0 episode list (adaptation #4): accepts a bare JSON
    list, or a dict carrying the indices under "episodes" /
    "episode_indices" (possibly nested under "subset")."""
    import json as _json

    with open(path) as f:
        data = _json.load(f)

    def _find(obj):
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            for key in ("episodes", "episode_indices"):
                if key in obj and isinstance(obj[key], list):
                    return obj[key]
            for value in obj.values():
                found = _find(value)
                if found is not None:
                    return found
        return None

    episodes = _find(data)
    if episodes is None:
        raise ValueError(f"No episode list found in {path}")
    # P0 files carry per-episode record dicts; take their indices.
    out = []
    for e in episodes:
        if isinstance(e, dict):
            out.append(int(e["episode_index"]))
        else:
            out.append(int(e))
    return out


# Study 001 (D-11/G-6, row 19): the registered camera rename
# front->camera1, top->camera2, wrist->camera3. The study driver
# applies it in its data path (make_batches renames each item
# before stacking); lerobot 0.7.0's SmolVLA processor pipeline
# does NOT rename (its rename step carries an empty map - the
# certified pilot checkpoint's own policy_preprocessor.json shows
# rename_map {}), so on this substrate the rename belongs in the
# DataLoader collate, gated on the study configuration (the pooled
# preprocessor path being set). Mirrors _STUDY_RENAME_MAP in
# src/training/training_setup.py.
_STUDY_RENAME_MAP = {
    "observation.images.front": "observation.images.camera1",
    "observation.images.top": "observation.images.camera2",
    "observation.images.wrist": "observation.images.camera3",
}


def _study_rename_collate(batch_items):
    from torch.utils.data._utils.collate import default_collate

    renamed = [
        {_STUDY_RENAME_MAP.get(k, k): v for k, v in item.items()}
        for item in batch_items
    ]
    return default_collate(renamed)


def client_fn(context: Context) -> Client:
    """Construct a Client that will be run in a ClientApp."""

    print("[DEBUG client_fn] client_fn STARTED - context.node_config keys:", list(context.node_config.keys()), file=sys.stderr)
    print(f"[DEBUG client_fn] DATASET_NAME env: '{os.environ.get('DATASET_NAME', 'MISSING')}'", file=sys.stderr)
    print(f"[DEBUG client_fn] context keys: {list(context.__dict__.keys()) if hasattr(context, '__dict__') else 'No __dict__'}", file=sys.stderr)
    sys.stderr.flush()
    
    # Determine runtime mode based on presence of partition-id in node_config
    is_simulation = "partition-id" in context.node_config
    logger.info(f"🔍 MODE: is_simulation={is_simulation}, partition_id={context.node_config.get('partition-id', 'MISSING')}")
    print(f"[DEBUG client_fn] is_simulation={is_simulation} (node_config has partition-id: {'partition-id' in context.node_config})", file=sys.stderr)
    sys.stderr.flush()
    
    # Get dataset slug for unified path generation
    from src.common.utils import get_dataset_slug, load_env_safe
    logger.info(f"🔍 Calling get_dataset_slug in {'SIM' if is_simulation else 'PROD'} mode")
    print(f"[DEBUG FULL context.node_config] {context.node_config}", file=sys.stderr)
    print(f"[DEBUG FULL context.run_config] {context.run_config}", file=sys.stderr)
    dataset_slug = get_dataset_slug(context)
    print(f"[DEBUG client_fn] get_dataset_slug returned: '{dataset_slug}'", file=sys.stderr)
    
    # Determine client identifier
    if is_simulation:
        print("[DEBUG client_fn] Before raw_partition_id extraction", file=sys.stderr)
        sys.stderr.flush()
        # DEBUG: Log types and values before conversion for diagnosis
        raw_partition_id = context.node_config["partition-id"]
        print(f"[DEBUG client_fn] raw_partition_id='{raw_partition_id}' (type: {type(raw_partition_id)})", file=sys.stderr)
        sys.stderr.flush()
    
        print("[DEBUG client_fn] Before int conversion", file=sys.stderr)
        sys.stderr.flush()
        # Convert to int early for type consistency and assertion
        try:
            partition_id = int(raw_partition_id)
            print(f"[DEBUG client_fn] partition_id converted to int: {partition_id}", file=sys.stderr)
            sys.stderr.flush()
        except ValueError:
            print(f"[DEBUG client_fn] ValueError on int(): {raw_partition_id}", file=sys.stderr)
            sys.stderr.flush()
            logger.error(
                f"❌ client_fn: Invalid partition-id '{raw_partition_id}' - must be integer"
            )
            raise ValueError(f"Invalid partition-id in node_config: {raw_partition_id}")

        client_id = partition_id
    else:
        # Production mode: use node_id (reliable UUID from Flower)
        client_id = context.node_id

    logger.info(f"Client {client_id}: Running in {'simulation' if is_simulation else 'production'} mode, dataset_slug={dataset_slug}")


    load_env_safe()
    logger.debug("load_env_safe called in client_fn")

    # Runtime mode guards and validation
    if is_simulation:
        logger.info("🧪 Simulation mode: Ensuring local-only execution")
        # Guard: Warn if external dataset config is provided (will be ignored)
        if context.run_config.get("dataset.repo_id") or context.run_config.get("dataset.root"):
            logger.warning("⚠️ Simulation mode: dataset.repo_id or dataset.root in run_config will be ignored; using DatasetConfig partitions instead")
    else:
        logger.info("🔒 Production mode: Configuring for external networking")
        # Guard: For production, require DATASET_NAME env or node_config dataset-uri (ignore server run_config)
        if not (os.environ.get("DATASET_NAME") or context.node_config.get("dataset-uri")):
            error_msg = "❌ Production mode requires DATASET_NAME env var OR dataset-uri in node_config"
            logger.error(error_msg)
            raise ValueError(error_msg)

    # Determine client identifier
    if is_simulation:
        client_id = partition_id
    else:
        # Production mode: use node_id (reliable UUID from Flower)
        client_id = context.node_id
    logger.info(
        f"🚀 client_fn: Client function STARTED with node_id={context.node_id} and identifier {client_id}"
    )
    logger.debug(
        f"Client {client_id}: Full context.node_config: {context.node_config}"
    )
    logger.debug(
        f"Client {client_id}: Full context.run_config: {context.run_config}"
    )

    if is_simulation:
        # Read the node_config to fetch data partition associated to this node
        num_partitions = context.node_config["num-partitions"]
        logger.info(
            f"✅ Client {client_id}: Extracted partition_id={client_id}, num_partitions={num_partitions}"
        )

    # Extract save_path
    save_path = context.run_config.get("save_path")
    logger.info(f"Client {client_id}: save_path from context.run_config: '{save_path}' (type: {type(save_path)})")
    logger.info(f"Client {client_id}: context.run_config keys containing 'save' or 'path' or 'timestamp': {[k for k in context.run_config if 'save' in k.lower() or 'path' in k.lower() or 'time' in k.lower()]}")
    
    from src.common.utils import get_base_output_dir, get_client_dir
    base_dir = get_base_output_dir(save_path)
    client_dir = get_client_dir(base_dir, dataset_slug)
    
    logger.info(f"Client {client_id}: constructed base_dir={base_dir}, client_dir={client_dir}")

    # Discover device
    nn_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"✅ Client {client_id}: Device set to {nn_device}")

    # Read the run config to get settings to configure the Client
    model_name = context.run_config["model-name"]
    local_epochs = int(context.run_config["local-epochs"])
    logger.info(
        f"✅ Client {client_id}: Config loaded - model_name={model_name}, local_epochs={local_epochs}"
    )

    batch_size = context.run_config.get("batch_size", 64)
    logger.info(f"✅ Client {client_id}: Batch size set to {batch_size}")


    # Load dataset first to get metadata for model creation
    try:
        if not is_simulation:
            dataset_repo_id = dataset_slug  # Unified fallback from get_dataset_slug (node_config.dataset-uri / DATASET_NAME / run_config.dataset.repo_id)
            source = "dataset_slug fallback"
            logger.info(f"Client {client_id}: Loading dataset '{dataset_repo_id}' from {source}")
            dataset = load_lerobot_dataset(dataset_repo_id)
            dataset_name = dataset_repo_id
            logger.info(f"✅ Client {client_id}: Dataset loaded - samples: {len(dataset)}")
        else:
            # Simulation mode: load from partitions
            print(f"[DEBUG client_fn] Before DatasetConfig.load() for client_id={client_id}", file=sys.stderr)
            sys.stderr.flush()
            # Load dataset configuration
            from src.configs import DatasetConfig
        
            print("[DEBUG client_fn] DatasetConfig imported", file=sys.stderr)
            sys.stderr.flush()
            logger.info(f"🔍 Client {client_id}: Loading DatasetConfig")
            config = DatasetConfig.load()
            print(f"[DEBUG client_fn] DatasetConfig loaded, len(clients)={len(config.clients)}", file=sys.stderr)
            sys.stderr.flush()
            client_config = config.clients[client_id % len(config.clients)]
            dataset_name = client_config.name
            # Study 001 adaptation #8 (D-11/G-5): per-client seeding,
            # seed = 1000 + client_id (the registered seed 1000 with
            # the driver's per-node offset). The substrate had no
            # seeding anywhere; DataLoader shuffle draws from the
            # global RNG, so seeding here fixes each client's stream.
            import numpy as _np
            torch.manual_seed(1000 + int(client_id))
            _np.random.seed(1000 + int(client_id))
            logger.info(
                f"Client {client_id}: seeded torch/numpy RNG with {1000 + int(client_id)} (adaptation #8)"
            )
            print(f"[DEBUG client_fn] Selected dataset_name={dataset_name}", file=sys.stderr)
            sys.stderr.flush()
            logger.info(f"🔍 Client {client_id}: Selected dataset: {dataset_name}")

            # Load dataset directly
            print(f"[DEBUG client_fn] Before load_lerobot_dataset({dataset_name})", file=sys.stderr)
            sys.stderr.flush()
            logger.info(
                f"📥 Client {client_id}: Calling load_lerobot_dataset({dataset_name})"
            )
            # Study 001 adaptation #4 (D-11/G-1): when the client
            # config carries a root + P0 episodes file, load exactly
            # this node's task subset from the pinned on-disk tree.
            _episodes = None
            if getattr(client_config, "episodes_file", None):
                _episodes = _load_episode_list(client_config.episodes_file)
                logger.info(
                    f"Client {client_id}: P0 episode restriction from {client_config.episodes_file}: {len(_episodes)} episodes"
                )
            dataset = load_lerobot_dataset(
                dataset_name,
                root=getattr(client_config, "root", None),
                episodes=_episodes,
            )
            print(f"[DEBUG client_fn] load_lerobot_dataset returned dataset len={len(dataset)}", file=sys.stderr)
            sys.stderr.flush()
            logger.info(
                f"✅ Client {client_id}: Dataset loaded - samples: {len(dataset)}"
            )

        # Create dataloader (clients use full dataset for training) - common for both modes
        logger.info(
            f"🔄 Client {client_id}: Creating DataLoader (batch_size={batch_size}, num_workers=0)"
        )
        # Study 001 (D-11/G-6): in study configuration (pooled
        # preprocessor path set), collate with the registered
        # camera rename; otherwise the substrate default.
        _collate_fn = None
        try:
            from src.common.utils import get_tool_config as _gtc

            if (
                _gtc("flwr", "pyproject.toml")
                .get("app", {})
                .get("config", {})
                .get("pooled_preprocessor_path")
            ):
                _collate_fn = _study_rename_collate
                logger.info(
                    f"Client {client_id}: study camera-rename collate active (adaptation #9)"
                )
        except Exception:
            _collate_fn = None
        trainloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=0,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=False,
            drop_last=True,
            collate_fn=_collate_fn,
        )
        logger.info(
            f"✅ Client {client_id}: DataLoader created - length: {len(trainloader)}"
        )

        train_samples = len(dataset)
        logger.info(
            f"✅ Client {client_id}: Dataset loaded successfully - training samples: {train_samples}, trainloader length: {len(trainloader)}"
        )
    except Exception as e:
        logger.error(f"❌ Client {client_id}: Failed to load dataset: {e}")
        import traceback

        logger.error(
            f"❌ Client {client_id}: Dataset loading traceback: {traceback.format_exc()}"
        )
        raise

    # Client-side WandB removed - clients get recycled between rounds

    # Setup client logging once here (unified dir, before client creation)
    from src.logger import setup_client_logging
    setup_client_logging(
        save_path=save_path,
        client_id=client_id,
        dataset_slug=dataset_slug
    )
    logger.info(f"Client {client_id}: Client logging setup complete")

    # Stateless: No state management
    target_rounds = int(context.run_config.get("num-server-rounds", 250))
    logger.info(f"Client {client_id}: Stateless - will run all {target_rounds} rounds")
    
    logger.info(
        f"🏗️ Client {client_id}: Creating SmolVLAClient with {local_epochs} epochs"
    )
    try:
        logger.info(f"🔧 Client {client_id}: Instantiating SmolVLAClient (stateless)")
        
        client = SmolVLAClient(
            client_id=client_id,
            local_epochs=local_epochs,
            trainloader=trainloader,
            nn_device=nn_device,
            batch_size=batch_size,
            dataset_repo_id=dataset_slug,
            is_simulation=is_simulation,
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
