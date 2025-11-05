"""Server utility functions for SmolVLA federated learning."""

from __future__ import annotations


from pathlib import Path
from loguru import logger

# Import utils functions at module level for easier testing
from src.core.utils import get_tool_config
from src.server.model_utils import (
    extract_training_hyperparameters,
    extract_datasets,
    compute_in_memory_insights,
    save_model_checkpoint,
    generate_model_card,
)














def create_model_template():
    """Create a reusable model template for parameter operations using real dataset meta.

    This function abstracts the template model creation logic from AggregateEvaluationStrategy.__init__.
    It tries to load a real dataset first, then falls back to SO-100 compatible meta if datasets are unavailable.

    Returns:
        torch.nn.Module: SmolVLA model template with correct parameter shapes
    """
    try:
        # Try to load real dataset meta (same as server initialization)
        from src.core.utils import load_lerobot_dataset
        from src.configs import DatasetConfig
        from src.training.model_utils import get_model

        dataset_config = DatasetConfig.load()
        if dataset_config.server:
            server_config = dataset_config.server[0]
            dataset = load_lerobot_dataset(server_config.name)
            dataset_meta = dataset.meta
            logger.info(
                f"✅ Created model template using real dataset: {server_config.name}"
            )
            return get_model(dataset_meta=dataset_meta)
        else:
            raise ValueError("No server datasets configured")
    except Exception as e:
        logger.warning(
            f"Failed to load real dataset for template: {e} - falling back to SO-100 meta"
        )

        # Fallback to SO-100 compatible meta for standalone use
        class SO100Meta:
            def __init__(self):
                self.action_dim = 7
                self.state_dim = 0
                self.episode_length = 100
                self.stats = {"action": {"mean": [0.0] * 7, "std": [1.0] * 7}}
                self.features = {
                    "observation.image": {"dtype": "uint8", "shape": [3, 480, 640]},
                    "observation.state": {"dtype": "float32", "shape": [0]},
                    "action": {"dtype": "float32", "shape": [7]},
                }
                self.repo_id = "so100-generic"

        from src.training.model_utils import get_model

        meta = SO100Meta()
        template_model = get_model(dataset_meta=meta)
        logger.info("✅ Created model template using SO-100 fallback meta")
        return template_model


def prepare_server_eval_metrics(strategy, server_round):
    """Prepare server evaluation metrics for JSON logging and model cards.

    Args:
        strategy: The AggregateEvaluationStrategy instance
        server_round: Current server round number

    Returns:
        dict: Structured metrics dict for JSON serialization
    """
    # Get the latest eval loss
    composite_eval_loss = (
        strategy.server_eval_losses[-1]
        if hasattr(strategy, "server_eval_losses") and strategy.server_eval_losses
        else "N/A"
    )

    # Get aggregated client metrics
    aggregated_client_metrics = (
        strategy.last_aggregated_metrics
        if hasattr(strategy, "last_aggregated_metrics") and strategy.last_aggregated_metrics
        else {}
    )

    # Get individual client metrics
    individual_client_metrics = (
        strategy.last_client_metrics
        if hasattr(strategy, "last_client_metrics") and strategy.last_client_metrics
        else []
    )

    # Get per-dataset results
    server_eval_dataset_results = (
        strategy.last_per_dataset_results
        if hasattr(strategy, "last_per_dataset_results") and strategy.last_per_dataset_results
        else []
    )

    # Count datasets evaluated
    num_datasets_evaluated = len(server_eval_dataset_results)

    return {
        "composite_eval_loss": composite_eval_loss,
        "aggregated_client_metrics": aggregated_client_metrics,
        "individual_client_metrics": individual_client_metrics,
        "server_eval_dataset_results": server_eval_dataset_results,
        "num_datasets_evaluated": num_datasets_evaluated,
    }


