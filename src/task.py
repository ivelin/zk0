"""zk0: A Flower / Hugging Face LeRobot app."""

# Import all training functions from modular components
from src.training.model_utils import (
    get_model,
    compute_param_norms,
    log_param_status,
    get_params,
    get_trainable_params,
    set_trainable_params,
    extract_trainable_params,
    set_params,
)
from src.training.fedprox_utils import compute_fedprox_proximal_loss
from src.training.training_setup import (
    setup_training_components,
    reset_learning_rate_scheduler,
    create_train_metrics,
)
from src.training.training_loop import run_training_step, run_training_loop
from src.training.train import train
from src.training.evaluation import test
from src.training.scheduler_utils import (
    compute_adjustment_factor,
    create_scheduler,
    compute_adaptive_lr_factor,
    reset_scheduler_adaptive,
    compute_joint_adjustment,
)
