# Technologies and Development Setup

**Created**: 2025-09-06
**Last Updated**: 2025-09-26
**Version**: 1.0.2
**Author**: Kilo Code

## Core Technologies

### Python Ecosystem
- **Python 3.10+**: Primary programming language
- **PyTorch**: Deep learning framework for model training
- **Hugging Face Transformers**: Model loading and inference
- **LeRobot**: Robotics data handling and model integration
- **PEFT (Parameter-Efficient Fine-Tuning)**: LoRA adapters for memory-efficient training of SmolVLA
  - **SmolVLA Architecture**: Vision (86M frozen SigLIP) + Text (204M frozen SmolLM2) + Action Expert (101M trainable)
  - **LoRA Application**: Only to action expert (3.4M adapters, 96.6% efficiency); vision/text untouched
  - **LoRA Configuration**: rank=16, alpha=32, dropout=0.1; targets ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]
  - **Integration**: Wraps SmolVLAPolicy post-loading; trainable params 5M (1.1% vs 450M full); custom aggregation for adapters
  - **Runtime Issues**: Target modules mismatch, forward pass compatibility, AMP integration - fixes identified

### Federated Learning Framework
- **Flower Framework**: Federated learning orchestration and aggregation
  - **Version**: Flower 1.20.0
  - **Architecture**: Client-Server with Deployment Engine
  - **Supported ML Frameworks**: PyTorch, TensorFlow, JAX, MLX, 🤗 Transformers, PyTorch Lightning, scikit-learn, XGBoost, fastai, Pandas
  - **Key Components**: ClientApp, ServerApp, Strategies (FedAvg, FedProx), Mods
  - **Execution Modes**: Simulation Mode, Deployment Mode, GPU Support
  - **Custom Strategy**: LoRAFedAvg for adapter-only aggregation and merging (inspired by FlowerTune)

### SmolVLA Model
- **Model Size**: 453M parameters total (with LoRA adapters)
- **Vision-Language Model (VLM)**: SmolVLM2 backbone
  - **Vision Encoder**: SigLIP (86M params, always frozen)
  - **Language Decoder**: SmolLM2 (204M params, always frozen)
- **Action Expert**: Flow matching transformer (101M parameters, only trainable component)
- **Design Choices for Efficiency and Robustness**:
  - Visual token reduction (64 tokens per frame using PixelShuffle)
  - Layer skipping (half of VLM layers for faster inference)
  - Interleaved cross and self-attention blocks
  - Flow matching objective for continuous action prediction
  - Reduced hidden size (75% of VLM's) for lightweight deployment
- **Training Data**: Community-shared datasets under `lerobot` tag
- **Supported Hardware**: Consumer GPUs, CPUs, even MacBooks
- **Asynchronous Inference**: 30% faster response, 2× task throughput
- **Real-world Performance**: SO-100 and SO-101 compatibility
- **LoRA Adaptation**: 3.4M adapters on action expert only (96.6% efficiency); vision/text untouched

### Datasets
- **SO-100 Datasets**: Real-world robotics training data from SO-100 robot platform
- **SO-101 Datasets**: Advanced robotics training data from SO-101 robot platform
- **Format**: LeRobot format with lerobot tag
- **Annotation**: Clear task descriptions (max 30 characters)
- **Camera Views**: Standardized naming (OBS_IMAGE_1, OBS_IMAGE_2, etc.)
- **Frame Rate**: 30 FPS for SO-100/SO-101 datasets
- **Synchronization**: Proper tolerance (0.0001s = 1/fps) for accurate timestamp validation
- **Quality Assurance**: Automatic hotfix for doubled datasets (GitHub issue #1875)
- **Configuration**: Centralized YAML configuration in [`src/configs/datasets.yaml`](src/configs/datasets.yaml)

#### Federated Learning Client Datasets
See [`src/configs/datasets.yaml`](src/configs/datasets.yaml) for complete client dataset configuration including:
- **4 validated clients** with diverse robotics manipulation tasks
- **Dataset sizes and validation status** for each client
- **Train/eval episode splits** for proper federated learning setup
- **Quality assurance indicators** (CLEAN vs HOTFIX APPLIED)

#### Evaluation Datasets
- **choyf3/so101_test_20250908**: SO-101 test dataset (12 episodes, 9,128 frames)
- **griffinlabs/record-trial-2**: Research laboratory trials (20 episodes, 14,607 frames)
- **JamesChen007/so101-test**: SO-101 comprehensive test (10 episodes, 3,946 frames)

## Integration Specifications

### SmolVLA + Flower Integration
- **Framework Compatibility**: Flower 1.20.0 with Ray 2.31.0
- **Dataset Format**: Flower Datasets for partitioning
- **Model Loading**: Direct integration with LeRobot SmolVLA models
- **Federated Dataset**: FederatedLeRobotDataset for distributed data
- **PEFT/LoRA Integration**: Adapter-only parameter exchange; server merges adapters into base model

### Key Integration Points
1. **Client Implementation**: Extend NumPyClient for SmolVLA with LoRA wrapping
2. **Server Strategy**: Use LoRAFedAvg for adapter aggregation (average A/B matrices, merge on server)
3. **Data Partitioning**: LeRobotDatasetPartitioner for episode-based splitting
4. **Model Aggregation**: Flower's parameter aggregation mechanisms adapted for LoRA (small payloads ~1MB)

## Development Environment
- **Primary**: Docker container (`zk0`) via train.sh for reproducible, isolated execution of training and simulations
- **Alternative**: Conda environment ("zk0") for local development and training runs (validated for federated learning execution)
- **VSCode**: IDE with Docker integration and automatic environment detection
- **Git**: Version control with GitHub integration
- **Pre-commit Hooks**: Code quality and formatting checks

## Dependencies Management
- **requirements.txt**: Pinned versions for reproducibility
- **pyproject.toml**: Project configuration and metadata
- **New Dependencies for LoRA** (add to requirements.txt):
  - peft>=0.13.0  # LoRA adapters (Hugging Face PEFT library)
  - accelerate>=1.10.1  # Distributed training, AMP for mixed precision in FL
  - bitsandbytes>=0.47.0  # Optional: 8-bit quantization for further memory reduction during training

## Hardware Requirements
- **Python Version**: 3.8+ (recommended 3.10)
- **PyTorch Version**: Compatible with SmolVLA requirements
- **CUDA Version**: 11.0+ for GPU support
- **Memory Requirements**: 8GB+ RAM, 4GB+ VRAM for GPU (reduced to ~6-8GB peak with LoRA; ~5-10GB for batch=32, 1000 steps)
- **GPU Support**: CUDA-compatible GPUs for model training
- **Storage**: Sufficient space for datasets and model checkpoints

## Testing and Quality Assurance
- **Test Coverage**: 80% minimum for new code
- **pytest**: Unit and integration testing framework (run via Docker for consistency)
- **Coverage.py**: Test coverage measurement
- **Black**: Code formatting
- **mypy**: Type checking
- **Documentation**: Complete API documentation
- **Reproducibility**: Seeds for all experiments
- **LoRA-Specific Tests**: Unit tests for adapter loading/saving and forward pass; integration tests for FL aggregation (LoRAFedAvg) and memory efficiency

## Logging and Monitoring
- **Loguru Framework**: Structured logging with rotation, compression, and multi-process safety
- **Unified Log File**: `outputs/<timestamp>/simulation.log` mirrors console output exactly (via tee in train.sh) and consolidates all app, Flower, and Ray messages
- **Server Log**: Dedicated `outputs/<timestamp>/server/server.log` for server-specific messages (aggregation, configuration)
- **Per-Client Logs**: Individual `outputs/<timestamp>/clients/client_{id}/client.log` for each client's training/evaluation messages
- **Evaluation Stats**: Client evaluation metrics saved to `clients/client_X/round_N.json`, server aggregates to `server/round_N_aggregated.json`
- **Process Safety**: `enqueue=True` for thread-safe multi-process logging across Ray workers
- **Structured Format**: Includes client_id, process_id, round number, VRAM/RAM diagnostics, and contextual information
- **Flower/Ray Integration**: Full bridging of Flower and Ray loggers to Loguru handlers for complete capture in appropriate files with original caller information
- **Rotation Policy**: 500MB files with 10-day retention and zip compression
- **Diagnostics**: VRAM/RAM monitoring, training metrics, error tracking, and resource usage
- **Configuration**: Automatic setup via `src/logger.py` with client/server coordination; clean separation prevents log duplication

## Deployment and Operations
- **Docker**: Containerization for reproducible deployments
- **Kubernetes**: Orchestration for distributed clients
- **Monitoring**: Logging and performance tracking
- **CI/CD**: Automated testing and deployment pipelines
- **Network Configuration**: Proper TLS and authentication setup
- **Scalability**: Plan for multiple clients and rounds

## Installation and Setup

### Docker-Based Setup (Recommended)
```bash
# Build the Docker image with all dependencies
docker build -t zk0 .

# Run federated learning with serialized execution
./train.sh

# Or run directly with Docker
docker run --gpus all --shm-size=10.24gb \
  -v $(pwd):/workspace \
  -v $(pwd)/outputs:/workspace/outputs \
  -v /tmp:/tmp \
  -w /workspace \
  zk0 flwr run . local-simulation-serialized-gpu --run-config "num-server-rounds=2"
```

### Standard Training Execution
**ALWAYS use `./train.sh` from the project root directory for training runs.** This script provides:
- Proper Docker environment setup
- GPU resource management
- Cache volume mounting for model persistence
- Error handling and logging
- Consistent execution across different systems

**Usage:**
```bash
# Basic training (uses pyproject.toml defaults: 1 round, 2 steps/epochs, serialized GPU)
./train.sh

# For custom config, use direct Flower run with overrides
flwr run . local-simulation-serialized-gpu --run-config "num-server-rounds=5 local-epochs=10"

# Or with Docker directly (example with overrides)
docker run --gpus all --shm-size=10.24gb \
  -v $(pwd):/workspace \
  -v $(pwd)/outputs:/workspace/outputs \
  -v /tmp:/tmp \
  -v $HOME/.cache/huggingface:/home/user_lerobot/.cache/huggingface \
  -w /workspace \
  zk0 flwr run . local-simulation-serialized-gpu --run-config "num-server-rounds=5"
```

**Configuration Notes:**
- Edit `[tool.flwr.app.config]` in `pyproject.toml` for defaults (e.g., num-server-rounds=1, local-epochs=2).
- Use `local-simulation-serialized-gpu` for reliable execution (prevents SafeTensors issues; max-parallelism=1).
- train.sh has no CLI parameters; all config via pyproject.toml.

### Conda Environment Setup (Alternative)
#### Create and Configure Conda Environment
```bash
# Create the zk0 environment
conda create -n zk0 python=3.10 -y

# Install system dependencies
conda install ffmpeg=7.1.1 -c conda-forge

# Activate environment
conda activate zk0

# Install Python dependencies
pip install -r requirements.txt
pip install -e .
```

#### SmolVLA Installation
```bash
# Install LeRobot with SmolVLA dependencies from PyPI
pip install 'lerobot[smolvla]'
```

#### Flower Installation
```bash
# Install Flower
pip install flwr[simulation]

# For latest features
pip install flwr[simulation] --pre
```

#### Conda-Based Training Execution
```bash
# Run federated learning directly in conda environment
conda run -n zk0 flwr run . local-simulation-serialized-gpu --run-config "num-server-rounds=2"

# Or activate environment first
conda activate zk0
flwr run . local-simulation-serialized-gpu --run-config "num-server-rounds=2"
```

**Note**: Conda execution provides a lightweight alternative to Docker for development and testing, with validated reliability for federated learning runs.

### Basic Usage Examples

#### SmolVLA Finetune
```bash
python lerobot/scripts/train.py \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=<dataset_from_config> \
  --batch_size=64 \
  --steps=20000 \
  --output_dir=outputs/train/my_smolvla \
  --job_name=my_smolvla_training \
  --policy.device=cuda \
  --wandb.enable=true
```
*See [`src/configs/datasets.yaml`](src/configs/datasets.yaml) for available dataset options*

#### Flower Simulation
```bash
# Run basic simulation (10 clients, CPU by default)
flwr run .

# Run GPU simulation (4 clients - RECOMMENDED for SmolVLA/SO-100)
flwr run . local-simulation-gpu

# Run CPU simulation if GPU not available (4 clients)
flwr run . local-simulation --run-config "backend.client-resources.num-gpus=0"

# Override configuration
flwr run . local-simulation-gpu --run-config "num-server-rounds=5 fraction-fit=0.1"
```

**Default Recommendation**: Always use `local-simulation-gpu` when GPU is available (100x faster than CPU). Only fall back to `local-simulation` with `backend.client-resources.num-gpus=0` if GPU is not available. This matches the 4-client SO-100 dataset architecture and provides optimal performance.

## Evaluation and Visualization

### Evaluation System
- **Robot Rollout Evaluation**: Comprehensive evaluation with predicted vs ground truth action comparison
- **Metrics Calculation**: Success rate, action MSE, trajectory length analysis
- **Multi-Episode Support**: Evaluate across multiple SO-100 episodes
- **Fallback Mechanisms**: Graceful degradation when real evaluation unavailable

### Visualization Tools
- **LeRobot Compatibility**: Data export compatible with Hugging Face dataset visualizer
- **Robot Rollout Videos**: Animated visualizations of predicted trajectories
- **Comparison Visualizations**: Side-by-side comparison of predicted vs ground truth
- **Progress Tracking**: Visual progress across federated learning rounds
- **Eval MSE Chart**: End-of-session line chart showing per-client and server average action_mse over all rounds (eval_mse_chart.png)
- **MSE History JSON**: Reproducible historical MSE data saved as eval_mse_history.json

### Key Modules
- **src/evaluation.py**: SmolVLAEvaluator class for comprehensive robot rollout evaluation
- **src/visualization.py**: SmolVLAVisualizer class for LeRobot-compatible data export
- **Enhanced client_app.py**: Integrated evaluation with robot rollouts and visualization

## Official Resources

### LeRobot Resources
- **GitHub Repository**: https://github.com/huggingface/lerobot
- **Documentation**: https://huggingface.co/docs/lerobot/
- **Datasets Collection**: https://huggingface.co/lerobot/datasets
- **Dataset Visualizer**: https://huggingface.co/spaces/lerobot/visualize_dataset

### SmolVLA Resources
- **Model Hub**: https://huggingface.co/lerobot/smolvla_base
- **Documentation**: https://huggingface.co/docs/lerobot/smolvla
- **Blog Post**: https://huggingface.co/blog/smolvla
- **Paper**: https://huggingface.co/papers/2506.01844
- **Code Repository**: https://github.com/huggingface/lerobot

### Flower Resources
- **Framework Documentation**: https://flower.ai/docs/framework/index.html
- **Running Simulations**: https://flower.ai/docs/framework/how-to-run-simulations.html
- **Configuring Logging**: https://flower.ai/docs/framework/how-to-configure-logging.html
- **Configuring Audit Logging**: https://flower.ai/docs/framework/how-to-configure-audit-logging.html
- **GitHub Repository**: https://github.com/adap/flower
- **General Examples**: https://github.com/adap/flower/tree/main/examples
- **LeRobot Integration Example**: https://flower.ai/docs/examples/quickstart-lerobot.html
- **LeRobot Example Repository**: https://github.com/adap/flower/tree/main/examples/quickstart-lerobot

### Ray Resources
- **General Documentation**: https://docs.ray.io/en/latest/
- **Logging Configuration Guide**: https://docs.ray.io/en/latest/ray-observability/user-guides/configure-logging.html
- **GitHub Repository**: https://github.com/ray-project/ray

### PEFT/LoRA Resources
- **Hugging Face PEFT Docs**: https://huggingface.co/docs/peft/main/en/conceptual_guides/lora (core LoRA implementation for Transformers)
- **FlowerTune Repo**: https://github.com/adap/flower/tree/main/examples/flower-tune (FL with PEFT/LoRA for LLMs, adapted for SmolVLA)
- **arXiv Paper on Federated PEFT**: https://arxiv.org/abs/2506.02961 (FedLoRA for heterogeneous tasks like robotics)
- **SmolVLM Fine-Tuning Guide**: https://huggingface.co/blog/smolvla (LoRA on SmolVLM backbone for vision-language tasks)

## Local Repositories
- **LeRobot**: Local git repository available at `$HOME/lerobot`. Contains source code, tests, examples, and documentation for LeRobot framework.
- **Flower**: Local git repository available at `$HOME/flower`. Includes examples directory with LeRobot integration.
  - **Pusht Diffusion LeRobot Example**: Located at `$HOME/flower/examples/quickstart-lerobot`. Demonstrates federated learning with LeRobot's pusht dataset using diffusion policies.

### Repository Sync Instructions
To avoid version mismatches between installed dependencies and local repo code, sync the repositories to match zk0's requirements.txt:
- **Flower Sync**: Extract version from requirements.txt (e.g., `FLOWER_VERSION=$(grep '^flwr' requirements.txt | cut -d '~' -f 3 | cut -d '.' -f 1-2)`), then `cd $HOME/flower && git fetch --tags && git checkout tags/v${FLOWER_VERSION}.0` (adjust for exact tag format).
- **LeRobot Sync**: Use `LEROBOT_VERSION=$(pip show lerobot | grep Version | cut -d ' ' -f 2)` then `cd $HOME/lerobot && git fetch --tags && git checkout tags/v${LEROBOT_VERSION}` (or nearest matching tag if exact not available).
Run these commands after any requirements.txt updates or when switching branches to maintain consistency. Always verify the checked-out version matches the installed dependency.

## Known Fixes and Configurations

### FL Scheduler Reset for Partial Rounds
- **Issue**: LeRobot's default cosine scheduler decays LR over full training (~20k steps), but in zk0's FL, each round creates a new scheduler starting at decayed LR (~1e-7), causing negligible param updates (norm delta ~0.0003).
- **Fix Location**: src/task.py, line ~159 in train() (after make_optimizer_and_scheduler).
- **Implementation**:
  - Override optimizer param groups to initial LR=1e-4 (standard for SmolVLA finetuning).
  - Reset scheduler with `lr_scheduler.last_epoch = -1` to restart decay per round.
  - Logs: "FL scheduler reset: Set initial LR=1e-4, last_epoch=-1".
- **Rationale**: Ensures meaningful updates in short FL rounds (10-1000 steps). Mild intra-round decay balances exploration/stability. Aligns with Flower's stateless clients.
- **Validation**: Check logs for override message, pre/post-norm delta >0.01, decreasing loss across rounds.
- **Benefits**: Effective learning without global step tracking; compatible with zk0's partial-step architecture (200 steps/round default).

### LoRA Adapter Exchange
- **Client-Server Flow**: Clients load base + adapters; train adapters only; send adapter state_dict (~1MB). Server averages A/B matrices, merges into base, broadcasts merged adapters.
- **Memory Savings**: 98.9% parameters frozen, 19.3MB trainable vs 1729.8MB total; 99.75% bandwidth reduction.
- **Parameter Breakdown**: 3.4M LoRA adapters on action expert (3.36% of expert params); vision/text untouched.
- **Runtime Issues**: Target modules configuration mismatch, forward pass compatibility, AMP integration.
- **Fix Plan**: Update target_modules to SmolVLA expert naming, add validation, optimize hyperparameters.
- **Compatibility**: Backward-compatible flag in config; falls back to full fine-tuning if disabled.