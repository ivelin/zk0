# Technologies and Development Setup

**Created**: 2025-09-06
**Last Updated**: 2025-11-01
**Version**: 1.0.5
**Author**: Kilo Code

## Core Technologies

### Python Ecosystem
- **Python 3.10+**: Primary programming language
- **PyTorch**: Deep learning framework for model training
- **Hugging Face Transformers**: Model loading and inference
- **LeRobot**: Robotics data handling and model integration

### Federated Learning Framework
- **Flower Framework**: Federated learning orchestration and aggregation
  - **Version**: Flower 1.22.0
  - **Architecture**: Client-Server with Deployment Engine
  - **Supported ML Frameworks**: PyTorch, TensorFlow, JAX, MLX, 🤗 Transformers, PyTorch Lightning, scikit-learn, XGBoost, fastai, Pandas
  - **Key Components**: ClientApp, ServerApp, Strategies (FedAvg, FedProx), Mods
  - **Execution Modes**: Simulation Mode, Deployment Mode, GPU Support

#### Enhanced Security & Validation
- **Bidirectional Hash Validation**: SHA256 parameter integrity checking in both directions.
  - **Server → Client:** Computes hash before transmission, sends via config.
  - **Client → Server:** Validates received parameters; computes and sends updated hashes via metrics.
  - **Server Validation:** Re-computes hashes for each client update and compares.
  - **Error Handling:** Raises `RuntimeError` on mismatches to prevent corrupted training.
- **Implementation:** Added to `src/server_app.py` (configure_fit, configure_evaluate, aggregate_fit) and `src/client_app.py` (fit, evaluate).
- **Rationale:** Addresses serialization regressions; catches dtype/array corruption early.
- **Status:** Production-ready; tested for compatibility with existing Flower/FedProx logic.

### SmolVLA Model
- **Model Size**: 450M parameters total
- **Vision-Language Model (VLM)**: SmolVLM2 backbone
  - Vision Encoder: SigLIP
  - Language Decoder: SmolLM2
- **Action Expert**: Flow matching transformer (~100M parameters)
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

#### SmolVLA Usage Examples
**Finetune Pretrained Model:**
```bash
python lerobot/scripts/train.py \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=lerobot/svla_so100_stacking \
  --batch_size=64 \
  --steps=20000 \
  --output_dir=outputs/train/my_smolvla \
  --job_name=my_smolvla_training \
  --policy.device=cuda \
  --wandb.enable=true
```

**Train from Scratch:**
```bash
python lerobot/scripts/train.py \
  --dataset.repo_id=lerobot/svla_so100_stacking \
  --batch_size=64 \
  --steps=200000 \
  --output_dir=outputs/train/my_smolvla \
  --job_name=my_smolvla_training \
  --policy.device=cuda \
  --wandb.enable=true
```

### Flower Framework
- **Version**: Flower 1.22.0
- **Architecture**: Client-Server with Deployment Engine
- **Supported ML Frameworks**: PyTorch, TensorFlow, JAX, MLX, 🤗 Transformers, PyTorch Lightning, scikit-learn, XGBoost, fastai, Pandas
- **Key Components**: ClientApp, ServerApp, Strategies (FedAvg, FedProx), Mods
- **Execution Modes**: Simulation Mode, Deployment Mode, GPU Support

#### Flower Installation and Setup
```bash
# Install Flower
pip install flwr[simulation]

# For latest features
pip install flwr[simulation] --pre
```

#### Flower Basic Usage
```bash
# Run simulation
flwr run .

# Run with GPU federation
flwr run . local-simulation-gpu

# Override configuration
flwr run . local-simulation-gpu --run-config "num-server-rounds=5 fraction-fit=0.1"
```

### Datasets
For detailed dataset information, including SO-100/SO-101 formats, client assignments, and evaluation datasets, see [docs/ARCHITECTURE.md#training-strategy](../docs/ARCHITECTURE.md#training-strategy).

## Integration Specifications

### SmolVLA + Flower Integration
- **Framework Compatibility**: Flower 1.22.0 with Ray 2.10.0
- **Dataset Format**: Flower Datasets for partitioning
- **Model Loading**: Direct integration with LeRobot SmolVLA models
- **Federated Dataset**: FederatedLeRobotDataset for distributed data

### Key Integration Points
1. **Client Implementation**: Extend NumPyClient for SmolVLA
2. **Server Strategy**: Use FedAvg or custom strategies
3. **Data Partitioning**: LeRobotDatasetPartitioner for episode-based splitting
4. **Model Aggregation**: Flower's parameter aggregation mechanisms

## Development Environment
For installation, setup, and hardware requirements, see [docs/INSTALLATION.md](../docs/INSTALLATION.md).

### Testing Standards
- **Real Scenarios**: Prioritize actual FL workflows, SmolVLA interactions, SO-100 processing.
- **No Mocks in Production**: Tests use real dependencies; fail-fast on missing env.
- **Coverage**: 30% minimum; focus on integration points (Flower ↔ SmolVLA).
- **Zk0-Specific**: Test parameter exchange, multi-repo partitioning, hash validation.
- **Execution**: Always in Docker (`zk0`) or conda (`zk0`) for consistency.
- **Parallel**: `pytest -n auto` with coverage reporting.

### Code Quality Standards
- **Code Style**: PEP 8, type hints, docstrings.
- **Modularity**: Separate concerns (e.g., task.py for training).
- **Error Handling**: Raise RuntimeError for missing deps.
- **Reproducibility**: Pin deps in pyproject.toml; use seeds (e.g., 42).
- **Tool Usage**: Batch file reads; diff format for edits.
- **Context Management**: Maintain project context through documentation updates and reviews.
- **Performance**: GPU optimization, AMP; monitor VRAM.
- **File Size Limits**: New source files must be under 500 lines of code (LOC) when possible for maintainability and readability. This ensures modular design and prevents bloated files. Exceptions require explicit approval in code reviews. Existing large files (e.g., server_app.py) should be refactored into smaller modules during maintenance.

### CI/CD and Deployment
- **Docker-Based Testing**: CI uses `Dockerfile.ci` (CPU-only) for isolation; local uses `Dockerfile.zk0` (GPU).
- **Parallel Coverage**: `.coveragerc` enables parallel mode; `coverage combine` merges files.
- **GitHub Actions**: Auto-build/push on tag; GPU-only platforms.
- **Local Simulation**: Test CI locally with `docker build -f Dockerfile.ci -t zk0-ci .` then run tests.
- **Deployment**: Docker Compose for prod; GHCR for images.

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

### Unified Logging Architecture
- **Coordination**: Server creates log file; passes path to clients via Flower config.
- **Safety**: `enqueue=True` for multi-process (Ray) compatibility.
- **Format**: Includes client_id, PID, round, VRAM/RAM diagnostics.
- **Rotation**: 500MB files, 10-day retention, zip compression.
- **Integration**: Bridges Flower/Ray logs to Loguru.
- **Locations**: Unified `outputs/YYYY-MM-DD_HH-MM-SS/simulation.log`, server `server/server.log`, clients `clients/client_N/client.log`.

## Experiment Tracking and Monitoring

### WandB Integration
- **Unified Run Management**: Single WandB run per federated learning experiment with client-prefixed metrics
- **Run Naming**: `zk0-sim-fl-run-{timestamp}` format for server-created runs
- **Client Participation**: Clients join server's run using `run_id` passed via `context.run_config`
- **Metric Prefixing**: Client metrics use `client_{id}_` prefix (e.g., `client_0_training_loss`, `client_1_fedprox_regularization_loss`)
- **Server Metrics**: Server aggregates use `server_` prefix (e.g., `server_avg_policy_loss`, `server_num_clients`)
- **Session Management**: Automatic `wandb.finish()` call after final evaluation round
- **Configuration**: Enabled via `use-wandb: true` in `pyproject.toml` `[tool.flwr.app.config]`
- **Integration Points**: `src/wandb_utils.py` provides initialization and logging utilities
- **Validation**: Ensures no duplicate runs - clients join existing run, don't create separate ones

## Deployment and Operations
- **Docker**: Containerization for reproducible deployments
- **Kubernetes**: Orchestration for distributed clients
- **Monitoring**: Logging and performance tracking
- **CI/CD**: Automated testing and deployment pipelines
- **Network Configuration**: Proper TLS and authentication setup
- **Scalability**: Plan for multiple clients and rounds

## Contributing Guidelines
- **Node Operators**: Join with SO-100 arm + RTX 3090+ GPU for data/compute.
- **Code**: Bug fixes, features, docs, tests.
- **Process**: Fork repository, create feature branch, commit changes (lint with Ruff), run tests (`pytest`), submit PR.
- **Focus**: At Beta stage, focus on core FL functionality and community onboarding.


#### Model Pushing to Hugging Face Hub
```bash
# Push checkpoint directory to HF Hub (requires conda zk0 environment)
conda run -n zk0 python -m zk0.push_to_hf /path/to/checkpoint_dir --repo-id your-username/your-model

# Prerequisites:
# - HF token in .env: HF_TOKEN=your_token_here
# - Checkpoint directory must contain: model.safetensors, config.json, README.md, metrics.json, etc.
# - Uses api.upload_folder() for complete directory upload
```

## Evaluation and Visualization

### Evaluation System
- **Server-Side Evaluation**: Global model evaluated server-side using dedicated evaluation datasets
- **Policy Loss Evaluation**: Uses same loss metric as client training (policy.forward() loss, ~1 scale)
- **Episode Limiting**: Evaluation limited to first N episodes from evaluation datasets
- **Fallback Mechanisms**: Graceful degradation when real evaluation unavailable

### Visualization Tools
- **LeRobot Compatibility**: Data export compatible with Hugging Face dataset visualizer
- **Robot Rollout Videos**: Animated visualizations of predicted trajectories
- **Comparison Visualizations**: Side-by-side comparison of predicted vs ground truth
- **Progress Tracking**: Visual progress across federated learning rounds
- **Eval Policy Loss Chart**: End-of-session line chart showing per-client and server average policy_loss over all rounds (eval_policy_loss_chart.png)
- **Policy Loss History JSON**: Reproducible historical policy_loss data saved as eval_policy_loss_history.json

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

## Local Repositories
- **LeRobot**: Local git repository available at `$HOME/lerobot`. Contains source code, tests, examples, and documentation for LeRobot framework.
- **Flower**: Local git repository available at `$HOME/flower`. Includes examples directory with LeRobot integration.
  - **Pusht Diffusion LeRobot Example**: Located at `$HOME/flower/examples/quickstart-lerobot`. Demonstrates federated learning with LeRobot's pusht dataset using diffusion policies.

### Repository Sync Instructions
To avoid version mismatches between installed dependencies and local repo code, sync the repositories to match zk0's pyproject.toml:
- **Flower Sync**: Extract version from pyproject.toml (e.g., `FLOWER_VERSION=$(grep '^flwr' pyproject.toml | cut -d '~' -f 3 | cut -d '.' -f 1-2)`), then `cd $HOME/flower && git fetch --tags && git checkout tags/v${FLOWER_VERSION}.0` (adjust for exact tag format).
- **LeRobot Sync**: Use `LEROBOT_VERSION=$(pip show lerobot | grep Version | cut -d ' ' -f 2)` then `cd $HOME/lerobot && git fetch --tags && git checkout tags/v${LEROBOT_VERSION}` (or nearest matching tag if exact not available).
Run these commands after any pyproject.toml updates or when switching branches to maintain consistency. Always verify the checked-out version matches the installed dependency.

## Code Organization and Refactoring

### File Size Limits
- **New Source Files**: All new source files should be under 500 lines of code (LOC) when possible for maintainability and readability. This ensures modular design and prevents bloated files. Exceptions require explicit approval in code reviews.

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