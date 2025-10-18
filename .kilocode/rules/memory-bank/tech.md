# Technologies and Development Setup

**Created**: 2025-09-06
**Last Updated**: 2025-10-14
**Version**: 1.0.3
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
- **SO-100 Datasets**: Real-world robotics training data from SO-100 robot platform
- **SO-101 Datasets**: Advanced robotics training data from SO-101 robot platform
- **Format**: LeRobot format with lerobot tag
- **Annotation**: Clear task descriptions (max 30 characters)
- **Camera Views**: Standardized naming (OBS_IMAGE_1, OBS_IMAGE_2, etc.)
- **Frame Rate**: 30 FPS for SO-100/SO-101 datasets
- **Synchronization**: Proper tolerance (0.0001s = 1/fps) for accurate timestamp validation
- **Quality Assurance**: Automatic hotfix for doubled datasets (GitHub issue #1875)
#### Federated Learning Client Datasets
See [Configuration System](architecture.md#configuration-system) in `pyproject.toml` under `[tool.zk0.datasets]` for complete client dataset configuration including:
- **4 validated clients** with diverse robotics manipulation tasks
- **Dataset sizes and validation status** for each client
- **Train/eval episode splits** for proper federated learning setup
- **Quality assurance indicators** (CLEAN vs HOTFIX APPLIED)
- **Latest Update (2025-10-14)**: Updated to 4 validated clients with diverse SO-100/SO-101 tasks:
  - **Client 0**: shaunkirby/record-test - "Put the red LEGO in the bin"
  - **Client 1**: ethanCSL/direction_test - "turn to the right side" (VALIDATED CLEAN)
  - **Client 2**: gimarchetti/so101-winnie-us5 - "rub the plush toy with the bottle" (VALIDATED CLEAN)
  - **Client 3**: olingoudey/so101_stationary_mug445 - "Put the stuffed animal in the mug" (VALIDATED CLEAN)

#### Evaluation Datasets
- **Hupy440/Two_Cubes_and_Two_Buckets_v2**: SO-101 test dataset (12 episodes, 9,128 frames) - "Pick up a cube. Is the cube red, put it in the white bucket. Is the cube white, put it in the red bucket."
- **choyf3/so101_test_20250908**: SO-101 test dataset (12 episodes, 9,128 frames)
- **griffinlabs/record-trial-2**: Research laboratory trials (20 episodes, 14,607 frames)
- **JamesChen007/so101-test**: SO-101 comprehensive test (10 episodes, 3,946 frames)

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
- **Primary**: Conda environment ("zk0") for local development and training runs (validated for federated learning execution)
- **Alternative**: Docker container (`zk0`) via train.sh for fallback execution
- **VSCode**: IDE with Docker integration and automatic environment detection
- **Git**: Version control with GitHub integration
- **Pre-commit Hooks**: Code quality and formatting checks
- **Time Zone Handling**: Always check environment_details for the current user time zone when interpreting timestamps in logs, file modification times, or system outputs, as it may vary between sessions and developers.

## Command Execution Requirements (CRITICAL)

**MANDATORY ENVIRONMENT CONSTRAINTS FOR ALL COMMANDS**:

1. **Primary Environment**: ALL code execution, testing, and validation MUST use Conda environment "zk0" (`conda run -n zk0 <command>`)
2. **Alternative Environment**: Docker container (`zk0`) via `train.sh` for fallback execution
3. **Prohibited**: Never use system Python or bare commands without proper environment activation
4. **Validation**: Always verify environment before execution - check that required packages (torch, lerobot, etc.) are available

**EXECUTION PATTERNS**:
- **Testing**: `conda run -n zk0 python -m pytest -n auto --cov=src --cov-report=term-missing`
- **Syntax Check**: `conda run -n zk0 python -c "import ast; ast.parse(open('file.py').read())"`
- **Code Execution**: `conda run -n zk0 python -c "your code here"`
- **Training**: **ALWAYS use `./train.sh` for training runs** (preferred over direct flwr commands)
  - **Tiny Training**: `./train.sh --tiny` for quick validation runs
  - **Full Training**: `./train.sh` for standard training with pyproject.toml config
  - **Docker Training**: `./train.sh --docker` for containerized execution
  - **Direct FL Commands**: Only use `conda run -n zk0 flwr run . local-simulation-serialized-gpu --run-config "..."` for debugging or custom configurations when train.sh cannot accommodate

**FAILURE TO COMPLY**: Commands will fail with import errors (ModuleNotFoundError). Always use proper environment activation.

## Dependencies Management
- **requirements.txt**: Pinned versions for reproducibility
- **pyproject.toml**: Project configuration and metadata

## Hardware Requirements
- **Python Version**: 3.8+ (recommended 3.10)
- **PyTorch Version**: Compatible with SmolVLA requirements
- **CUDA Version**: 11.0+ for GPU support
- **Memory Requirements**: 8GB+ RAM, 4GB+ VRAM for GPU
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

### Testing Standards for zk0 Project
- **Real Scenario Emphasis**: Prioritize testing actual federated learning workflows, SmolVLA model interactions, and SO-100 dataset processing over isolated mocking
- **Zk0-Specific Scenarios**: Focus on federated learning client-server communication, model aggregation with Flower strategies, and SmolVLA parameter handling in distributed environments
- **Integration Points**: Test real data flow between components (e.g., Flower ↔ SmolVLA, client ↔ server parameter exchange)
- **Parameter Data Types**: Validate correct handling of SmolVLA tensors, Flower NumPy arrays, and SO-100 dataset formats
- **Strict Dependency Enforcement**: Tests MUST fail when required dependencies are missing - no graceful fallbacks or skips that mask environment setup issues
- **Environment Validation**: All tests require full environment setup with real dependencies (TorchCodec, FFmpeg, LeRobot, SmolVLA) - no optional dependencies
- **User Journey Priority**: Focus on complete federated learning rounds and robotics task workflows rather than method-level coverage
- **Real Environment Testing**: Test against actual conda zk0 environment, real imports, and SO-100 data, not mocked states
- **Remove Low-Value Tests**: Delete tests that only exercise mocking without testing real federated learning behavior
- **Fail-Fast Testing**: Tests should expose environment and dependency issues immediately rather than hiding them with fallbacks
- **Integration Testing**: Verify bidirectional hash validation in full FL rounds.
- **Edge Cases**: Test artificial corruption to confirm `RuntimeError` triggers.
- **Performance**: Monitor hash computation overhead (<1% of round time).

### Compliance Requirements
- **Technical Standards**: Python 3.8+, PyTorch compatible with SmolVLA, CUDA 11.0+, 8GB+ RAM, 4GB+ VRAM
- **Dataset Standards**: LeRobot format with lerobot tag, clear task descriptions (max 30 characters), standardized camera naming, 30 FPS
- **Quality Standards**: 80% test coverage minimum, complete API documentation, seeds for reproducibility, regular benchmarking
- **Implementation Guidelines**: Use conda environment "zk0", follow LeRobot and Flower patterns, comprehensive unit and integration tests, inline documentation
- **Deployment Considerations**: Match official hardware recommendations, proper TLS setup, logging and monitoring, scalability planning
- **Maintenance Requirements**: Stay current with SmolVLA and Flower releases, apply security patches, regular optimization, follow community updates

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
- **Local package installation**: Automatically installs the project as an editable package (`pip install -e .`) to ensure the latest version is used

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
*See [Configuration System](architecture.md#configuration-system) for available dataset options*

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