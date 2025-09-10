# Technologies and Development Setup

**Created**: 2025-09-06
**Last Updated**: 2025-09-07
**Version**: 1.0.0
**Author**: Kilo Code

## Core Technologies

### Python Ecosystem
- **Python 3.10+**: Primary programming language
- **PyTorch**: Deep learning framework for model training
- **Hugging Face Transformers**: Model loading and inference
- **LeRobot**: Robotics data handling and model integration

### Federated Learning Framework
- **Flower Framework**: Federated learning orchestration and aggregation
  - **Version**: Flower 1.20.0
  - **Architecture**: Client-Server with Deployment Engine
  - **Supported ML Frameworks**: PyTorch, TensorFlow, JAX, MLX, 🤗 Transformers, PyTorch Lightning, scikit-learn, XGBoost, fastai, Pandas
  - **Key Components**: ClientApp, ServerApp, Strategies (FedAvg, FedProx), Mods
  - **Execution Modes**: Simulation Mode, Deployment Mode, GPU Support

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

### Key Integration Points
1. **Client Implementation**: Extend NumPyClient for SmolVLA
2. **Server Strategy**: Use FedAvg or custom strategies
3. **Data Partitioning**: LeRobotDatasetPartitioner for episode-based splitting
4. **Model Aggregation**: Flower's parameter aggregation mechanisms

## Development Environment
- **Conda Environment**: "zk0" for isolated dependency management
- **VSCode**: IDE with automatic conda environment detection
- **Git**: Version control with GitHub integration
- **Pre-commit Hooks**: Code quality and formatting checks

## Dependencies Management
- **requirements.txt**: Pinned versions for reproducibility
- **pyproject.toml**: Project configuration and metadata
- **Conda Environment File**: Complete environment specification

## Hardware Requirements
- **Python Version**: 3.8+ (recommended 3.10)
- **PyTorch Version**: Compatible with SmolVLA requirements
- **CUDA Version**: 11.0+ for GPU support
- **Memory Requirements**: 8GB+ RAM, 4GB+ VRAM for GPU
- **GPU Support**: CUDA-compatible GPUs for model training
- **Storage**: Sufficient space for datasets and model checkpoints

## Testing and Quality Assurance
- **Test Coverage**: 80% minimum for new code
- **pytest**: Unit and integration testing framework
- **Coverage.py**: Test coverage measurement
- **Black**: Code formatting
- **mypy**: Type checking
- **Documentation**: Complete API documentation
- **Reproducibility**: Seeds for all experiments

## Deployment and Operations
- **Docker**: Containerization for reproducible deployments
- **Kubernetes**: Orchestration for distributed clients
- **Monitoring**: Logging and performance tracking
- **CI/CD**: Automated testing and deployment pipelines
- **Network Configuration**: Proper TLS and authentication setup
- **Scalability**: Plan for multiple clients and rounds

## Installation and Setup

### SmolVLA Installation
```bash
# Install LeRobot with SmolVLA dependencies from PyPI
pip install 'lerobot[smolvla]'
```

### Flower Installation
```bash
# Install Flower
pip install flwr[simulation]

# For latest features
pip install flwr[simulation] --pre
```

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

**Default Recommendation**: Use `local-simulation-gpu` for SmolVLA training as it matches the 4-client SO-100 dataset architecture and provides better GPU utilization. If GPU is not available, use `local-simulation` with `backend.client-resources.num-gpus=0`.

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
- **GitHub Repository**: https://github.com/adap/flower
- **Documentation**: https://flower.ai/docs/
- **Framework Hub**: https://flower.ai
- **General Examples**: https://github.com/adap/flower/tree/main/examples
- **LeRobot Integration Example**: https://flower.ai/docs/examples/quickstart-lerobot.html
- **LeRobot Example Repository**: https://github.com/adap/flower/tree/main/examples/quickstart-lerobot