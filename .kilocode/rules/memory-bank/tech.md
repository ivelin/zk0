# Technologies and Development Setup

**Created**: 2025-09-06
**Last Updated**: 2025-09-06
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
- **SO-100 Datasets**: Real-world robotics training data
- **Format**: LeRobot format with lerobot tag
- **Annotation**: Clear task descriptions (max 30 characters)
- **Camera Views**: Standardized naming (OBS_IMAGE_1, OBS_IMAGE_2, etc.)
- **Frame Rate**: 30 FPS for SO-100/SO-101

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
  --dataset.repo_id=lerobot/svla_so100_stacking \
  --batch_size=64 \
  --steps=20000 \
  --output_dir=outputs/train/my_smolvla \
  --job_name=my_smolvla_training \
  --policy.device=cuda \
  --wandb.enable=true
```

#### Flower Simulation
```bash
# Run simulation
flwr run .

# Run with GPU federation
flwr run . local-simulation-gpu

# Override configuration
flwr run . local-simulation-gpu --run-config "num-server-rounds=5 fraction-fit=0.1"
```

## Official Resources

### SmolVLA Resources
- **Blog Post**: https://huggingface.co/blog/smolvla
- **Model Hub**: https://huggingface.co/lerobot/smolvla_base
- **Paper**: https://huggingface.co/papers/2506.01844
- **Code**: https://github.com/huggingface/lerobot

### Flower Resources
- **Documentation**: https://flower.ai/docs/framework/
- **Framework Hub**: https://flower.ai
- **GitHub**: https://github.com/adap/flower
- **Examples**: https://github.com/adap/flower/tree/main/examples