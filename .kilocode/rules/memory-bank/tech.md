# Technologies and Development Setup

**Created**: 2025-09-06
**Last Updated**: 2025-09-06
**Version**: 1.0.0
**Author**: Kilo Code

## Core Technologies
- **Python 3.10+**: Primary programming language
- **Flower Framework**: Federated learning orchestration and aggregation
- **SmolVLA**: Vision-language-action models for robotics
- **SO-100 Datasets**: Real-world robotics training data
- **PyTorch**: Deep learning framework for model training
- **Hugging Face Transformers**: Model loading and inference
- **LeRobot**: Robotics data handling and model integration

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
- **GPU Support**: CUDA-compatible GPUs for model training
- **Memory**: 16GB+ RAM recommended for large models
- **Storage**: Sufficient space for datasets and model checkpoints

## Testing and Quality Assurance
- **pytest**: Unit and integration testing framework
- **Coverage.py**: Test coverage measurement
- **Black**: Code formatting
- **mypy**: Type checking

## Deployment and Operations
- **Docker**: Containerization for reproducible deployments
- **Kubernetes**: Orchestration for distributed clients
- **Monitoring**: Logging and performance tracking
- **CI/CD**: Automated testing and deployment pipelines