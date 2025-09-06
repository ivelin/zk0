# Flower Framework Specifications

**Created**: 2025-09-06
**Last Updated**: 2025-09-06
**Version**: 1.0.0
**Author**: Kilo Code

## Framework Version
- **Current Version**: Flower 1.20.0
- **Framework Type**: Federated Learning Framework
- **Architecture**: Client-Server with Deployment Engine

## Supported ML Frameworks
- PyTorch
- TensorFlow
- JAX
- MLX
- 🤗 Transformers
- PyTorch Lightning
- scikit-learn
- XGBoost
- fastai
- Pandas

## Key Components
- **ClientApp**: Defines client-side federated learning logic
- **ServerApp**: Defines server-side aggregation logic
- **Strategies**: Federated learning algorithms (FedAvg, FedProx, etc.)
- **Mods**: Built-in modifications for differential privacy, compression, etc.

## Execution Modes
- **Simulation Mode**: Local testing and development
- **Deployment Mode**: Production distributed execution
- **GPU Support**: Configurable GPU resource allocation

## Installation and Setup
```bash
# Install Flower
pip install flwr[simulation]

# For latest features
pip install flwr[simulation] --pre
```

## Basic Usage
```bash
# Run simulation
flwr run .

# Run with GPU federation
flwr run . local-simulation-gpu

# Override configuration
flwr run . local-simulation-gpu --run-config "num-server-rounds=5 fraction-fit=0.1"