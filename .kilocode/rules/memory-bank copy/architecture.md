# System Architecture

**Created**: 2025-09-06
**Last Updated**: 2025-09-07
**Version**: 1.0.0
**Author**: Kilo Code

## Source Code Paths
- **Client Application**: [`src/client_app.py`](src/client_app.py)
- **Server Application**: [`src/server_app.py`](src/server_app.py)
- **Configuration Files**:
  - [`src/configs/default.yaml`](src/configs/default.yaml)
  - [`src/configs/policy/vla.yaml`](src/configs/policy/vla.yaml)
- **Test Suite**: [`tests/`](tests/)
  - Unit tests: [`tests/unit/`](tests/unit/)
  - Integration tests: [`tests/integration/`](tests/integration/)

## Overview
The system implements a federated learning architecture using the Flower framework with SmolVLA models for robotics AI tasks. The architecture follows a client-server model where multiple clients train models locally on their private datasets and a central server coordinates the federated learning process.

## Core Components

### Client Layer
- **SmolVLA Models**: Vision-language-action models for robotics manipulation
- **Local Datasets**: SO-100 real-world robotics datasets
- **Training Logic**: Local model training with privacy preservation
- **Parameter Exchange**: Secure communication with central server

### Server Layer
- **Aggregation Engine**: Flower framework for parameter aggregation
- **Federated Strategies**: FedAvg and FedProx algorithms
- **Model Distribution**: Broadcasting updated global models to clients
- **Orchestration**: Managing federated learning rounds and client coordination

### Communication Layer
- **Secure Channels**: Encrypted parameter transmission
- **Asynchronous Updates**: Support for clients joining/leaving dynamically
- **Bandwidth Optimization**: Efficient parameter compression and transmission

## Technical Decisions

### Framework Selection
- **Flower Framework**: Chosen for its simplicity, scalability, and PyTorch integration
- **SmolVLA Integration**: Direct compatibility with LeRobot ecosystem
- **SO-100 Datasets**: Standardized robotics data format

### Architecture Patterns
- **Client-Server Model**: Standard federated learning topology
- **Parameter Server**: Centralized aggregation for efficiency
- **Modular Design**: Separable client and server components

### Scalability Considerations
- **Horizontal Scaling**: Support for multiple clients
- **Resource Management**: GPU allocation and memory optimization
- **Fault Tolerance**: Handling client failures and network issues

## Data Flow

1. **Initialization**: Server distributes initial SmolVLA model to clients
2. **Local Training**: Clients train on local SO-100 datasets
3. **Parameter Upload**: Clients send model updates to server
4. **Aggregation**: Server combines updates using federated strategies
5. **Model Update**: Server broadcasts improved global model
6. **Iteration**: Process repeats for multiple rounds

## Security Architecture

- **Privacy Preservation**: No raw data leaves client environments
- **Encrypted Communication**: TLS for all client-server interactions
- **Access Control**: Authentication and authorization mechanisms
- **Audit Logging**: Comprehensive logging for compliance

## Performance Considerations

### SmolVLA Performance
- **SO-100 Success Rate**: 78.3% (with community pretraining)
- **SO-101 Generalization**: Strong transfer capabilities
- **Simulation Benchmarks**: Matches/exceeds larger VLAs on LIBERO, Meta-World
- **Real-world Tasks**: Pick-place, stacking, sorting, tool manipulation

### Flower Performance
- **Scalability**: Supports 10+ clients in simulation
- **Communication Efficiency**: Optimized parameter transmission
- **GPU Utilization**: Efficient resource allocation
- **Memory Management**: Streaming and batch processing

### System Performance
- **Asynchronous Inference**: 30% faster response, 2× task throughput
- **Resource Management**: GPU allocation and memory optimization for distributed training
- **Fault Tolerance**: Handling client failures and network issues
- **Bandwidth Optimization**: Efficient parameter compression and transmission