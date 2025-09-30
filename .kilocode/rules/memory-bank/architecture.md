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
  - [`src/configs/datasets.yaml`](src/configs/datasets.yaml) - Dataset configuration and validation
- **Test Suite**: [`tests/`](tests/)
  - Unit tests: [`tests/unit/`](tests/unit/)
  - Integration tests: [`tests/integration/`](tests/integration/)
  - Dataset validation: [`tests/unit/test_dataset_validation.py`](tests/unit/test_dataset_validation.py)

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
- **Federated Strategies**: FedProx algorithm (primary) with FedAvg as baseline
- **Model Distribution**: Broadcasting updated global models to clients
- **Orchestration**: Managing federated learning rounds and client coordination

#### Strategy Decision: FedProx for Heterogeneous Convergence
- **Primary Strategy**: FedProx selected for heterogeneous SO-100 data handling
- **Rationale**: Addresses non-IID data drift with proximal regularization (mu * ||w_local - w_global||^2)
- **Implementation**: Client-side proximal term in training loop, server-side standard aggregation
- **Benefits**: Stabilizes convergence, improves global model quality, minimal code changes
- **Configuration**: mu=0.01, applied only to trainable parameters (~100M/450M)

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

## Training Strategy

### Federated Learning Setup
The system implements a carefully designed federated learning strategy to ensure robust, privacy-preserving training of SmolVLA models across distributed clients.

#### Client Task Assignments
Each client is assigned a unique robotics manipulation task to prevent data overlap and ensure diverse skill learning. See [`src/configs/datasets.yaml`](src/configs/datasets.yaml) for complete client dataset configuration including:
- **4 validated clients** with diverse robotics manipulation tasks
- **Dataset sizes and validation status** for each client
- **Train/eval episode splits** for proper federated learning setup
- **Quality assurance indicators** (CLEAN vs HOTFIX APPLIED)

#### Data Quality and Uniqueness Requirements
- **High-Quality Datasets**: All datasets must contain clear, well-annotated episodes
- **Unique Tasks**: No task overlap between clients to ensure diverse skill acquisition
- **Fresh Data**: None of the datasets used for training the base SmolVLA model
- **Evaluation Isolation**: Separate evaluation datasets never seen during training

#### Evaluation Strategy
- **Client-Specific Evaluation**: Each client evaluated on their assigned task
- **Cross-Task Evaluation**: Global model evaluated on all client tasks
- **Unseen Task Evaluation**: Additional evaluation on completely novel SO-100/SO-101 tasks
- **Data Leak Prevention**: Strict validation to ensure no evaluation data in training sets

#### Server Evaluation Datasets (Unseen Tasks)
The server evaluates the global model on all client tasks as well as additional unseen tasks to verify generalization capabilities. See [`src/configs/datasets.yaml`](src/configs/datasets.yaml) for the complete list of validated evaluation datasets including:
- **SO-101 cross-platform datasets** for generalization testing
- **Research laboratory scenarios** for specialized task validation
- **Comprehensive test suites** for thorough evaluation
- **Real-time performance datasets** for inference validation

#### Evaluation Metrics
- **Task Success Rate**: Percentage of successfully completed episodes
- **Action Accuracy**: Precision of predicted vs. ground truth actions
- **Generalization Score**: Performance on unseen evaluation tasks
- **Cross-Task Performance**: Average performance across all evaluation datasets

## Data Flow

1. **Initialization**: Server distributes initial SmolVLA model to clients
2. **Task Assignment**: Each client receives unique task dataset for training
3. **Local Training**: Clients train on their assigned SO-100 task datasets
4. **Parameter Upload**: Clients send model updates to server
5. **Aggregation**: Server combines updates using federated strategies
6. **Model Update**: Server broadcasts improved global model
7. **Cross-Evaluation**: Global model evaluated on all client tasks plus unseen tasks
8. **Iteration**: Process repeats for multiple rounds with performance tracking

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

## Improvement Suggestions from Pusht Example Analysis

Based on analysis of the Flower LeRobot pusht example, the following improvements are recommended for zk0 architecture:

1. **Modular Task Separation**: Adopt a separate task.py file for model initialization, training, and evaluation logic to improve code organization and maintainability, similar to the example.
2. **Standardized Partitioning**: Integrate Flower Datasets with GroupedNaturalIdPartitioner for episode partitioning to leverage built-in functionality alongside zk0's multi-repo support.
3. **Gym Integration for Evaluation**: Add gym-based rollout evaluation with video rendering for standardized testing, complementing existing SmolVLAEvaluator.
4. **Config Flexibility**: Enhance YAML config handling to match the example's structure, including support for image transforms and online training parameters.
5. **GPU and AMP Support**: Implement Automatic Mixed Precision (AMP) and better GPU handling as in the example for improved training efficiency.
6. **Output Management**: Standardize output directories for models, evaluations, and videos across clients and server.

These changes would align zk0 more closely with LeRobot best practices while maintaining SmolVLA focus.