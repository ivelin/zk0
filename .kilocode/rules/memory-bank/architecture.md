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

## Training Strategy

### Federated Learning Setup
The system implements a carefully designed federated learning strategy to ensure robust, privacy-preserving training of SmolVLA models across distributed clients.

#### Client Task Assignments
Each client is assigned a unique robotics manipulation task to prevent data overlap and ensure diverse skill learning:

- **Client 0**: Stacking manipulation ([`lerobot/svla_so100_stacking`](https://huggingface.co/datasets/lerobot/svla_so100_stacking))
  - Task: Stack objects of different shapes and sizes
  - Dataset Size: High-quality episodes with varied object configurations
  - Unique Aspect: Focus on precise placement and balance

- **Client 1**: Sorting manipulation ([`lerobot/svla_so100_sorting`](https://huggingface.co/datasets/lerobot/svla_so100_sorting))
  - Task: Sort objects by color, shape, or type
  - Dataset Size: Comprehensive sorting scenarios
  - Unique Aspect: Classification and organization skills

- **Client 2**: Pick and place operations (Future: `lerobot/svla_so100_pick_place`)
  - Task: Pick objects from various locations and place them precisely
  - Dataset Size: Diverse pick-and-place scenarios
  - Unique Aspect: Trajectory planning and grasping

- **Client 3**: Tool manipulation (Future: `lerobot/svla_so100_tool_use`)
  - Task: Use tools to manipulate objects (e.g., screwdrivers, pliers)
  - Dataset Size: Tool-based manipulation episodes
  - Unique Aspect: Tool affordance understanding

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
The server evaluates the global model on all client tasks as well as additional unseen tasks to verify generalization capabilities:

- **SO-101 Test Dataset** ([choyf3/so101_test_20250908](https://lerobot-visualize-dataset.hf.space/choyf3/so101_test_20250908/episode_0))
  - Task: Advanced SO-101 manipulation testing
  - Purpose: Verify generalization to newer robot platform

- **Griffin Labs Trial** ([griffinlabs/record-trial-2](https://lerobot-visualize-dataset.hf.space/griffinlabs/record-trial-2/episode_1))
  - Task: Research laboratory manipulation scenarios
  - Purpose: Test performance on specialized research tasks

- **SO-101 Test Suite** ([JamesChen007/so101-test](https://lerobot-visualize-dataset.hf.space/JamesChen007/so101-test/episode_0))
  - Task: Comprehensive SO-101 evaluation scenarios
  - Purpose: Validate cross-platform generalization

- **SO-101 Grab Task** ([chrvngr/so101_grab](https://lerobot-visualize-dataset.hf.space/chrvngr/so101_grab/episode_0))
  - Task: Object grasping and manipulation
  - Purpose: Test fundamental grasping capabilities

- **Inference Recording Test** ([zacapa/infer_rec_test](https://lerobot-visualize-dataset.hf.space/zacapa/infer_rec_test/episode_3))
  - Task: Inference and recording validation
  - Purpose: Verify real-time performance metrics

- **Recording Test Suite** ([tinjyuu/record-test6](https://lerobot-visualize-dataset.hf.space/tinjyuu/record-test6/episode_0))
  - Task: Comprehensive recording and playback scenarios
  - Purpose: Test data collection and replay capabilities

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