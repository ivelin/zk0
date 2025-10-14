# System Architecture

**Created**: 2025-09-06
**Last Updated**: 2025-10-14
**Version**: 1.0.2
**Author**: Kilo Code

## Source Code Paths
- **Client Application**: [`src/client_app.py`](src/client_app.py)
- **Server Application**: [`src/server_app.py`](src/server_app.py)

## Directory Structure

```
zk0/
├── src/                          # Source code
│   ├── __init__.py
│   ├── client_app.py            # Flower client implementation
│   ├── server_app.py            # Flower server implementation
│   ├── task.py                  # Model training and evaluation functions
│   ├── utils.py                 # Utility functions (model loading, dataset handling)
│   ├── logger.py                # Logging configuration
│   ├── visualization.py         # Evaluation visualization tools
│   ├── wandb_utils.py           # Weights & Biases integration
│   └── configs/                 # Configuration management
│       ├── __init__.py
│       └── datasets.py          # Dataset configuration
├── tests/                       # Test suite
│   ├── __init__.py
│   ├── conftest.py              # Pytest configuration
│   ├── unit/                    # Unit tests
│   │   ├── test_basic_functionality.py
│   │   ├── test_dataset_loading.py
│   │   ├── test_dataset_splitting.py
│   │   ├── test_dataset_validation.py
│   │   ├── test_error_handling.py
│   │   ├── test_logger.py
│   │   ├── test_model_loading.py
│   │   └── test_smolvla_client.py
│   └── integration/             # Integration tests
│       ├── __init__.py
│       └── test_integration.py
├── outputs/                     # Runtime outputs (created during execution)
│   └── [timestamp]/             # Timestamped run directories
│       ├── simulation.log       # Unified logging output
│       ├── server/              # Server-side outputs
│       │   ├── server.log       # Server-specific logs
│       │   ├── round_N_server_eval.json  # Server evaluation results
│       │   ├── federated_metrics.json     # Aggregated FL metrics
│       │   ├── federated_metrics.png      # Metrics visualization
│       │   ├── eval_policy_loss_chart.png  # Policy loss chart
│       │   └── eval_policy_loss_history.json # Historical policy loss data
│       ├── clients/              # Client-side outputs
│       │   └── client_N/         # Per-client directories
│       │       ├── client.log    # Client-specific logs
│       │       └── round_N.json  # Client evaluation metrics
│       └── models/               # Saved model checkpoints
│           └── checkpoint_round_N.safetensors
├── .kilocode/                   # Kilo Code configuration
│   └── rules/
│       └── memory-bank/         # Memory bank documentation
├── pyproject.toml               # Project configuration
├── requirements.txt             # Python dependencies
├── train.sh                     # Docker training script
├── train-lerobot-standalone.sh  # Standalone training script
├── Dockerfile                   # Docker container definition
├── LICENSE                      # License file
├── README.md                    # Project documentation
├── CONTRIBUTING.md              # Contribution guidelines
├── datasets.yaml                # Dataset metadata
├── .gitignore                   # Git ignore patterns
├── .env.example                 # Environment variables template
└── CNAME                        # GitHub Pages domain
```

## Configuration System

**Centralized Configuration Architecture:**
- **`pyproject.toml`** - Primary configuration file containing:
  - `[tool.flwr.app.config]` - Flower federated learning parameters (rounds, epochs, strategies, early stopping, model checkpointing)
  - `[tool.zk0.datasets]` - Client dataset assignments and evaluation configurations
  - `[tool.zk0.logging]` - Application logging configuration
  - `[project]` - Project metadata and dependencies
- **`.env`** - Environment variables for sensitive configuration (API keys, paths)
- **Code defaults** - Fallback values in source code for robustness

**Configuration Loading:**
- Dataset configuration loaded via `src/configs/datasets.py` → `DatasetConfig.load()`
- Flower configuration loaded via `src/utils.py` → `get_tool_config()`
- Environment variables loaded via `python-dotenv`

**Key Configuration Parameters:**
- **Federated Learning**: `num-server-rounds`, `local-epochs`, `fraction-fit`, `fraction-evaluate`, `batch_size`
- **Model Settings**: `model-name`, `proximal_mu`, `initial_lr`, `server-device`
- **Evaluation**: `eval-frequency`, `eval_batches`, `early_stopping_patience`
- **Experiment Tracking**: `use-wandb`, `hf_repo_id`, `checkpoint_interval`
- **Advanced Features**: `dynamic_lr_enabled` for adaptive learning rate adjustment
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
Each client is assigned a unique robotics manipulation task to prevent data overlap and ensure diverse skill learning. See [Configuration System](#configuration-system) for complete client dataset configuration including:
- **4 validated clients** with diverse robotics manipulation tasks:
  - **Client 0**: shaunkirby/record-test - "Put the red LEGO in the bin"
  - **Client 1**: ethanCSL/direction_test - "turn to the right side" (VALIDATED CLEAN)
  - **Client 2**: gimarchetti/so101-winnie-us5 - "rub the plush toy with the bottle" (VALIDATED CLEAN)
  - **Client 3**: olingoudey/so101_stationary_mug445 - "Put the stuffed animal in the mug" (VALIDATED CLEAN)
- **Dataset sizes and validation status** for each client
- **Train/eval episode splits** for proper federated learning setup
- **Quality assurance indicators** (CLEAN vs HOTFIX APPLIED)

#### Data Quality and Uniqueness Requirements
- **High-Quality Datasets**: All datasets must contain clear, well-annotated episodes
- **Unique Tasks**: No task overlap between clients to ensure diverse skill acquisition
- **Fresh Data**: None of the datasets used for training the base SmolVLA model
- **Evaluation Isolation**: Separate evaluation datasets never seen during training

#### Evaluation Strategy
- **Server-Side Evaluation**: Global model evaluated server-side using dedicated evaluation datasets
- **Unseen Task Evaluation**: Evaluation on completely novel SO-100/SO-101 tasks not seen during training
- **Data Leak Prevention**: Strict validation to ensure no evaluation data in training sets
- **Episode Limiting**: Evaluation limited to first N episodes from evaluation datasets

#### Server Evaluation Datasets (Unseen Tasks)
The server evaluates the global model on all client tasks as well as additional unseen tasks to verify generalization capabilities. See [Configuration System](#configuration-system) for the complete list of validated evaluation datasets including:
- **Hupy440/Two_Cubes_and_Two_Buckets_v2**: SO-101 test dataset (12 episodes, 9,128 frames) - "Pick up a cube. Is the cube red, put it in the white bucket. Is the cube white, put it in the red bucket."
- **SO-101 cross-platform datasets** for generalization testing
- **Research laboratory scenarios** for specialized task validation
- **Comprehensive test suites** for thorough evaluation
- **Real-time performance datasets** for inference validation

#### Evaluation Metrics
- **Policy Loss**: Average policy forward loss per batch (same as client training loss, ~1 scale)
- **Action MSE**: Raw mean squared error on predicted vs ground truth actions (reference metric, ~1700 scale)
- **Generalization Score**: Performance on unseen evaluation tasks
- **Cross-Task Performance**: Average performance across all evaluation datasets

## Data Flow

1. **Initialization**: Server distributes initial SmolVLA model to clients
2. **Task Assignment**: Each client receives unique task dataset for training
3. **Local Training**: Clients train on their assigned SO-100 task datasets (all episodes used)
4. **Parameter Upload**: Clients send model updates to server
5. **Aggregation**: Server combines updates using federated strategies
6. **Model Update**: Server broadcasts improved global model
7. **Server Evaluation**: Global model evaluated server-side on dedicated evaluation datasets
8. **Iteration**: Process repeats for multiple rounds with performance tracking

## Security Architecture

- **Privacy Preservation**: No raw data leaves client environments
- **Encrypted Communication**: TLS for all client-server interactions
- **Access Control**: Authentication and authorization mechanisms
- **Audit Logging**: Comprehensive logging for compliance
- **Parameter Validation**: Bidirectional SHA256 hash checking ensures parameter integrity
- **Corruption Detection**: Automatic detection and exclusion of corrupted client updates
- **Fail-Fast Security**: Runtime errors raised immediately on validation failures

## Integration Requirements

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

## Performance Benchmarks

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

## Project Constraints

**Summary:** Work in root dir, use zk0 conda env, focus on SmolVLA/Flower/SO-100
⚠️ **MANDATORY**: These constraints must be included in EVERY task, subtask, and mode transition:

### 1. Working Directory Constraint
- **Write Location**: File modifications and changes only within the local project repository root directory (/home/ivelin/zk0)
- **Read Access**: Read operations allowed in all workspace folders, including lerobot and flower
- **No External Changes**: No modifications to sibling or parent directories outside the project root
- **Scope Limitation**: All development work must remain within the project root directory

### 2. Environment Requirements
- **Primary Environment**: Docker container (`zk0`) via train.sh for reproducible, isolated execution of training and simulations
- **Alternative Environment**: Conda environment "zk0" for local development and training runs (validated for federated learning execution)
- **VSCode Integration**: VSCode with Docker integration and automatic environment detection
- **Training Script**: Use `./train.sh` for Docker-based executions or `conda run -n zk0 flwr run . local-simulation-serialized-gpu` for conda-based executions
- **Dependencies**: Use pinned versions from `requirements.txt`

### 3. Technical Focus
- **Primary Model**: Focus on SmolVLA model integration
- **Dataset**: Use SO-100 real-world robotics datasets
- **Framework**: Flower federated learning framework
- **Reference Structure**: Borrow structure from quickstart-lerobot but adapt for SmolVLA requirements

### 4. Quality Standards
- **Testing**: Maintain comprehensive test coverage (80% minimum)
- **Documentation**: Keep README and documentation current
- **Reproducibility**: Ensure all experiments are reproducible with seeds

### 5. Production Code Quality Standards
- **NO MOCKS IN PRODUCTION**: Mocks, stubs, or fake implementations are STRICTLY FORBIDDEN in production code
- **Fail Fast Principle**: Production code must fail fast with clear error messages when real dependencies are unavailable
- **Real Dependencies Only**: All production components must work with actual dependencies and real data
- **Error Handling**: Implement proper exception handling that raises RuntimeError for missing dependencies
- **Testing Isolation**: Mocks are only acceptable in unit tests for dependency isolation, never in production paths
- **Code Review Requirement**: Any use of mocks in production code will be rejected during code review
- **No Simulation Fallbacks**: Production code must not fall back to simulated training or evaluation

### 6. Progress and Status Management Standards
- **Production Readiness Criteria**: Application code cannot be considered solid and production-ready if any tests fail
- **Project Status Assessment**: Current project status is alpha stage - active early development that is largely untested by real users, not even close to beta where other developers can test it and expect substantial features
- **Progress Declaration Policy**: Do not declare progress without explicit user approval
- **Version Increment Guidelines**: When substantial progress is approved, update memory bank with progress status and increment project version according to level of progress (minor, major, breaking, etc.)
- **Task Completion Assessment**: When a big task is completed that involves substantial code changes, assess and propose project progress update, but wait for approval before making any changes

### 7. Testing Execution Requirements
- **Environment**: All tests must run in Docker container (`zk0`) for consistency
- **Parallel Execution**: Use `pytest -n auto` for parallel test execution
- **Coverage**: Always include `--cov=src --cov-report=term-missing` for coverage reporting
- **Command Format**: Use train.sh or direct Docker run with `python -m pytest -n auto --cov=src --cov-report=term-missing`

### 8. Environment Dependency Management
- **Reproducible Dependencies**: When new OS-level or Python dependencies are needed, update Dockerfile and/or requirements.txt for reproducibility
- **Docker-first**: Prefer Docker-based dependency management over local environment modifications
- **Version Pinning**: Always pin dependency versions in requirements.txt to ensure consistent environments
- **Documentation**: Document any new dependencies and their purpose in commit messages and memory bank

### 9. Testing and CI Standards
- **NO STANDALONE TEST FILES**: If anything needs testing about runtime environment, dependencies, models, datasets, integration interfaces, or app logic, it MUST be part of the CI test suite, not standalone files
- **CI Test Suite Usage**: Anytime you want to check runtime environment correctness and readiness, run tests from the existing test suite
- **Reusable Test Code**: Do not write code outside of the test suite if it can be reused for CI needs
- **Test Suite Integration**: All environment validation, dependency checking, and integration testing must be integrated into the existing pytest test suite
- **Standalone File Prohibition**: Creating one-off test scripts for environment validation is STRICTLY FORBIDDEN - all such testing must be part of the structured test suite

## Workflow Rules

**Summary:** Preserve context across tasks, validate constraints, document decisions, optimize tool usage efficiency

### Task Inheritance
1. **Context Preservation**: All subtasks must inherit parent task context
2. **Constraint Propagation**: Key constraints must be explicitly passed down
3. **Success Criteria**: Clear success criteria for each subtask
4. **Documentation**: Document all assumptions and decisions

### Mode Transitions
1. **Context Transfer**: Preserve all critical context when switching modes
2. **Handover Documentation**: Document what is being handed over and why
3. **Validation**: Verify that target mode can handle the context
4. **Rollback Plan**: Have clear rollback procedures if needed

### Tool Usage Efficiency
1. **Multi-File Reads**: Always use multi-file reads via the `read_file` tool (up to 5 files) or gitingest to minimize round trips and improve efficiency
2. **Batch Operations**: Group related file reads together in single tool calls rather than sequential individual reads
3. **Context Gathering**: Read all necessary files upfront before making changes to ensure complete understanding
4. **Efficiency Priority**: Prefer tools that allow batch operations over multiple sequential calls to reduce latency and maintain focus
5. **Diff Format for Edits**: Always present code edits in diff format for easier review and approval. Use unified diff format with clear context showing what is being changed, added, or removed.

### Quality Assurance
1. **Pre-Check**: Validate constraints before starting work
2. **Progress Tracking**: Regular validation against success criteria
3. **Post-Validation**: Ensure deliverables meet requirements
4. **Documentation Update**: Update memory bank with lessons learned

### Context Management Rules
1. **Session Transitions**: Honor context window alerts and transition proactively
2. **Essential Preservation**: Always inherit critical technical context in new sessions
3. **Memory Bank Updates**: Update memory bank before major transitions
4. **Clear Documentation**: Document current state and next steps before transitioning
5. **Fresh Start Benefits**: Use transitions to maintain optimal performance and focus

## Emergency Procedures

**Summary:** Stop work if context lost, use memory bank to reconstruct

### If Context is Lost
1. **Immediate Stop**: Pause work and assess what was lost
2. **Memory Bank Reference**: Use this document to reconstruct context
3. **Gap Documentation**: Document what was missing for prevention
4. **Supervisor Notification**: Notify if critical context cannot be recovered

### If Constraints are Violated
1. **Stop Work**: Immediately cease the violating activity
2. **Impact Assessment**: Evaluate potential damage or inconsistencies
3. **Correction Plan**: Develop plan to return to compliant state
4. **Prevention Update**: Update procedures to prevent recurrence

## Validation Checklist

Before starting ANY task:
- [ ] Working directory is the project root directory
- [ ] Docker container (`zk0`) is available for reproducible execution
- [ ] VSCode has Docker integration enabled
- [ ] SmolVLA and SO-100 focus is maintained
- [ ] Reference to quickstart-lerobot structure is considered
- [ ] Parent task context is fully understood
- [ ] Success criteria are clearly defined
- [ ] Quality standards are referenced

Before running tests:
- [ ] Tests run in Docker container (`zk0`)
- [ ] Use parallel execution with `-n auto` for multiple tests
- [ ] Include coverage reporting with `--cov=src --cov-report=term-missing`
- [ ] Ensure coverage remains above 80%
- [ ] Use existing test suite instead of creating standalone test files

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