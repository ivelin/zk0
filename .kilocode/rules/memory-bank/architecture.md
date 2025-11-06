# System Architecture

**Created**: 2025-09-06
**Last Updated**: 2025-11-06
**Version**: 1.0.8
**Author**: Kilo Code

## Source Code Paths

For key source code locations and complete directory structure, see [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md).

## Configuration System

**Centralized Configuration Architecture:**
- **`pyproject.toml`** - Primary configuration file containing:
  - `[tool.flwr.app.config]` - Flower federated learning parameters (rounds, epochs, strategies, model checkpointing)
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
- **Evaluation**: `eval-frequency`, `eval_batches`
- **Experiment Tracking**: `use-wandb`, `hf_repo_id`, `checkpoint_interval`
- **Advanced Features**: `dynamic_lr_enabled` for adaptive learning rate adjustment (early stopping removed; runs complete to configured num-server-rounds)

## Overview
The system implements a federated learning architecture using the Flower framework with SmolVLA models for robotics AI tasks. The architecture follows a client-server model where multiple clients train models locally on their private datasets and a central server coordinates the federated learning process. For detailed overview, see [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md).

## Core Components
For detailed core components, see [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md).

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
For detailed technical decisions including framework selection and scalability, see [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md).

#### Hugging Face Hub Push Optimization
- **Conditional Model Push**: HF Hub push is skipped if `num-server-rounds < checkpoint_interval` to avoid uploading incomplete or debug models from short runs (e.g., tiny tests with 1-10 rounds).
- **Rationale**: Prevents repository clutter with low-value checkpoints; ensures only meaningful training runs (e.g., ≥20 rounds) contribute to the shared model.
- **Implementation**: Checked in `src/server_app.py` at the end of `aggregate_fit()`; logs the skip decision for transparency.
- **Configuration**: Controlled via `pyproject.toml` (`num-server-rounds`, `checkpoint_interval=20`); always saves local checkpoints regardless.

### Enhanced HF Hub Push (v0.3.7)
- **Rich Model Cards**: Automatically generated README.md with comprehensive training details including hyperparameters, datasets used, final evaluation metrics, training insights, and usage instructions.
- **Dynamic Content Extraction**: Pulls data from pyproject.toml configs, JSON evaluation files, federated metrics, and policy loss history for complete documentation.
- **Git Tagging**: Creates local git tags (`fl-run-{timestamp}-v{version}`) and HF repo tags (`fl-round-{num_rounds}`) for version tracking.
- **Simulation Mode Detection**: Conditionally includes simulation training notes in model cards when running in local-simulation mode.
- **Enhanced Commit Messages**: Includes final loss and client count in push commit messages for better traceability.
- **Fallback Handling**: Gracefully handles missing metrics files with sensible defaults; validates generated content for parsing safety.
- **Implementation**: New `push_model_to_hub_enhanced()` function in `src/server/server_utils.py` with helper functions for data extraction and model card generation.

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
For detailed training strategy including data flow and evaluation, see [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md).

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
- **Dependencies**: Use pinned versions from `pyproject.toml`

### 3. Technical Focus
- **Primary Model**: Focus on SmolVLA model integration
- **Dataset**: Use SO-100 real-world robotics datasets
- **Framework**: Flower federated learning framework
- **Reference Structure**: Borrow structure from quickstart-lerobot but adapt for SmolVLA requirements

### 4. Quality Standards
- **Testing**: Maintain meaningful test coverage (focus on critical paths, current target 30% minimum)
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
- **Version Synchronization**: Project version should always be in sync with release tags and git tags

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
- [ ] Ensure coverage remains above 30% (current target)
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

## Code Organization and Refactoring

**Refactoring Overview (2025-10-27)**: To address code bloat in src/server_app.py (1766 lines), a comprehensive refactoring was performed focusing on modularity, testability, and maintainability. The primary goal was to reduce the size of the AggregateEvaluationStrategy class (especially the 266-line aggregate_fit() method) while leveraging existing utilities in src/server/server_utils.py.

### Key Changes
1. **Modular Breakdown of aggregate_fit()**:
   - **validate_client_parameters()**: Already extracted; handles hash validation and filtering of corrupted updates.
   - **aggregate_parameters()**: New function in server_utils.py for core FedProx aggregation logic (moved from super().aggregate_fit() call).
   - **aggregate_and_log_metrics()**: New function to handle client metrics aggregation, norm computation, and logging.
   - **save_and_push_model()**: New function to manage checkpoint saving and conditional HF Hub push (combines save_model_checkpoint() and push_model_to_hub() logic).
   - **finalize_round_metrics()**: New function for merging metrics, adding diagnostics (mu/LR), and returning the final tuple.

2. **Leveraging server_utils.py**:
   - Moved aggregate_client_metrics(), collect_individual_client_metrics(), and compute_server_param_update_norm() usage is now central; added new functions like aggregate_parameters() and save_and_push_model() to this module.
   - Reduced redundancy: Removed duplicated hash computation/validation logic by centralizing in validate_and_log_parameters().

3. **Dangling/Unused Code Removal**:
   - Removed unused imports (e.g., redundant flwr.common imports).
   - Eliminated dangling methods: configure_evaluate() (returns empty list, but kept for Flower compatibility; no active logic).
   - Trimmed verbose logging in aggregate_fit() by extracting to dedicated log functions.

4. **Benefits**:
   - **Reduced Size**: aggregate_fit() now ~80 lines (70% reduction); server_app.py total ~1200 lines.
   - **Testability**: Smaller functions enable unit testing (e.g., test_aggregate_parameters(), test_save_and_push_model()).
   - **Maintainability**: Clear separation of concerns; easier to extend (e.g., new strategies).
   - **No Functional Changes**: Preserves all behavior, including conditional HF push, security validation, and metrics logging.

### Implementation Guidelines
- All new functions in server_utils.py follow existing patterns (type hints, docstrings, logging).
- Tests updated in tests/unit/test_server_app.py to cover new functions (coverage remains >80%).
- Version bump: v0.3.6 (minor refactor, no breaking changes).

This refactoring aligns with modular design principles while maintaining SmolVLA/Flower/SO-100 focus.

## Improvement Suggestions from Pusht Example Analysis

Based on analysis of the Flower LeRobot pusht example, the following improvements are recommended for zk0 architecture:

1. **Modular Task Separation**: Adopt a separate task.py file for model initialization, training, and evaluation logic to improve code organization and maintainability, similar to the example.
2. **Standardized Partitioning**: Integrate Flower Datasets with GroupedNaturalIdPartitioner for episode partitioning to leverage built-in functionality alongside zk0's multi-repo support.
3. **Gym Integration for Evaluation**: Add gym-based rollout evaluation with video rendering for standardized testing, complementing existing SmolVLAEvaluator.
4. **Config Flexibility**: Enhance YAML config handling to match the example's structure, including support for image transforms and online training parameters.
5. **GPU and AMP Support**: Implement Automatic Mixed Precision (AMP) and better GPU handling as in the example for improved training efficiency.
6. **Output Management**: Standardize output directories for models, evaluations, and videos across clients and server.

These changes would align zk0 more closely with LeRobot best practices while maintaining SmolVLA focus.

For hyperparameter analysis, see [docs/HYPERPARAMETER_ANALYSIS.md](../docs/HYPERPARAMETER_ANALYSIS.md).

For core architecture details, see [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md).