# Repetitive Task Workflows

**Created**: 2025-09-06
**Last Updated**: 2025-11-01
**Version**: 1.0.3
**Author**: Kilo Code

## Latest Update (2025-11-06)
**✅ SEO Quick Wins Implementation**: Enhanced zk0.bot Jekyll site discoverability with auto-generated sitemaps, Open Graph/Twitter Cards, schema markup, front matter optimizations, and internal linking. Jekyll build verified with 36.73% test coverage maintained. Version bumped to 0.5.0.

**✅ CI Workflow Consolidation**: Consolidated separate unit and integration test jobs into a single matrix job for cleaner GitHub Actions workflow. Removed redundant tee piping, test-output.log artifacts, and standardized on Python 3.10. Fixed lerobot installation for CI compatibility. Version bumped to 0.3.11.

**✅ Enhanced HF Model Push Workflow**: Implemented comprehensive HF Hub push enhancements including rich model cards, git tagging, dynamic README generation, and simulation mode detection. New `push_model_to_hub_enhanced()` function extracts data from configs/JSON outputs, generates detailed model cards with hyperparameters/datasets/metrics/insights, creates local/HF git tags, and handles edge cases. Updated memory bank and added comprehensive unit tests. Version bumped to 0.3.7.

**✅ Standalone Model Push Module Migration**: Migrated standalone push_to_hf.py to src/zk0/push_to_hf.py module for proper package integration. Updated imports to use absolute package paths (zk0.server.server_utils, zk0.utils), removed shebang for module execution, added console_scripts entry point in pyproject.toml, updated README.md documentation, and removed standalone file. Module can now be executed via `python -m zk0.push_to_hf` with full functionality preserved.

## Critical Bug Fixes

### Aggregate Fit Parameter Safety Fix
**Last performed:** 2025-10-14
**Context:** Fixed critical bug where aggregate_fit() could return None parameters, causing `TypeError: cannot unpack non-iterable NoneType object` in Flower's server unpacking.

**Files modified:** `src/server_app.py`

**Steps:**
1. **Issue Detection**: Tiny training runs crashed with `TypeError: cannot unpack non-iterable NoneType object` in Flower server unpacking
2. **Root Cause Analysis**: aggregate_fit() could return None for aggregated_parameters in edge cases (early stopping, no clients, aggregation failures)
3. **Fix Implementation**: Added critical safety check at end of aggregate_fit():
   ```python
   # CRITICAL: Always return valid parameters tuple to prevent Flower unpacking errors
   # This fixes the "cannot unpack non-iterable NoneType object" error
   if aggregated_parameters is None:
       logger.warning(
           f"⚠️ Server: No parameters aggregated for round {server_round}, returning initial parameters"
       )
       aggregated_parameters = self.initial_parameters

   return aggregated_parameters, metrics
   ```
4. **Parameter Storage**: Ensured `self.initial_parameters` is available as fallback (stored in __init__)
5. **Validation**: Fixed code ensures Flower always receives valid `(parameters, metrics)` tuple
6. **Testing**: Next tiny training run should complete without unpacking errors

**Benefits:**
- Stable training that doesn't crash Flower server
- Reliable parameter handling for all run lengths
- Maintains model quality by returning best available parameters
- Prevents data loss from unexpected crashes during training

## Additional Critical Fixes

### Client Parameter Type Handling Fix
**Last performed:** 2025-10-10
**Context:** Fixed client parameter compatibility issues with Flower framework by ensuring proper NumPy array handling.

**Files modified:** `src/client_app.py`

**Steps:**
1. **Issue Detection**: Flower client failing with parameter type errors during federated rounds
2. **Root Cause Analysis**: SmolVLA parameters not properly converted to Flower-compatible NumPy arrays
3. **Fix Implementation**: Added parameter type conversion in client fit() method
4. **Validation**: Client successfully participates in FL rounds without type errors

### Server Eval Mode Passing Fix
**Last performed:** 2025-10-12
**Context:** Fixed server evaluation to properly pass eval_mode for full/quick evaluation modes.

**Files modified:** `src/server_app.py`

**Steps:**
1. **Issue Detection**: Server evaluation not using correct evaluation mode
2. **Fix Implementation**: Updated _server_evaluate() to pass eval_mode parameter
3. **Benefits**: Proper evaluation mode selection for different FL phases

### Full Evaluation Episode Limits Fix
**Last performed:** 2025-10-13
**Context:** Fixed evaluation to use dataset.episodes count for proper episode limiting.

**Files modified:** `src/evaluation.py`

**Steps:**
1. **Issue Detection**: Evaluation not respecting dataset episode limits
2. **Fix Implementation**: Updated episode limiting logic to use dataset.episodes
3. **Benefits**: Accurate evaluation on correct number of episodes

### Enhanced Security with Bidirectional SHA256 Validation
**Last performed:** 2025-10-14
**Context:** Implemented bidirectional SHA256 parameter validation for enhanced security.

**Files modified:** `src/server_app.py`, `src/client_app.py`

**Steps:**
1. **Implementation**: Added SHA256 hash validation in both directions
2. **Server Validation**: Computes and validates client parameter hashes
3. **Client Validation**: Validates received server parameters
4. **Error Handling**: Raises RuntimeError on hash mismatches
5. **Benefits**: Prevents corrupted parameter transmission, ensures data integrity

## Consolidated Metrics Implementation
**Last performed:** 2025-10-13
**Context:** Consolidated client metrics into server evaluation files for unified reporting and analysis.

**Files modified:** `src/server_app.py`, `src/client_app.py`, `pyproject.toml`

**Steps:**
1. **Version Increment**: Updated project version from 0.1.15 to 0.1.19 in pyproject.toml
2. **Server Aggregation Storage**: Modified `aggregate_fit()` in server_app.py to store individual client metrics (client_id, loss, fedprox_loss, grad_norm, param_hash, dataset_name) in `self.last_client_metrics`
3. **Server Eval Consolidation**: Updated `_server_evaluate()` to include both `aggregated_client_metrics` and `individual_client_metrics` in round_X_server_eval.json files
4. **Client Dataset Reporting**: Added `dataset_name` to client fit() metrics in client_app.py for server-side aggregation
5. **Documentation Updates**: Enhanced metrics descriptions in server eval JSON to explain aggregated and individual client metrics

**Benefits:**
- Unified metrics reporting in server evaluation files
- Complete federated learning analysis from single location
- Dataset-aware client identification and tracking
- Enhanced debugging and performance monitoring
- Simplified post-run analysis and reporting

**✅ Added Parameter Type Handling and Eval Mode Fixes Workflow**: Documented fixes for client parameter type compatibility, server eval_mode passing, and full evaluation episode limits to resolve federated learning crashes and limited evaluation scope.

**✅ Added Server-Side Loss Calculation Fix Workflow**: Fixed server evaluation to use policy loss as primary loss (SmolVLA flow-matching model), ensuring appropriate evaluation metrics

**✅ Added Client Metrics Aggregation Fix Workflow**: Enhanced server-side aggregation to collect and average client metrics (avg_loss, std_loss, proximal_loss, grad_norm, param_update_norm) for proper federated learning monitoring

**✅ Added Documentation Update Workflow**: Updated README.md, visualization.py, server_app.py, and tests to reflect policy loss metrics, including chart generation, file names, and metric descriptions

**✅ Added LR/MU Scheduling Enhancement Workflow**: Implemented advanced LR/MU scheduling with warm restarts, per-client adaptive LR boosts, dynamic mu adjustment, and spike detection. Added comprehensive unit tests, documentation updates, and memory bank entries. Targets <0.15 server policy loss with 100% client engagement.

## HF Push Workflow
**Last performed:** 2025-10-28
**Context:** Complete checkpoint directory-based HF Hub push system for SmolVLA federated learning models. Server creates full HF-ready directories, standalone script validates and uploads them.

**Prerequisites:**
- Conda environment "zk0" activated: `conda run -n zk0`
- HF token set in `.env`: `HF_TOKEN=your_token_here`
- Checkpoint directory with required files (model.safetensors, config.json, etc.)

**Files modified:**
- `src/server/server_utils.py` - `save_model_checkpoint()`, `push_model_to_hub_enhanced()`, `save_and_push_model()`
- `src/push_to_hf.py` - Updated to accept checkpoint directories
- `tests/unit/test_server_utils.py` - Added tests for directory creation and push logic
- `README.md` - Updated with checkpoint directory structure
- `.kilocode/rules/memory-bank/tasks.md` - Documented workflow

**Steps:**
1. **Server Checkpoint Creation**: `save_model_checkpoint()` creates `checkpoint_round_N/` directory with:
   - `model.safetensors` - Model weights in safetensors format
   - `config.json` - Model configuration from template
   - `README.md` - Auto-generated model card with hyperparameters, datasets, metrics, insights
   - `metrics.json` - Aggregated and individual client metrics
   - Base model files (tokenizer.json, preprocessor.json, etc.) copied from HF cache

2. **Server Push Logic**: `save_and_push_model()` calls `save_model_checkpoint()` (returning directory path), then `push_model_to_hub_enhanced(checkpoint_dir, repo_id)` for final round

3. **Standalone Push**: `conda run -n zk0 python -m zk0.push_to_hf /path/to/checkpoint_dir --repo-id user/model` validates required files and uploads entire directory

4. **HF Hub Upload**: `push_model_to_hub_enhanced()` uses `api.upload_folder()` to upload complete directory, creates git tags, handles cleanup

**Benefits:**
- **No Shape Mismatches**: Server creates directories with correct template/conversion logic
- **Complete Metadata**: Every checkpoint has full README, metrics, and all required HF files
- **Session Caching**: Base model files cached per server session using HF Hub cache
- **Standalone Simplicity**: Script just validates and uploads pre-built directories
- **Backward Compatibility**: Existing safetensors saves preserved; new directories for HF-ready checkpoints

**Testing:**
- Unit tests for `save_model_checkpoint()` and `push_model_to_hub_enhanced()`
- Integration test: Run tiny FL training, verify checkpoint directory creation and HF push
- Standalone test: Push saved directory manually, verify HF repo contents

## Prepare for Commit Workflow
**Last performed:** 2025-10-22
**Context:** Standardized workflow for preparing code changes for commit, ensuring quality, version management, and documentation consistency. Use this workflow for "prepare for commit", "wrap it up", or similar requests.

**Trigger Phrases:**
- "prepare for commit"
- "wrap it up"
- "ready for commit"
- "finalize changes"

**Steps:**
1. **Version Management**: Read pyproject.toml, increment version based on change scope (patch/minor/major), update version field
2. **Testing & Quality**: Run full test suite with coverage (`conda run -n zk0 python -m pytest -n auto --cov=src --cov-report=term-missing`), verify all pass (>=80% coverage), fix any failures
3. **Rules Compliance**: Review and update project rules as needed, document new workflows in memory bank
4. **Memory Bank Updates**: Revise context.md with recent progress, add new repetitive tasks to tasks.md, preserve knowledge
5. **Documentation Updates**: Update README.md with new features/changes, update technical docs for architecture changes
6. **Git Operations**: Stage changes, commit with conventional format message, create annotated tag, push to origin main

**Success Criteria:**
- All tests pass with >=80% coverage
- Version incremented appropriately
- Memory bank updated with progress
- Documentation current
- Git commit created with descriptive message
- Tag created and pushed

**Benefits:**
- Consistent code quality and standards
- Proper version management
- Complete documentation trail
- Reliable git history
- Easy future reference for similar tasks

**Important Notes:**
- Switch to code mode if test failures require fixes
- Use semantic versioning for version bumps
- Follow conventional commit format for messages
- Update memory bank before final commit

## Model Training Workflow
1. **Environment Setup**: Activate conda environment "zk0"
2. **Model Loading**: Load SmolVLA model from Hugging Face
3. **Dataset Preparation**: Load and preprocess SO-100 dataset
4. **Training Configuration**: Set hyperparameters and training parameters
5. **Local Training**: Execute training loop on local hardware
6. **Parameter Extraction**: Extract model updates for federation
7. **Server Communication**: Send updates to central server
8. **Model Update**: Receive aggregated global model
9. **Validation**: Test updated model performance

## Client Setup Workflow
1. **Dependency Installation**: Ensure all required packages are installed
2. **Flower Client Initialization**: Create NumPyClient instance
3. **Dataset Partitioning**: Load and partition local SO-100 dataset
4. **Model Configuration**: Configure SmolVLA model parameters
5. **Server Connection**: Establish secure connection to Flower server
6. **Training Loop**: Participate in federated learning rounds
7. **Parameter Synchronization**: Send/receive model parameters
8. **Logging and Monitoring**: Track training progress and metrics

## Server Setup Workflow
1. **Flower Server Initialization**: Configure ServerApp
2. **Strategy Selection**: Choose FedAvg or FedProx strategy
3. **Client Registration**: Accept client connections
4. **Round Orchestration**: Manage federated learning rounds
5. **Parameter Aggregation**: Combine client updates
6. **Model Distribution**: Broadcast updated global model
7. **Performance Monitoring**: Track convergence and metrics
8. **Security Validation**: Ensure privacy and security compliance

## Testing Workflow
1. **Unit Test Execution**: Run pytest on individual components
2. **Integration Testing**: Test client-server communication
3. **Federated Simulation**: Run local simulation with multiple clients
4. **Performance Benchmarking**: Measure training and inference times
5. **Coverage Analysis**: Ensure test coverage meets 80% requirement
6. **Validation Reporting**: Document test results and issues

## Deployment Workflow
1. **Environment Validation**: Verify conda environment and dependencies
2. **Configuration Review**: Check all configuration files
3. **Security Setup**: Configure TLS and authentication
4. **Resource Allocation**: Set up GPU and memory resources
5. **Service Startup**: Launch Flower server and clients
6. **Monitoring Setup**: Configure logging and alerting
7. **Health Checks**: Validate system health and connectivity

## Tiny Training Run Workflow
**Last performed:** 2025-10-13
**Context:** Quick validation runs with minimal parameters to test critical functionality without long execution times
**Command:** `./train-fl-simulation.sh --tiny` (recommended for consistency) or `conda run -n zk0 flwr run . local-simulation-serialized-gpu --run-config "num-server-rounds=1 local-epochs=1 batch_size=1 eval_batches=1"`

**Steps:**
1. **Minimal Configuration**: Set num-server-rounds=1, local-epochs=1, batch_size=1 for fastest possible execution
2. **Full Functionality Test**: Verifies all critical paths: client training, server aggregation, model push, evaluation
3. **Quick Validation**: Complete run should finish in ~5-10 minutes on modern hardware
4. **Error Detection**: Any failures indicate critical bugs that need immediate fixing
5. **Success Criteria**: All 4 clients train successfully, server aggregates parameters, model saved and pushed to HF
6. **No Client Evaluation**: Confirm clients only perform fit() operations, no evaluate() calls (server-side only)
7. **Memory Bank Update**: Document any issues found and fixes applied

**Benefits:**
- Rapid validation of critical functionality before full experiments
- Early detection of integration issues and bugs
- Confidence that core FL pipeline works correctly
- Minimal resource usage for frequent testing during development

## Post-FL Run Analysis Workflow
**Last performed:** 2025-10-18
**Context:** Analyzed 50-round dynamic decay run (2025-10-17_08-02-19) vs. 30-round baseline; confirmed decay smooths convergence (min loss 0.810 vs. 0.827), but highlights client dropouts (85% participation) in extended runs. Propose mu=0.05 and patience=10 for future stability.

**Steps:**
1. **Output Structure Review**: Use list_files to examine run directory; confirm clients/, server/, models/ presence and key files (checkpoints, evals, metrics)
2. **Key File Analysis**: Read config.json (run params), federated_metrics.json (client trends), policy_loss_history.json (server history), sample round evals (r0,10,20,30 for progression)
3. **Metrics Summary**: Extract client loss trends (avg_client_loss decline), param_update_norm stability, server policy_loss progression (r0 low → r10 peak → r30 recovery)
4. **Anomaly Detection**: Identify issues like round 21 (1 client dropout), zeroed metrics, empty history JSON
5. **Baseline Comparison**: Contrast with prior runs; assess policy_loss scale and convergence quality
6. **Issue Identification**: Flag logging bugs (missing client aggregation, history generation failure), hyperparam impacts
7. **Improvement Proposals**: Suggest code fixes, hyperparam tuning, resource monitoring
8. **Documentation**: Update memory bank with findings; propose version bump if progress significant
9. **Next Steps**: Recommend validation run post-fixes, full experiment with tuning
10. **Validation**: Ensure analysis actionable; todo list updated with findings

**Benefits:**
- Systematic FL performance assessment with quantifiable trends
- Early bug detection before full experiments
- Informed hyperparam tuning based on actual run data
- Preserved institutional knowledge via memory bank updates
- Clear handoff to code mode with prioritized fixes
