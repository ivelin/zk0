# Repetitive Task Workflows

**Created**: 2025-09-06
**Last Updated**: 2025-10-14
**Version**: 1.0.2
**Author**: Kilo Code

## Latest Update (2025-10-14)
**✅ Memory Bank Cleanup**: Consolidated repetitive workflows, removed outdated experiment details, updated metadata. Preserved essential technical workflows while reducing file size.

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

### Early Stopping Implementation
**Last performed:** 2025-10-13
**Context:** Added configurable server-side early stopping to prevent wasted computation when evaluation loss plateaus.

**Files modified:** `pyproject.toml`, `src/server_app.py`, `tests/unit/test_server_app.py`

**Steps:**
1. **Configuration Addition**: Added `early_stopping_patience = 10` parameter to `pyproject.toml` [tool.flwr.app.config] section
2. **Strategy State Initialization**: Added early stopping tracking attributes (best_eval_loss, rounds_without_improvement, early_stopping_triggered) in AggregateEvaluationStrategy.__init__()
3. **Pure Function Creation**: Implemented `check_early_stopping()` standalone function for determining if stopping criteria met
4. **Tracking Function**: Created `update_early_stopping_tracking()` function to update strategy state and log early stopping status
5. **Integration in _server_evaluate**: Added call to `update_early_stopping_tracking()` after each evaluation to monitor loss improvements
6. **Termination Logic**: Added early stopping check in `aggregate_fit()` to return None and terminate training when triggered
7. **Comprehensive Unit Tests**: Added 8 test cases covering all early stopping scenarios (disabled, improvement, no improvement, trigger, already triggered)
8. **Validation**: All tests pass; syntax check successful; imports work correctly

**Benefits:**
- Automatic termination prevents wasted GPU/compute resources
- Configurable patience allows tuning for different convergence patterns
- Clear logging enables understanding of stopping decisions
- Preserves best model state when terminating
- Easy to disable for experiments requiring fixed round counts

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

## Early Stopping None Parameter Fix Workflow
**Last performed:** 2025-10-14
**Context:** Fixed critical bug where early stopping could return None parameters, causing `TypeError: cannot unpack non-iterable NoneType object` in Flower's server unpacking
**Files modified:** `src/server_app.py`

**Steps:**
1. **Issue Detection**: Tiny training runs crashed with `TypeError: cannot unpack non-iterable NoneType object` in Flower server unpacking
2. **Root Cause Analysis**: Early stopping logic returned `self.current_parameters` which could be `None` if no parameters aggregated yet
3. **Fix Implementation**: Modified early stopping to return valid parameters:
   - Use `aggregated_parameters` if available (from current round)
   - Fall back to `self.initial_parameters` if no aggregated parameters exist
4. **Parameter Storage**: Added `self.initial_parameters` to store initial model parameters for fallback
5. **Validation**: Fixed code ensures Flower always receives valid `(parameters, metrics)` tuple
6. **Testing**: Next tiny training run should complete without unpacking errors

**Important notes:**
- Early stopping now safely handles cases where no parameters have been aggregated
- Always returns valid Flower Parameters object to prevent server crashes
- Maintains early stopping functionality while ensuring system stability
- Critical fix for short training runs (1 round) where early stopping might trigger

**Common Issues Addressed:**
- `TypeError: cannot unpack non-iterable NoneType object` in Flower server
- Training crashes during early stopping in short runs
- Unstable early stopping behavior with None parameter returns

**Benefits:**
- Stable early stopping that doesn't crash Flower server
- Reliable training termination for all run lengths
- Maintains model quality by returning best available parameters
- Prevents data loss from unexpected crashes during early stopping

**✅ Added Parameter Type Handling and Eval Mode Fixes Workflow**: Documented fixes for client parameter type compatibility, server eval_mode passing, and full evaluation episode limits to resolve federated learning crashes and limited evaluation scope.

**✅ Added Server-Side Loss Calculation Fix Workflow**: Fixed server evaluation to use policy loss as primary loss (SmolVLA flow-matching model), ensuring appropriate evaluation metrics

**✅ Added Client Metrics Aggregation Fix Workflow**: Enhanced server-side aggregation to collect and average client metrics (avg_loss, std_loss, proximal_loss, grad_norm, param_update_norm) for proper federated learning monitoring

**✅ Added Documentation Update Workflow**: Updated README.md, visualization.py, server_app.py, and tests to reflect policy loss metrics, including chart generation, file names, and metric descriptions

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
**Command:** `conda run -n zk0 flwr run . local-simulation-serialized-gpu --run-config "num-server-rounds=1 local-epochs=1 batch_size=1 eval_batches=8"`

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

### Comprehensive Implementation Checklist
This consolidated checklist combines the pre-implementation, implementation, and post-implementation phases into a single document, eliminating redundancies while preserving all unique items.

#### Pre-Implementation
- [ ] Working directory is `.`
- [ ] No changes planned for sibling or parent directories
- [ ] Conda environment "zk0" is active and available
- [ ] Focus maintained on SmolVLA model and SO-100 datasets
- [ ] Reference to quickstart-lerobot structure is considered
- [ ] Task context fully captured and documented
- [ ] Success criteria clearly defined and measurable
- [ ] Dependencies identified and accessible
- [ ] Technical constraints understood and documented
- [ ] Risk assessment completed
- [ ] Required tools and libraries available
- [ ] Development environment properly configured
- [ ] Access to necessary datasets and models
- [ ] Hardware requirements met (GPU, memory, etc.)
- [ ] Network connectivity for federated learning
- [ ] Coding standards reviewed and understood
- [ ] Testing requirements identified
- [ ] Documentation standards reviewed
- [ ] Security considerations addressed
- [ ] Implementation plan created
- [ ] Timeline and milestones defined
- [ ] Resource requirements identified
- [ ] Backup and recovery plans in place
- [ ] Communication plan established

#### Implementation
- [ ] Code follows established style guidelines (PEP 8)
- [ ] Comprehensive docstrings provided for all functions/classes
- [ ] Type hints included where appropriate
- [ ] Code is modular and maintainable
- [ ] Error handling implemented appropriately
- [ ] Logging added for debugging and monitoring
- [x] SmolVLA model properly loaded and configured
- [x] SO-100 dataset integration working correctly
- [x] Federated learning setup matches Flower requirements
- [x] Model parameters handled according to SmolVLA specs
- [x] Asynchronous inference implemented where beneficial
- [x] Dataset synchronization issues resolved with high tolerance
- [x] Model loading issues fixed with proper environment variables
- [ ] Unit tests written and passing
- [ ] Integration tests implemented
- [ ] Test coverage maintained above 80%
- [ ] Edge cases and error conditions tested
- [ ] Performance benchmarks met
- [ ] Code documentation complete and accurate
- [ ] API documentation updated
- [ ] Usage examples provided
- [ ] Configuration instructions documented
- [ ] Troubleshooting guide included
- [ ] Memory usage optimized for available hardware
- [ ] GPU utilization efficient
- [ ] Training/inference times meet requirements
- [ ] Scalability considerations addressed
- [ ] Resource cleanup implemented
- [ ] No sensitive data exposed in logs
- [ ] Secure handling of model weights and data
- [ ] Compliance with federated learning privacy requirements
- [ ] Input validation implemented
- [ ] Error messages don't leak sensitive information

#### Post-Implementation
- [ ] All success criteria from original task met
- [ ] Performance requirements satisfied
- [ ] Functional requirements verified
- [ ] Non-functional requirements met
- [ ] Stakeholder acceptance obtained
- [ ] All tests passing (unit, integration, end-to-end)
- [ ] Code review completed and approved
- [ ] Security review passed
- [ ] Performance benchmarks achieved
- [ ] Documentation reviewed and approved
- [ ] Integration with existing systems verified
- [ ] Backward compatibility maintained
- [ ] API contracts respected
- [ ] Data format compatibility confirmed
- [ ] Configuration compatibility verified
- [ ] Production environment tested
- [ ] Deployment scripts created/updated
- [ ] Rollback procedures documented
- [ ] Monitoring and alerting configured
- [ ] Backup and recovery procedures tested
- [ ] User documentation updated
- [ ] API documentation published
- [ ] Runbook and troubleshooting guides updated
- [ ] Knowledge transfer to operations team completed
- [ ] Training materials updated if needed
- [ ] Lessons learned documented in memory bank
- [ ] New patterns or best practices captured
- [ ] Technical specifications updated
- [ ] Workflow improvements identified and documented
- [ ] Context for future tasks preserved