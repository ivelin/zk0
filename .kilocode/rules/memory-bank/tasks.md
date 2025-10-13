# Repetitive Task Workflows

**Created**: 2025-09-06
**Last Updated**: 2025-10-05
**Version**: 1.0.1
**Author**: Kilo Code

## Latest Update (2025-10-13)
**✅ Added Early Stopping Implementation Workflow**: Implemented configurable server-side early stopping to prevent wasted computation when evaluation loss plateaus. Training terminates automatically if no improvement occurs for configurable patience rounds (default: 10). Set `early_stopping_patience = 0` to disable.

**✅ Added Parameter Type Handling and Eval Mode Fixes Workflow**: Documented fixes for client parameter type compatibility, server eval_mode passing, and full evaluation episode limits to resolve federated learning crashes and limited evaluation scope.

**✅ Added Server-Side Loss Calculation Fix Workflow**: Fixed server evaluation to use policy loss as primary loss (SmolVLA flow-matching model), ensuring appropriate evaluation metrics instead of MSE

**✅ Added Client Metrics Aggregation Fix Workflow**: Enhanced server-side aggregation to collect and average client metrics (avg_loss, std_loss, proximal_loss, grad_norm, param_update_norm) for proper federated learning monitoring

**✅ Added Documentation Update Workflow**: Updated README.md, visualization.py, server_app.py, and tests to reflect policy loss metrics instead of MSE, including chart generation, file names, and metric descriptions

## Client Metrics Aggregation Fix Workflow
**Last performed:** 2025-10-09
**Context:** Enhanced server-side aggregation to collect and average client metrics (avg_loss, std_loss, proximal_loss, grad_norm, param_update_norm) for proper federated learning monitoring
**Files modified:** `src/server_app.py`

**Steps:**
1. **Issue Detection**: federated_metrics.json showed all zeros (num_clients=0, avg_action_mse=0.0), preventing visibility into client-side training progress
2. **Root Cause Analysis**: Server aggregate_fit() called super().aggregate_fit() but didn't extract or aggregate client metrics from FitRes results
3. **Metrics Extraction**: Added client metrics aggregation before calling parent aggregate_fit():
   - Extract loss, fedprox_loss, grad_norm from each validated client's FitRes.metrics
   - Compute averages and standard deviations across all clients
   - Calculate parameter update norm between rounds (L2 distance of parameter changes)
4. **Storage for Evaluation**: Store aggregated metrics in self.last_aggregated_metrics for use in _server_evaluate()
5. **Federated Metrics Update**: Modified _server_evaluate() to use actual client count and metrics instead of hardcoded zeros
6. **Visualization Enhancement**: Updated round_metrics to include client avg_loss and param_update_norm for plotting
7. **Validation**: Syntax check passed; next FL run should populate federated_metrics.json with real client data
8. **Memory Bank Update**: Documented aggregation enhancements in context.md and tasks.md

**Important notes:**
- Client metrics are returned by fit() in final_metrics dict (loss, grad_norm, fedprox_loss, etc.)
- Parameter update norm tracks model evolution between rounds (should decrease as convergence approaches)
- Federated metrics now include client participation count, loss statistics, and convergence indicators
- Metrics are merged with parent FedProx metrics to preserve existing functionality
- Visualization plots now show actual client performance trends

**Common Issues Addressed:**
- Zeroed federated metrics preventing FL progress monitoring
- Missing client-side loss aggregation for convergence analysis
- Lack of parameter update tracking between rounds
- Inaccurate client count in evaluation metrics
- Insufficient debugging data for FL optimization

**Benefits:**
- Real-time visibility into client training performance and participation
- Proper convergence monitoring with loss trends and parameter update norms
- Enhanced debugging capabilities with comprehensive metric collection
- Better FL experiment analysis with client-level statistics
- Improved federated_metrics.json accuracy for post-run analysis

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

## FedProx Debugging and Fix Workflow
**Last performed:** 2025-10-01
**Context:** Diagnosed and fixed increasing loss issue in federated learning training
**Files modified:** `src/task.py`, `src/client_app.py`

**Steps:**
1. **Issue Detection**: Monitor client logs for increasing loss patterns (e.g., loss growing from 0.075 to 50+ over steps)
2. **Log Analysis**: Examine FedProx proximal_loss and adjusted loss values to identify formula/timing issues
3. **Root Cause Identification**: Check FedProx formula (should be mu/2 * ||w - w_global||^2, not mu * ||w - w_global||^2)
4. **Formula Correction**: Update proximal loss calculation to use correct FedProx formula with /2.0 division
5. **Timing Fix**: Move proximal calculation after `update_policy()` to use post-forward parameters
6. **Code Refactoring**: Break monolithic `train()` function into focused helpers:
   - `setup_training_components()`: Optimizer, scheduler, metrics setup
   - `run_training_step()`: Single training step with FedProx regularization
   - `run_training_loop()`: Main training loop orchestration
7. **Validation**: Test with short training run to confirm loss decreases properly
8. **Memory Bank Update**: Document fixes and refactoring in context.md

**Important notes:**
- FedProx formula must include /2.0 for correct regularization strength
- Proximal term should be calculated once per step after policy update
- Refactoring improves maintainability and debugging of training issues
- Always verify loss decreases during training before considering fix complete
- Update memory bank with any new patterns discovered during debugging

**Common Issues:**
- Wrong formula: Missing /2.0 in proximal loss calculation
- Timing issues: Calculating proximal term before vs after update_policy()
- Accumulation bugs: Proximal loss growing exponentially across steps
- Loss addition: Proximal term added multiple times per step

## FedProx Hyperparameter Tuning Workflow
**Last performed:** 2025-10-01
**Context:** Fixed loss explosion by reducing proximal_mu from 0.1 to 0.001
**Files modified:** `pyproject.toml`

## Client Logging Enhancement Workflow
**Last performed:** 2025-10-03
**Context:** Improved client logging for better debugging of federated learning runs, especially for data corruption issues
**Files modified:** `src/client_app.py`, `src/task.py`

**Steps:**
1. **Dataset Name Logging**: Add dataset repository ID logging in client fit() and evaluate() methods for traceability
2. **Batch Skipping Tracking**: Implement skipped_batches counter in training loops to track corrupt frame handling
3. **Max Attempts Increase**: Increase max_attempts from 5 to 50 in both global and SmolVLATrainer run_training_loop functions
4. **Enhanced Error Details**: Include attempt counts and total skipped batches in warning messages
5. **Completion Logging**: Add skipped_batches count to training completion logs
6. **Code Quality Check**: Run ruff check to ensure no linting issues
7. **Memory Bank Update**: Document changes in context.md

**Important notes:**
- Dataset name logging helps identify which client is using which dataset during debugging
- Increased max_attempts (50) allows clients with corrupted datasets to complete more training steps
- Skipped batch tracking provides visibility into data quality issues
- Changes apply to both the global run_training_loop and SmolVLATrainer.run_training_loop methods
- All changes maintain backward compatibility and existing functionality

**Common Issues Addressed:**
- Unclear which dataset a failing client is using
- Insufficient attempts to skip corrupt frames, causing premature training failure
- Lack of visibility into how many batches are being skipped due to data corruption
- Incomplete logging of training completion status

## Server Parameter Aggregation Fix Workflow
**Last performed:** 2025-10-01
**Context:** Fixed issue where clients started each round with identical loss values due to server not properly returning aggregated parameters
**Files modified:** `src/server_app.py`

**Steps:**
1. **Issue Detection**: Monitor client logs for identical starting loss values across rounds (indicates parameters not updating)
2. **Root Cause Analysis**: Check if server `aggregate_fit` method properly returns aggregated parameters from `super().aggregate_fit()`
3. **Parameter Flow Verification**: Compare parameter norms between:
   - Parameters sent to fit round
   - Parameters returned from fit round
   - Parameters sent to evaluate round
   - Parameters sent to next fit round
4. **Fix Implementation**: Ensure `aggregate_fit` method captures and returns aggregated parameters:
   ```python
   aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
   return aggregated_parameters, metrics
   ```
5. **Validation**: Verify parameter norms change between rounds and loss values are different at round starts
6. **Testing**: Run multi-round training to confirm proper parameter flow through entire FL pipeline

**Important notes:**
- Flower's strategy pattern requires returning aggregated parameters to update server's global model
- Parameter extraction (`get_params`) and setting (`set_params`) functions should work correctly
- FedProx aggregation happens at the strategy level, server just needs to return the results
- Always verify parameter flow by comparing norms across fit/evaluate round boundaries
- Loss progression should show different starting values if parameters are updating correctly

**Common Issues:**
- Server not returning aggregated parameters from `aggregate_fit`
- Parameter extraction/setting functions corrupting data
- Strategy configuration not properly handling parameter updates
- Validation: Parameter norms should change between rounds, loss should not be identical at round starts

**Steps:**
1. **Issue Detection**: Monitor training logs for rapidly growing loss (proximal_loss increasing exponentially)
2. **Hyperparameter Analysis**: Check current proximal_mu value in pyproject.toml `[tool.flwr.app.config]` section
3. **Root Cause Identification**: Determine if proximal_mu is too high (typical values: 0.001-0.01, not 0.1+)
4. **Gradual Reduction**: Reduce proximal_mu by factor of 10-100x (e.g., 0.1 → 0.001, 0.01 → 0.0001)
5. **Configuration Update**: Modify `proximal_mu` value in pyproject.toml with explanatory comment
6. **Validation Testing**: Run short training test to verify loss decreases properly
7. **Fine-tuning**: If still problematic, try further reduction (0.001 → 0.0005) or slight increase (0.001 → 0.002)
8. **Memory Bank Update**: Document the fix and new hyperparameter value in context.md

**Important notes:**
- Start with conservative reductions (factor of 10) to avoid losing regularization benefits
- proximal_mu=0.001 provides gentle regularization (0.1% loss weight)
- proximal_mu=0.0001 provides very gentle regularization (0.01% loss weight)
- proximal_mu=0.01 provides moderate regularization (1% loss weight)
- Always test after changes to ensure loss decreases during training
- Document the reasoning for the chosen value in comments

**Common Issues:**
- Too high proximal_mu: Loss grows exponentially, training fails
- Too low proximal_mu: No regularization benefit, poor convergence on heterogeneous data
- Loss trends: Monitor both proximal_loss and adjusted_loss for proper convergence
- Validation: Always run test training after hyperparameter changes

## Server-Side Evaluation Transition Workflow
**Last performed:** 2025-10-08
**Context:** Transitioned from client-side to server-side evaluation only, removing episode filtering logic that wasn't working with current datasets
**Files modified:** `src/task.py`, `src/server_app.py`

**Steps:**
1. **Issue Detection**: Identified that `dataset.episodes` is None in current datasets (like "shaunkirby/record-test") due to codec/FFmpeg issues, making episode-based filtering impossible
2. **Episode Filtering Removal**: Removed `filter_dataset_by_split()` and `load_data()` functions that relied on unavailable episode metadata
3. **Test Function Update**: Modified `test()` function to use server evaluation dataset directly, limiting evaluation to first N episodes by tracking episode changes during iteration
4. **Server Strategy Modification**: Updated `configure_evaluate()` to return empty config (no client evaluation requests) and `aggregate_evaluate()` to perform server-side evaluation
5. **Parameter Storage**: Added `current_parameters` attribute to strategy to store aggregated parameters for server evaluation
6. **Code Cleanup**: Removed duplicate code sections and updated all `load_data` calls to use `load_lerobot_dataset` directly
7. **Validation**: Run syntax checks and test federated learning simulation to verify server-side evaluation works correctly
8. **Memory Bank Update**: Document architecture changes in context.md, architecture.md, product.md, and tech.md

**Important notes:**
- Server-side evaluation ensures consistent evaluation across rounds using dedicated evaluation datasets
- Training uses all available episodes (no filtering needed), evaluation uses first N episodes from server datasets
- Current datasets have `dataset.episodes = None` due to codec issues, making episode-based filtering impossible
- Server evaluates the latest aggregated model parameters after each round
- Cleaner architecture with simplified data handling and more reliable evaluation

**Common Issues Addressed:**
- Episode filtering failing silently when `dataset.episodes` is None
- Inconsistent train/eval episode separation between manual filtering and automated filtering
- Client-side evaluation creating unnecessary complexity and potential inconsistencies
- Dataset loading issues with current SO-100 datasets due to codec problems

**Benefits:**
- More reliable evaluation that doesn't depend on episode metadata availability
- Simplified architecture with server-only evaluation
- Consistent evaluation methodology across all federated learning rounds
- Better separation of concerns between training (clients) and evaluation (server)

## Debug Fixes Workflow
**Last performed:** 2025-10-08
**Context:** Fixed HF push 403 Forbidden and server-side eval issues in federated learning setup
**Files modified:** `src/server_app.py`

**Steps:**
1. **HF Push 403 Diagnosis**: Identified 403 Forbidden error during model push to Hugging Face Hub due to non-existent repo "ivelin/zk0-smolvla-fl"
2. **Repo Auto-Creation Fix**: Added `api.create_repo(exist_ok=True)` in `push_model_to_hub` to automatically create missing repos
3. **Token Validation**: Added logging to check HF_TOKEN presence and repo existence before upload
4. **Server Eval Flow Fix**: Removed redundant eval block from `aggregate_fit`; eval now runs exclusively via `evaluate_fn` (called by Flower's `strategy.evaluate` post-fit, gated by `eval-frequency`)
5. **Model Reuse Optimization**: Replaced redundant model creation in `_server_evaluate` and norm computation with cached `self.template_model`
6. **Code Cleanup**: Removed unused `get_evaluate_config_callback` function and redundant imports
7. **Validation**: Added detailed logging for debugging (entry/exit points, dataset loading, model operations)
8. **Memory Bank Update**: Documented fixes in context.md and tasks.md

**Important notes:**
- HF 403 was due to missing repo; auto-creation ensures reliable uploads
- Server eval must use `evaluate_fn` for proper Flower integration, not manual blocks in `aggregate_fit`
- Cached `template_model` eliminates redundant model loading across rounds
- Eval frequency gating prevents unnecessary evaluations
- All changes maintain Flower compatibility and FedProx inheritance

**Common Issues Addressed:**
- HF push failures due to missing repos or invalid tokens
- Duplicate eval execution causing confusion and overhead
- Unnecessary model recreation impacting performance
- Missing logging for debugging complex FL flows
- Unused code cluttering the codebase

## Parameter Type Handling and Eval Mode Fixes Workflow
**Last performed:** 2025-10-09
**Context:** Fixed client parameter type compatibility, server eval_mode passing, and full evaluation episode limits to resolve federated learning crashes and limited evaluation scope
**Files modified:** `src/client_app.py`, `src/server_app.py`, `src/task.py`

**Steps:**
1. **Client Parameter Type Issue**: Identified AttributeError 'list' object has no attribute 'tensors' in client fit() when Flower passes parameters as list of ndarrays instead of Parameters object
2. **Type Check Addition**: Added `if isinstance(parameters, list): received_ndarrays = parameters else: received_ndarrays = parameters_to_ndarrays(parameters)` in client fit() hash validation
3. **Server Eval Mode Passing**: Fixed `_server_evaluate` to retrieve `eval_mode` from `self.context.run_config.get("eval_mode", "quick")` instead of defaulting incorrectly
4. **Full Eval Episode Limit**: Corrected test() function to use `len(dataset.episodes)` for full mode max_episodes instead of `len(dataset)` (frames vs episodes)
5. **Ray GCS Crash Investigation**: Latest run crashed with Ray GCS communication error after 32 rounds, likely resource exhaustion in long-running simulation
6. **Validation**: Applied fixes and documented in memory bank; next run should show client training success and full evaluation processing all episodes
7. **Memory Bank Update**: Updated context.md, brief.md, and tasks.md with latest fixes and status

**Important notes:**
- Flower parameter format varies by version; type checking ensures compatibility
- Eval mode must be passed from run config to test function for proper full/quick mode operation
- Dataset.episodes provides episode count, not frames; critical for full evaluation limits
- Ray crashes in extended simulations suggest resource monitoring needed
- All fixes maintain backward compatibility and existing functionality

**Common Issues Addressed:**
- Client crashes preventing federated learning training rounds
- Incorrect evaluation mode causing limited scope (quick instead of full)
- Wrong episode counting leading to incomplete full evaluations
- Resource exhaustion in long-running FL simulations
- Parameter serialization incompatibilities between Flower versions

## Context Management Guidelines

### Context Passing Mechanism
The context passing mechanism ensures that critical information flows seamlessly between tasks, subtasks, and mode transitions without loss or degradation.

#### Core Components
- **Context Capture**: What to capture and initial extraction
- **Context Packaging**: How to package and format context
- **Context Transfer**: How to transfer between tasks and modes
- **Context Validation**: How to validate context integrity

#### Subtask Creation Protocol
1. **Context Extraction**: Extract relevant context from parent task, identify inheritance requirements, determine validation needs
2. **Context Packaging**: Use task context template, include all mandatory constraints, add subtask-specific context
3. **Context Transfer**: Embed context in subtask description, reference memory bank for details, include validation checkpoints
4. **Context Validation**: Verify completeness, confirm constraints included, validate success criteria

#### Context Formats
**Template-Based Format:**
```markdown
## Project Constraints (MANDATORY)
See [.kilocode/rules/memory-bank/project-constraints.md](.kilocode/rules/memory-bank/project-constraints.md) for the complete list of project constraints.

## Task Context
[Task-specific context]

## Success Criteria
[Measurable outcomes]
```

**Structured Format:**
```json
{
  "constraints": {
    "workDirectory": "local project repository root directory",
    "environment": "zk0",
    "focus": "SmolVLA + SO-100"
  },
  "context": {
    "taskId": "TASK-001",
    "successCriteria": [...],
    "dependencies": [...]
  },
  "validation": {
    "checklist": [...],
    "metadata": {
      "created": "2025-09-03",
      "version": "1.0"
    }
  }
}
```

### Context Inheritance Rules
Context inheritance ensures that critical information, constraints, and requirements are automatically passed from parent tasks to subtasks, preventing loss of important details during task decomposition.

#### Inheritance Levels
- **Level 1: Mandatory Constraints** (Always inherited - no exceptions): Project working directory, environment requirements, technical focus, scope limitations
- **Level 2: Task Context** (Inherited for all subtasks): Success criteria, key technical decisions, critical assumptions, risk assessments
- **Level 3: Operational Context** (Inherited based on relevance): Current system state, recent changes, known issues, performance baselines

#### Inheritance Mechanism
**Automatic Inheritance:**
- Project Constraints Block reference
- Parent Task Reference (ID, success criteria, context summary)
- Technical Context (current working state, dependencies, known constraints)

**Manual Inheritance:**
- Business Logic (task purpose, contribution to parent goal, success criteria alignment)
- Technical Decisions (architecture choices, design patterns, implementation constraints)
- Risk Context (known risks, mitigation strategies, fallback plans)

### Context Management Strategies
Context management strategies optimize context handling across session transitions and task workflows.

#### Session Transition Template
```
[Memory Bank: Active]
Project: [project name]
Status: [current state summary]
Last Updated: [date]
Key Technical Context:
- Environment: [conda env, key dependencies]
- Architecture: [core components, patterns]
- Constraints: [critical rules, standards]
Current Focus: [immediate next task]
```

#### Context Health Monitoring
- **Indicators of Approaching Limits**: Increased back-referencing, longer response times, more memory bank consultations
- **Optimal Transition Points**: After major milestones, before complex debugging, when conversation exceeds 50 messages
- **Recovery Strategies**: Immediate assessment, memory bank review, gap documentation, user verification

#### Memory Bank Update Rules
**MANDATORY Updates:**
- Before session transitions: Update context.md with current task status
- After major milestones: Document completed features and decisions
- When critical decisions made: Capture architectural choices and trade-offs
- Before complex debugging: Document current state and known issues
- After subtask completion: Update progress and results

### Context Loss Prevention and Recovery
- **Risk Identification**: High-risk areas include mode transitions, task decomposition, external interruptions
- **Prevention Strategies**: Template usage, documentation of assumptions, validation checkpoints, context backups
- **Recovery Procedures**: Immediate stop, assessment, reconstruction using memory bank, validation, resume
- **Quality Metrics**: Completeness score (>90% Level 2), preservation rate tracking, loss rate monitoring

## Appendix: Task Context Template

### Task Information
**Task ID**: [Unique identifier]
**Parent Task**: [Parent task ID if applicable]
**Created**: [Date/time]
**Mode**: [Current mode]
**Priority**: [High/Medium/Low]

### Task Description
**Objective**: [Clear, specific description of what needs to be accomplished]
**Scope**: [What is included and what is excluded]
**Success Criteria**: [Measurable outcomes that define success]

### Technical Context
**Current State**: [Description of current system/project state]
**Dependencies**: Code, data, and external dependencies
**Constraints**: Technical, time, and resource constraints

### Context to Preserve
**Critical Information**: [Information that MUST be maintained across subtasks]
**Key Decisions**: [Important decisions made and rationale]
**Assumptions**: [Assumptions that subtasks should be aware of]

### Subtask Requirements
**Context Inheritance**: [What context each subtask must inherit]
**Handover Information**: [What information needs to be passed to next task/mode]
**Validation Points**: [Points where context preservation should be validated]

### Risk Mitigation
**Potential Context Loss Points**: [Where context might be lost]
**Recovery Procedures**: [How to recover if context is lost]
**Prevention Measures**: [How to prevent context loss]

### Quality Assurance
**Pre-Task Checklist**:
- [ ] All constraints verified
- [ ] Context fully captured
- [ ] Success criteria defined
- [ ] Dependencies identified

**Post-Task Validation**:
- [ ] Success criteria met
- [ ] Context preserved
- [ ] Documentation updated
- [ ] Quality standards maintained

### Mode Transition Notes
[Notes for transitioning to other modes]

**Context Captured by**: [Name/Mode]
**Last Updated**: [Date/Time]
**Version**: [Version number]

## Implementation Checklists

### Pre-Implementation Checklist
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

### Implementation Checklist
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

### Post-Implementation Checklist
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

## Server-Side Loss Calculation Fix Workflow
**Last performed:** 2025-10-09
**Context:** Fixed server evaluation to use policy loss as primary loss (SmolVLA flow-matching model), ensuring appropriate evaluation metrics
**Files modified:** `src/task.py`

**Steps:**
1. **Issue Detection**: Server evaluation needed alignment with SmolVLA's flow-matching objective
2. **Root Cause Analysis**: Ensured test() returns policy_loss as primary loss
3. **Fix Implementation**: Set primary loss to policy_loss in test() function
4. **Metric Preservation**: Kept policy_loss in metrics dict for detailed evaluation context
5. **Validation**: Code change applied; next FL run should show consistent policy_loss reporting
6. **Memory Bank Update**: Updated context.md and tasks.md with fix details

**Important notes:**
- Primary loss now policy_loss for SmolVLA flow-matching alignment
- Policy_loss available in metrics for detailed analysis
- Ensures server evaluation is comparable to client-side metrics for convergence tracking
- Aligns with zk0 evaluation practices

**Common Issues Addressed:**
- Inability to track FL convergence due to metric inconsistency
- Missing alignment with SmolVLA evaluation standards

**Benefits:**
- Enables meaningful convergence tracking across federated learning rounds
- Consistent metrics with zk0 FL experiments
- Better debugging with policy_loss metrics
- Proper loss trend analysis for hyperparameter tuning

## Post-FL Run Analysis Workflow
**Last performed:** 2025-10-11
**Context:** Analyzed 30-round baseline run outputs to assess convergence, identify logging issues, and propose improvements for MSE tracking and client dropout prevention
**Files analyzed:** outputs/2025-10-09_13-59-05/* (config.json, federated_metrics.json, eval_mse_history.json, round_X_server_eval.json)

**Steps:**
1. **Output Structure Review**: Use list_files to examine run directory; confirm clients/, server/, models/ presence and key files (checkpoints, evals, metrics)
2. **Key File Analysis**: Read config.json (run params), federated_metrics.json (client trends), eval_mse_history.json (server history), sample round evals (r0,10,20,30 for progression)
3. **Metrics Summary**: Extract client loss trends (avg_client_loss decline), param_update_norm stability, server policy_loss progression (r0 low → r10 peak → r30 recovery)
4. **Anomaly Detection**: Identify issues like round 21 (1 client dropout), zeroed action_mse, empty MSE history JSON
5. **Baseline Comparison**: Contrast with prior 20-round run (MSE ~2722 plateau, mu=0.001); assess policy_loss scale shift and convergence quality (final 0.544 vs. prior ~0.43 avg)
6. **Issue Identification**: Flag logging bugs (missing client MSE aggregation, history generation failure), hyperparam impacts (mu=0.01 fluctuations)
7. **Improvement Proposals**: Suggest code fixes (add MSE to client metrics, fix history population), hyperparam tuning (mu=0.001), monitoring (Ray resources)
8. **Documentation**: Update memory bank (context.md new baseline, tasks.md this workflow); propose version bump if progress significant
9. **Next Steps**: Recommend validation run (5 rounds post-fixes), full experiment (50 rounds tuned), mode switch to code for implementations
10. **Validation**: Ensure analysis actionable; todo list updated with findings

**Important notes:**
- Focus on policy_loss as primary (SmolVLA flow-matching) but retain MSE for action prediction comparability
- Anomalies like client dropout often Ray simulation artifacts; check simulation.log for OOM/timeout
- Always read multi-files (up to 10) in single read_file for efficiency
- Compare scales: policy_loss ~0.3-1.5 (sensitive), MSE ~2000-4000 (action quality)
- Document baselines in context.md for future runs; target 10-20% improvement per iteration

**Common Issues Addressed:**
- Incomplete run analysis missing client/server metric correlation
- Unaddressed logging gaps preventing convergence diagnosis
- No systematic anomaly investigation (e.g., dropouts, zero metrics)
- Baseline comparisons without metric scale normalization
- Missing workflow for repeatable post-run evaluations

**Benefits:**
- Systematic FL performance assessment with quantifiable trends
- Early bug detection (logging, aggregation) before full experiments
- Informed hyperparam tuning based on actual run data
- Preserved institutional knowledge via memory bank updates
- Clear handoff to code mode with prioritized fixes

## Early Stopping Implementation Workflow
**Last performed:** 2025-10-13
**Context:** Added configurable server-side early stopping to prevent wasted computation when evaluation loss plateaus. Training terminates automatically if no improvement occurs for configurable patience rounds (default: 10). Set `early_stopping_patience = 0` to disable.

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

**Important notes:**
- Early stopping monitors server evaluation loss after each round
- Patience parameter controls rounds without improvement before stopping (0 = disabled)
- Only triggers after consistent lack of progress over multiple rounds
- Preserves best model achieved when stopping
- Comprehensive logging shows decision-making process
- Pure functions enable easy unit testing in isolation

**Common Issues Addressed:**
- Wasted computation on plateaued training runs
- Manual monitoring required for convergence detection
- No automatic termination for non-converging experiments
- Resource inefficiency in long federated learning runs

**Benefits:**
- Automatic termination prevents wasted GPU/compute resources
- Configurable patience allows tuning for different convergence patterns
- Clear logging enables understanding of stopping decisions
- Preserves best model state when terminating
- Easy to disable for experiments requiring fixed round counts
- Unit testable core logic ensures reliability

## Policy Loss Standardization Workflow
**Last performed:** 2025-10-11
**Context:** Standardized all metrics to policy_loss for SmolVLA flow-matching objective; removed MSE calculations and reporting
**Files modified:** `src/task.py`, `src/client_app.py`, `src/server_app.py`, `src/visualization.py`, documentation

**Steps:**
1. **MSE Removal in task.py**: Eliminate raw_mse, normalized_loss in test() and training; return policy_loss as primary
2. **Client App Updates**: In fit()/evaluate(), compute/save only policy_loss; drop MSE from metrics
3. **Server App Updates**: Use policy_loss in _server_evaluate(); aggregate avg_policy_loss in federated_metrics.json; rename history to policy_loss_history.json
4. **Visualization Fix**: Update generate_charts() to plot policy_loss (server progression, client avgs) from federated_metrics and server evals; save as policy_loss_chart.png
5. **Client JSON Restore**: In client_app.py fit(), save {"round": round, "client_id": cid, "policy_loss": avg_loss, ...} to clients/client_N/round_round.json
6. **Documentation**: Update README, pyproject.toml comments to reflect policy_loss sole metric
7. **Tests**: Remove MSE assertions; add policy_loss validations
8. **Validation**: Run 5-round test to verify JSON saves, populated charts, no MSE remnants
9. **Memory Bank Update**: Revise context.md, tasks.md to remove MSE baselines; add this workflow

**Important notes:**
- Policy_loss (~0.3-1.5 scale) aligns with SmolVLA objective; simplifies tracking
- Client JSONs enable per-round analysis; charts now non-empty with policy data
- Backward compatible; no impact on training logic
- Target: policy_loss <0.4 in future runs

**Common Issues Addressed:**
- Confusing dual metrics (MSE vs. policy_loss)
- Empty charts due to MSE focus
- Missing client-side round metrics
- Inconsistent baselines with MSE plateau

**Benefits:**
- Focused convergence on SmolVLA's core objective
- Reliable visualizations and per-client tracking
- Reduced code complexity (no MSE computations)
- Clearer hyperparam tuning via single metric

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