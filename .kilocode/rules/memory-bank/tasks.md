# Repetitive Task Workflows

**Created**: 2025-09-06
**Last Updated**: 2025-10-01
**Version**: 1.0.0
**Author**: Kilo Code

## Latest Update (2025-10-01)
**✅ Added FedProx Hyperparameter Tuning Workflow**: Documented the process for fixing loss explosion issues by adjusting proximal_mu values. This addresses the critical issue where proximal regularization was too strong, causing training loss to grow exponentially instead of decreasing.

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