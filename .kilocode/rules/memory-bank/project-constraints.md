# Project-Wide Constraints and Rules

**Created**: 2025-09-06
**Last Updated**: 2025-09-19
**Version**: 1.0.1
**Author**: Kilo Code

## Critical Project Constraints

**Summary:** Work in root dir, use zk0 conda env, focus on SmolVLA/Flower/SO-100
⚠️ **MANDATORY**: These constraints must be included in EVERY task, subtask, and mode transition:

### 1. Working Directory Constraint
- **Write Location**: File modifications and changes only within the local project repository root directory (/home/ivelin/zk0)
- **Read Access**: Read operations allowed in all workspace folders, including lerobot and flower
- **No External Changes**: No modifications to sibling or parent directories outside the project root
- **Scope Limitation**: All development work must remain within the project root directory

### 2. Environment Requirements
- **Primary Environment**: Conda environment "zk0" for local development and training runs (validated for federated learning execution)
- **Secondary Environment**: Docker container (`zk0`) via train.sh for reproducible, isolated execution of training and simulations
- **VSCode Integration**: VSCode with automatic environment detection (conda preferred; Docker integration optional)
- **Training Script**: Use `conda activate zk0; flwr run . local-simulation-serialized-gpu` for primary conda executions or `./train.sh` for secondary Docker-based executions
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

### 5. Testing Execution Requirements
- **Environment**: All tests must run in primary conda zk0 environment for consistency; use Docker as secondary for isolation
- **Parallel Execution**: Use `pytest -n auto` for parallel test execution
- **Coverage**: Always include `--cov=src --cov-report=term-missing` for coverage reporting
- **Command Format**: Use `conda run -n zk0 python -m pytest -n auto --cov=src --cov-report=term-missing` for primary; or train.sh/Docker for secondary

### 7. Environment Dependency Management
- **Reproducible Dependencies**: When new OS-level or Python dependencies are needed, update Dockerfile and/or requirements.txt for reproducibility
- **Docker-first**: Prefer Docker-based dependency management over local environment modifications
- **Version Pinning**: Always pin dependency versions in requirements.txt to ensure consistent environments
- **Documentation**: Document any new dependencies and their purpose in commit messages and memory bank

### 6. Testing and CI Standards
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
