# Project-Wide Constraints and Rules

**Created**: 2025-09-06
**Last Updated**: 2025-09-10
**Version**: 1.1.0
**Author**: Kilo Code

## Critical Project Constraints

**Summary:** Work in root dir, use zk0 conda env, focus on SmolVLA/Flower/SO-100
⚠️ **MANDATORY**: These constraints must be included in EVERY task, subtask, and mode transition:

### 1. Working Directory Constraint
- **Work Location**: Work only within the local project repository root directory
- **No External Changes**: No modifications to sibling or parent directories
- **Scope Limitation**: All development must remain within the project root directory

### 2. Environment Requirements
- **Conda Environment**: Must use conda environment "zk0"
- **VSCode Integration**: VSCode automatically detects and uses the zk0 environment
- **Manual Execution**: If needed, use `conda run -n zk0 python ...`
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
- **Fail Fast Principle**: Production code must handle real failures gracefully or fail fast with clear error messages
- **Real Dependencies Only**: All production components must work with actual dependencies and real data
- **Error Handling**: Implement proper exception handling instead of mock fallbacks
- **Testing Isolation**: Mocks are only acceptable in unit tests for dependency isolation, never in production paths
- **Code Review Requirement**: Any use of mocks in production code will be rejected during code review

### 5. Testing Execution Requirements
- **Environment**: All tests must run in conda environment "zk0"
- **Parallel Execution**: Use `pytest -n auto` for parallel test execution
- **Coverage**: Always include `--cov=src --cov-report=term-missing` for coverage reporting
- **Command Format**: `conda run -n zk0 python -m pytest -n auto --cov=src --cov-report=term-missing`

## Workflow Rules

**Summary:** Preserve context across tasks, validate constraints, document decisions

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
- [ ] VSCode is configured to use conda environment "zk0"
- [ ] SmolVLA and SO-100 focus is maintained
- [ ] Reference to quickstart-lerobot structure is considered
- [ ] Parent task context is fully understood
- [ ] Success criteria are clearly defined
- [ ] Quality standards are referenced

Before running tests:
- [ ] Tests run in conda environment "zk0" with `conda run -n zk0 python -m pytest`
- [ ] Use parallel execution with `-n auto` for multiple tests
- [ ] Include coverage reporting with `--cov=src --cov-report=term-missing`
- [ ] Ensure coverage remains above 80%

## Version History

- **v1.1.0** (2025-09-10): Added explicit production code quality standards prohibiting mocks in production code
- **v1.0.0** (2025-09-03): Initial creation with core zk0 project constraints