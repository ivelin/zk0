# Repetitive Task Workflows

**Created**: 2025-09-06
**Last Updated**: 2025-09-06
**Version**: 1.0.0
**Author**: Kilo Code

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

## Appendix: Task Context Template

### Task Information
**Task ID**: [Unique identifier]
**Parent Task**: [Parent task ID if applicable]
**Created**: [Date/time]
**Mode**: [Current mode]
**Priority**: [High/Medium/Low]

### Project Constraints (MANDATORY)
See [.kilocode/rules/memory-bank/project-constraints.md](.kilocode/rules/memory-bank/project-constraints.md) for the complete list of project constraints.

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