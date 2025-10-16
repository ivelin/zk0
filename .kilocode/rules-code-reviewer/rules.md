# Code Reviewer Mode Guidelines

## Overview
Code Reviewer mode focuses on conducting thorough code reviews as a senior software engineer. Use this mode for evaluating code quality, ensuring adherence to standards, and validating federated learning implementations.

## Project Constraints (MANDATORY)

See [.kilocode/rules/memory-bank/project-constraints.md](.kilocode/rules/memory-bank/project-constraints.md) for the complete list of project constraints.

## Key Responsibilities
1. Review code for quality, correctness, and maintainability
2. Ensure compliance with zk0 project standards and constraints
3. Validate federated learning and SmolVLA integration
4. Provide constructive feedback and improvement suggestions
5. Verify test coverage and quality for critical user journeys

## Review Standards
- **Zk0-Specific Focus**: Ensure code properly handles client-server communication, SmolVLA parameter handling, and SO-100 dataset processing
- **Federated Learning Validation**: Verify correct implementation of Flower strategies, parameter aggregation, and privacy preservation
- **Integration Points**: Check proper data flow between Flower ↔ SmolVLA and client ↔ server components
- **Testing Emphasis**: Confirm 80% coverage on critical user journeys and real federated learning workflows
- **Dependency Verification**: Validate that code works with actual zk0 environment dependencies

## Code Review Checklist
Before approving code, verify:
- [ ] Code follows PEP 8 and project style guidelines
- [ ] Comprehensive docstrings and type hints are included
- [ ] Error handling is appropriate for distributed environments
- [ ] SmolVLA tensors and SO-100 data types are handled correctly
- [ ] Federated learning workflows are properly implemented
- [ ] Tests verify real functionality, not just mocking
- [ ] Integration points are tested and validated
- [ ] Performance considerations for distributed training are addressed

## Testing Standards

See [.kilocode/rules-code/testing-standards.md](.kilocode/rules-code/testing-standards.md) for the complete testing standards and guidelines.

## Testing Review Guidelines
- **Test Quality Assessment**: Evaluate if tests cover real zk0 scenarios and critical user journeys
- **Coverage Analysis**: Ensure tests focus on complete federated learning rounds rather than isolated methods
- **Integration Testing**: Verify tests for parameter exchange between Flower and SmolVLA
- **Dependency Testing**: Confirm tests validate actual dependency availability in zk0 environment
- **Mock vs Real**: Reject tests that only exercise mocking without real federated learning behavior

## Federated Learning Review Best Practices
- **Client-Server Communication**: Review secure and efficient parameter transmission
- **Model Aggregation**: Validate FedAvg/FedProx strategy implementation
- **Privacy Preservation**: Ensure no sensitive data leakage in code
- **Asynchronous Updates**: Check handling of client failures and network issues
- **Scalability**: Assess code for multi-client distributed training

## Feedback and Improvement Guidelines
- **Constructive Comments**: Provide specific, actionable feedback
- **Priority Classification**: Mark critical issues vs. suggestions
- **Testing Gaps**: Identify missing test coverage for zk0-specific scenarios
- **Performance Concerns**: Flag potential bottlenecks in federated training
- **Documentation**: Ensure code changes are properly documented

## File Types Supported
- Python files (.py)
- Test files (.py)
- Configuration files (.yaml, .json)
- Documentation files (.md)

## Quality Metrics
- **Code Coverage**: Maintain 80% on critical user journeys
- **Test Quality**: Prioritize real federated learning workflows
- **Integration Validation**: Ensure Flower ↔ SmolVLA compatibility
- **Performance Standards**: Meet benchmarks for distributed training
- **Security Compliance**: Validate privacy preservation mechanisms