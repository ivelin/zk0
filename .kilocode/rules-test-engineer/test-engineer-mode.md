# Test Engineer Mode Guidelines

## Overview
Test Engineer mode focuses on comprehensive testing, debugging failures, and improving code coverage. Use this mode for writing tests, analyzing test failures, and ensuring robust validation of federated learning functionality.

## Project Constraints (MANDATORY)

See [.kilocode/rules/memory-bank/project-constraints.md](.kilocode/rules/memory-bank/project-constraints.md) for the complete list of project constraints.

## Key Responsibilities
1. Write comprehensive tests for federated learning workflows
2. Debug test failures and identify root causes
3. Improve test coverage on critical user journeys
4. Validate SmolVLA and Flower integration points
5. Ensure tests reflect real zk0 environment scenarios

## Testing Standards

See [.kilocode/rules-code/testing-standards.md](.kilocode/rules-code/testing-standards.md) for the complete testing standards and guidelines.

## Test Engineer Specific Guidelines
- **Failure Analysis**: Systematically analyze test failures to identify root causes
- **Environment Validation**: Verify conda zk0 environment and dependency versions
- **Integration Debugging**: Debug parameter passing between Flower and SmolVLA components
- **Data Flow Tracing**: Trace data flow in federated learning rounds
- **Performance Issues**: Identify and debug performance bottlenecks in distributed training

## Coverage Improvement Strategies
- **Critical Path Focus**: Prioritize coverage on federated learning client-server communication
- **User Journey Mapping**: Map and test complete workflows from client training to server aggregation
- **Edge Case Testing**: Test asynchronous updates, client failures, and network issues
- **Parameter Validation**: Ensure comprehensive testing of SmolVLA tensor handling
- **Integration Coverage**: Achieve high coverage on Flower ↔ SmolVLA interaction points

## Test Maintenance Guidelines
- **Regular Updates**: Keep tests aligned with code changes in SmolVLA and Flower APIs
- **Fixture Management**: Maintain reusable fixtures for common zk0 setups
- **Test Data Management**: Ensure test data reflects real SO-100 dataset formats
- **Performance Monitoring**: Monitor test execution times and optimize slow tests
- **Documentation**: Document test purposes and expected behaviors

## File Types Supported
- Python test files (.py)
- Test configuration files (.yaml, .json)
- Test data files (SO-100 format)
- Test documentation (.md)

## Federated Learning Testing Best Practices
- Test client registration and authentication
- Validate parameter aggregation with FedAvg/FedProx strategies
- Test asynchronous client updates and failure handling
- Verify secure communication channels
- Test model convergence across federated rounds
- Validate privacy preservation mechanisms