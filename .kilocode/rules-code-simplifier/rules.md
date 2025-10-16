# Code Simplifier Mode Guidelines

## Overview
Code Simplifier mode focuses on refactoring code to make it clearer, more concise, and easier to maintain. Use this mode for simplifying complex logic, removing redundancy, and improving code readability while preserving functionality.

## Project Constraints (MANDATORY)

See [.kilocode/rules/memory-bank/project-constraints.md](.kilocode/rules/memory-bank/project-constraints.md) for the complete list of project constraints.

## Key Responsibilities
1. Identify and simplify complex code structures
2. Remove code duplication and redundancy
3. Improve code readability and maintainability
4. Ensure refactoring preserves federated learning functionality
5. Maintain test coverage and quality during simplifications

## Testing Standards

See [.kilocode/rules-code/testing-standards.md](.kilocode/rules-code/testing-standards.md) for the complete testing standards and guidelines.

## Simplification Standards
- **Zk0-Specific Focus**: Preserve SmolVLA parameter handling, Flower integration, and SO-100 dataset processing
- **Federated Learning Integrity**: Ensure simplifications don't break client-server communication or model aggregation
- **Testing Preservation**: Maintain 80% coverage on critical user journeys during refactoring
- **Performance Maintenance**: Avoid simplifications that impact distributed training performance
- **Dependency Awareness**: Respect actual zk0 environment dependencies and integrations

## Code Simplification Checklist
Before simplifying code, verify:
- [ ] Simplification preserves original functionality
- [ ] SmolVLA tensors and SO-100 data types are handled correctly
- [ ] Federated learning workflows remain intact
- [ ] Tests still pass and cover real scenarios
- [ ] Performance benchmarks are maintained
- [ ] Code remains readable and maintainable

## Refactoring Best Practices
- **Incremental Changes**: Make small, testable changes to avoid breaking functionality
- **Test-First Refactoring**: Run tests before and after each simplification
- **Federated Learning Awareness**: Consider distributed training implications
- **Parameter Handling**: Ensure SmolVLA parameter transformations are preserved
- **Integration Points**: Maintain Flower ↔ SmolVLA data flow integrity

## Testing During Simplification
- **Coverage Verification**: Ensure critical user journeys remain covered
- **Integration Testing**: Validate component interactions after changes
- **Real Scenario Testing**: Test actual federated learning rounds, not just mocks
- **Performance Testing**: Check for performance regressions in distributed environments
- **Dependency Testing**: Verify actual dependency availability post-refactoring

## Simplification Guidelines
- **Remove Redundancy**: Eliminate duplicate code while preserving federated learning logic
- **Simplify Logic**: Break down complex federated learning algorithms into clearer steps
- **Improve Naming**: Use descriptive names for SmolVLA parameters and Flower strategies
- **Extract Methods**: Separate concerns in client-server communication
- **Optimize Structures**: Simplify data structures for SO-100 datasets

## Quality Assurance
- **Pre-Simplification Testing**: Run full test suite before changes
- **Post-Simplification Validation**: Verify all tests pass and coverage maintained
- **Peer Review**: Have changes reviewed for federated learning correctness
- **Documentation Updates**: Update docstrings and comments for simplified code
- **Performance Monitoring**: Ensure no degradation in training or inference times

## File Types Supported
- Python files (.py)
- Configuration files (.yaml, .json)
- Documentation files (.md)

## Federated Learning Simplification Best Practices
- Simplify parameter aggregation logic without losing FedAvg/FedProx accuracy
- Streamline client-server communication protocols
- Optimize SmolVLA model loading and inference
- Reduce complexity in asynchronous update handling
- Maintain privacy preservation mechanisms during simplification