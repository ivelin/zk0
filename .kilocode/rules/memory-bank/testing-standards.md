# Testing Standards for zk0 Project

## Testing Quality Standards
- **Real Scenario Emphasis**: Prioritize testing actual federated learning workflows, SmolVLA model interactions, and SO-100 dataset processing over isolated mocking
- **Zk0-Specific Scenarios**: Focus on federated learning client-server communication, model aggregation with Flower strategies, and SmolVLA parameter handling in distributed environments
- **Integration Points**: Test real data flow between components (e.g., Flower ↔ SmolVLA, client ↔ server parameter exchange)
- **Parameter Data Types**: Validate correct handling of SmolVLA tensors, Flower NumPy arrays, and SO-100 dataset formats
- **Dependency Verification**: Include tests that verify actual dependency availability and integration in zk0 environment
- **User Journey Priority**: Focus on complete federated learning rounds and robotics task workflows rather than method-level coverage
- **Real Environment Testing**: Test against actual conda zk0 environment, real imports, and SO-100 data, not mocked states
- **Remove Low-Value Tests**: Delete tests that only exercise mocking without testing real federated learning behavior

## Test Quality Checklist
Before writing/keeping a test, ask:
- [ ] Does this test verify real federated learning functionality or just mocking?
- [ ] Does this test check actual dependency availability in zk0 environment?
- [ ] Does this test cover a complete federated learning round or robotics workflow?
- [ ] Does this test verify integration points (e.g., Flower ↔ SmolVLA parameter exchange)?
- [ ] Does this test validate correct handling of SmolVLA tensors and SO-100 data types?
- [ ] Does this test simulate real zk0-specific scenarios (client-server communication, model aggregation)?
- [ ] Would this test fail if the real federated learning functionality broke?

## Testing Best Practices
- **Dependency Tests**: Test that required packages are actually installed and working in zk0 environment
- **Integration Tests**: Test parameter passing and data flow between libraries (e.g., Flower ↔ SmolVLA, client ↔ server parameter exchange)
- **Parameter Data Type Validation**: Ensure SmolVLA tensors, Flower NumPy arrays, and SO-100 dataset formats are handled correctly
- **Federated Learning Scenarios**: Test client-server communication, model aggregation with FedAvg/FedProx, and asynchronous updates
- **User Journey Tests**: Test complete federated learning rounds and robotics task workflows from start to finish
- **Error Handling**: Test real error scenarios in distributed environments, not just mocked exceptions
- **Parallel Test Execution**: Leverage `pytest -n auto` for running tests in parallel to improve efficiency and reduce execution time
- **Coverage Guidelines**: Aim for 80% coverage on critical user journeys, focusing on real federated learning workflows rather than artificial percentages

## Test Readability Guidelines
- **Clear Test Names**: Use descriptive names that explain the scenario being tested (e.g., `test_federated_round_with_smolvla_aggregation`)
- **Arrange-Act-Assert Structure**: Organize tests with clear sections for setup, execution, and verification
- **Meaningful Assertions**: Use assertions that clearly indicate what is being validated (e.g., `assert model_parameters_updated` instead of generic checks)
- **Comments for Complex Logic**: Add comments explaining complex federated learning logic or parameter transformations
- **Consistent Naming Conventions**: Follow PEP 8 for test function and variable names

## Test Maintainability Guidelines
- **Modular Test Fixtures**: Create reusable fixtures for common setups (e.g., mock clients, sample SO-100 data)
- **Avoid Test Interdependencies**: Ensure tests can run independently without relying on other test results
- **Regular Test Cleanup**: Remove outdated tests and refactor duplicated code into helper functions
- **Version Control Integration**: Keep tests aligned with code changes and update them promptly
- **Documentation**: Include docstrings in test files explaining the purpose and context of test suites

## Test Execution Guidelines
> **Strong Recommendation**: Always use parallel execution with `pytest -n auto` when running multiple tests to significantly improve efficiency and reduce execution time, especially in development and CI/CD environments.

- **Development Mode**: Use `pytest -x --tb=short` for fast iteration - stops on first failure with concise output
- **Parallel Execution**: **Strongly encouraged** - Use `pytest -n auto` for faster test runs on multi-core systems. This is the default recommended approach for running test suites efficiently.
- **Combined Fast Mode**: Use `pytest -n auto -x --tb=short` for maximum speed during development - combines parallel execution with early failure detection
- **Full Test Suite**: Use `pytest --cov=src --cov-report=html` for complete coverage analysis (consider adding `-n auto` for faster coverage runs)
- **CI/CD Mode**: Use `pytest --cov=src --cov-fail-under=80` for automated quality gates (always include `-n auto` for optimal CI performance)

## Test Refactoring Guidelines
- **Identify Duplication**: Look for repeated test patterns and extract into parameterized tests or shared fixtures
- **Simplify Complex Tests**: Break down large integration tests into smaller, focused unit tests where appropriate
- **Update for API Changes**: Refactor tests when SmolVLA or Flower APIs change to maintain relevance
- **Performance Optimization**: Optimize slow tests by using fixtures or reducing unnecessary setup
- **Code Review Integration**: Include test refactoring in code review processes to maintain quality