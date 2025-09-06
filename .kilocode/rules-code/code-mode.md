# Code Mode Guidelines

## Overview
Code mode focuses on writing, modifying, and refactoring code. Use this mode for implementing features, fixing bugs, and making code improvements.

## Project Constraints (MANDATORY)

See [.kilocode/rules/memory-bank/project-constraints.md](.kilocode/rules/memory-bank/project-constraints.md) for the complete list of project constraints.

## Key Responsibilities
1. Implement features according to specifications
2. Write clean, maintainable code
3. Follow established patterns and conventions
4. Ensure code quality and testing

## Implementation Standards
- Follow PEP 8 style guidelines
- Write comprehensive docstrings
- Include type hints where appropriate
- Maintain test coverage above 80%

## File Types Supported
- Python files (.py)
- Configuration files (.yaml, .toml, .json)
- Documentation files (.md)
- Shell scripts (.sh)

## Federated Learning Best Practices
- Implement proper data partitioning for clients
- Use Flower strategies (FedAvg, FedProx) appropriately
- Ensure model aggregation handles SmolVLA parameters correctly
- Implement secure communication between clients and server
- Handle asynchronous client updates
- Validate data privacy and compliance with FL requirements
- Test with simulation mode before deployment
- Monitor communication efficiency and resource usage