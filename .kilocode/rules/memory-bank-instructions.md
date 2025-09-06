# Memory Bank Instructions for Kilo Code

This file contains instructions for Kilo Code to properly use the Memory Bank system for this project.

## Project Overview
This is a federated learning project for robotics AI tasks using SmolVLA model and SO-100 real-world datasets with Flower framework.

## Memory Bank Structure
- brief.md - High-level project overview and goals
- product.md - Product vision and problem solving
- context.md - Current work focus and recent changes
- architecture.md - System architecture and technical decisions
- tech.md - Technologies, frameworks, and development setup
- tasks.md - Repetitive task workflows

## Key Project Constraints

See [.kilocode/rules/memory-bank/project-constraints.md](.kilocode/rules/memory-bank/project-constraints.md) for the complete list of project constraints.

## Environment Setup
- Use conda environment "zk0" for all Python operations
- VSCode is configured to automatically detect and use the zk0 environment
- If manual execution needed: use `conda run -n zk0 python ...`

## Development Workflow
1. Read all memory bank files at the start of each session
2. Maintain context across sessions using memory bank updates
3. Document new patterns and workflows in appropriate files
4. Update memory bank after significant changes

## Quality Standards
- Maintain meaningful test coverage (focus on critical paths, not just 80% metric)
- Follow PEP 8 style guidelines
- Include type hints and comprehensive docstrings
- Keep README and documentation current

## Testing Standards
- **Avoid Mock-Only Tests**: Tests must verify real functionality, not just mocking
- **Dependency Verification**: Include tests that verify actual dependency availability
- **Integration Focus**: Prioritize tests of component interactions over isolated units
- **User Journey Priority**: Focus on complete workflows rather than method coverage
- **Real Environment Testing**: Test against actual imports and data, not mocked states