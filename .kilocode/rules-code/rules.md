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
- Maintain meaningful test coverage (focus on critical paths, not just 80% metric)
- Use 'ruff check --fix' to automatically fix Python formatting errors
- Run 'ruff check --fix' at the end of every task that involves changing Python code
- **Diff Format for Edits**: Always present code edits in diff format for easier review and approval. Use unified diff format with clear context showing what is being changed, added, or removed.
- **Annotated Diff Display**: Before using the code edit tool, always display an annotated diff of the proposed changes in a clear, easy-to-review format. The diff should use standard notation such as '+' for additions and '-' for deletions, highlight the specific lines or sections affected, and include a brief explanation of what is being changed and the rationale for it. This applies to any text file edits, not just code. The diff should be presented before applying the changes to allow for review and approval. If the changes are complex, consider breaking them into smaller, logical diffs for better clarity.
- **Preferred Edit Tool**: For surgical edits requiring review, prefer `apply_diff` over `search_and_replace` as it provides clearer diff visualization and more precise control over changes. Use `search_and_replace` only for simple, non-surgical text replacements that don't require detailed diff review.

## Testing Standards

See [.kilocode/rules-code/testing-standards.md](.kilocode/rules-code/testing-standards.md) for the complete testing standards and guidelines.

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