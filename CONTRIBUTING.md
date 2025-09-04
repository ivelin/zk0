# Contributing to Federated Learning for Robotics AI

Thank you for your interest in contributing to this project! We welcome contributions from the community to help improve our federated learning implementation for robotics AI tasks.

## Table of Contents

- [Ways to Contribute](#ways-to-contribute)
- [Node Operators](#node-operators)
- [Code Contributors](#code-contributors)
- [Development Setup](#development-setup)
- [Code Style and Guidelines](#code-style-and-guidelines)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)
- [Community Guidelines](#community-guidelines)
- [License](#license)

## Ways to Contribute

There are several ways you can contribute to this project:

1. **Node Operators**: Join the federated network with your hardware and data
2. **Code Contributors**: Improve the codebase, add features, fix bugs
3. **Documentation**: Help improve documentation and tutorials
4. **Testing**: Report bugs, test new features, improve test coverage
5. **Feedback**: Share your experience and suggestions

## Node Operators

At this Beta stage, we're particularly interested in node operators who can contribute to the federated learning network.

### Requirements

- **Hardware**: LeRobot SO100 arm (or newer SO101 version)
- **Compute**: Local machine with RTX 3090 GPU or better, compatible with LeRobot library
- **Network**: Stable internet connection for federated communication
- **Data**: Unique training data from your robotics setup

### Getting Started as a Node Operator

1. **Hardware Setup**:
   - Assemble and calibrate your LeRobot arm according to the official documentation
   - Ensure your GPU drivers are up to date
   - Verify LeRobot library compatibility

2. **Software Setup**:
   - Follow the installation instructions in the main README.md
   - Configure your environment variables
   - Test the basic functionality with simulation mode

3. **Join the Network**:
   - Contact the project maintainers to get setup instructions
   - Provide details about your hardware configuration
   - Receive authentication credentials for the federated network

4. **Contribute Data**:
   - Collect unique robotics manipulation data
   - Ensure data quality and annotation accuracy
   - Participate in federated training rounds

### Node Operator Responsibilities

- Maintain hardware availability during agreed training windows
- Ensure data privacy and security compliance
- Monitor training performance and report issues
- Participate in regular network health checks

## Code Contributors

We welcome code contributions to improve the project. Here's how to get started:

### Development Setup

1. **Prerequisites**:
   - Python 3.10+
   - Conda or virtualenv
   - Git

2. **Clone and Setup**:
   ```bash
   git clone <repository-url>
   cd <project-directory>
   conda create -n zk0 python=3.10 -y
   conda activate zk0
   pip install -r requirements.txt
   pip install -e .
   ```

3. **Environment Configuration**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Verify Setup**:
   ```bash
   python -c "import src; print('Setup successful')"
   ```

### Code Style and Guidelines

- **Python Style**: Follow PEP 8 guidelines
- **Type Hints**: Use type annotations for function parameters and return values
- **Docstrings**: Include comprehensive docstrings for all public functions
- **Imports**: Organize imports alphabetically, with standard library first
- **Naming**: Use descriptive variable and function names
- **Error Handling**: Implement proper exception handling

#### Code Formatting

We use the following tools for code quality:

- **Black**: For code formatting
- **isort**: For import sorting
- **flake8**: For linting
- **mypy**: For type checking

Run formatting before committing:
```bash
black src/ tests/
isort src/ tests/
flake8 src/ tests/
mypy src/
```

### Testing

#### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test categories
pytest tests/unit/ -v
pytest tests/integration/ -v
```

#### Writing Tests

- Write unit tests for individual components
- Include integration tests for end-to-end workflows
- Use descriptive test names and docstrings
- Mock external dependencies appropriately
- Aim for 80%+ test coverage

Example test structure:
```python
import pytest
from unittest.mock import Mock

def test_federated_client_initialization():
    """Test that FederatedClient initializes correctly."""
    config = {"num_clients": 10, "rounds": 100}
    client = FederatedClient(config)

    assert client.num_clients == 10
    assert client.rounds == 100
```

## Submitting Changes

### Pull Request Process

1. **Fork the Repository**: Create your own fork of the project
2. **Create a Branch**: Use descriptive branch names
   ```bash
   git checkout -b feature/add-new-algorithm
   git checkout -b bugfix/fix-memory-leak
   git checkout -b docs/update-contributing-guide
   ```

3. **Make Changes**: Implement your changes following the guidelines above
4. **Test Thoroughly**: Ensure all tests pass and add new tests if needed
5. **Update Documentation**: Update README.md or other docs if necessary
6. **Commit Changes**: Write clear, concise commit messages
   ```bash
   git commit -m "feat: add support for FedProx algorithm

   - Implement FedProx aggregation strategy
   - Add configuration parameters
   - Update tests and documentation"
   ```

7. **Push to Branch**: Push your changes to your fork
   ```bash
   git push origin feature/add-new-algorithm
   ```

8. **Create Pull Request**:
   - Go to the original repository
   - Click "New Pull Request"
   - Select your branch
   - Fill out the PR template with:
     - Clear description of changes
     - Screenshots/videos for UI changes
     - Test results
     - Breaking changes (if any)

### PR Review Process

- Maintainers will review your PR within 1-2 business days
- Address any feedback or requested changes
- Once approved, your PR will be merged
- Contributors retain copyright but grant license to the project

## Reporting Issues

### Bug Reports

When reporting bugs, please include:

- **Clear Title**: Summarize the issue concisely
- **Description**: Detailed explanation of the problem
- **Steps to Reproduce**: Step-by-step instructions
- **Expected Behavior**: What should happen
- **Actual Behavior**: What actually happens
- **Environment**: OS, Python version, hardware specs
- **Logs/Error Messages**: Include relevant output
- **Screenshots**: If applicable

### Feature Requests

For new features, please provide:

- **Use Case**: Why is this feature needed?
- **Proposed Solution**: How should it work?
- **Alternatives**: Other approaches considered
- **Impact**: How will this affect existing functionality?

### Issue Labels

We use the following labels to categorize issues:
- `bug`: Something isn't working
- `enhancement`: New feature or improvement
- `documentation`: Documentation updates needed
- `question`: General questions or discussions
- `help wanted`: Good first issues for new contributors
- `node-operator`: Issues related to node operation

## Community Guidelines

### Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please:

- Be respectful and constructive in communications
- Welcome newcomers and help them get started
- Focus on the merit of ideas, not the person proposing them
- Report any unacceptable behavior to maintainers

### Communication Channels

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For general questions and community discussion
- **Pull Request Comments**: For code review discussions

### Getting Help

If you need help getting started:

1. Check the README.md for basic setup instructions
2. Search existing issues for similar problems
3. Ask questions in GitHub Discussions
4. Contact maintainers directly for sensitive matters

## License

By contributing to this project, you agree that your contributions will be licensed under the same license as the project (see LICENSE file). This ensures that the project remains open source and accessible to the community.

---

Thank you for contributing to Federated Learning for Robotics AI! Your efforts help advance the field of federated robotics and make this technology more accessible to everyone.