# Development Guide

This guide covers development workflows for the zk0 project, including testing, logging, guidelines, and contributing. It extracts key practices from the project setup, with references to the memory bank for deeper constraints (`.kilocode/rules/memory-bank/`).

For project status and roadmap, see [README.md](README.md). For architecture, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Testing

The project includes a comprehensive test suite built with pytest to ensure reliability of the SmolVLA federated learning implementation. Tests emphasize real scenarios over mocks, focusing on FL workflows and integrations.

### Test Structure

```
tests/
├── __init__.py
├── conftest.py                    # Pytest fixtures and configuration
├── unit/                          # Unit tests
│   ├── __init__.py
│   ├── test_basic_functionality.py # Basic functionality verification
│   ├── test_dataset_loading.py     # Dataset loading and validation
│   ├── test_dataset_splitting.py   # Partitioning logic
│   ├── test_dataset_validation.py  # Timestamp sync and quality checks
│   ├── test_error_handling.py      # Failure scenarios
│   ├── test_logger.py              # Logging utilities
│   ├── test_model_loading.py       # SmolVLA loading
│   └── test_smolvla_client.py      # Flower API integration
└── integration/                   # Integration tests
    ├── __init__.py
    └── test_integration.py        # End-to-end FL workflow
```

### Running Tests

#### Install Test Dependencies

```bash
# In zk0 environment
pip install -e .[test]
```

#### Run All Tests

Use conda environment for consistency:

```bash
# Activate and run
conda activate zk0
pytest -v

# With coverage
pytest --cov=src --cov-report=term-missing

# Fast mode (stop on failure, short traceback)
pytest -x --tb=short

# Parallel execution
pytest -n auto

# Combined fast mode
pytest -n auto -x --tb=short

# CI/CD mode (fail under 80% coverage)
pytest --cov=src --cov-fail-under=80
```

#### Run Specific Categories

```bash
# Unit tests only
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v
```

### Test Configuration

Defined in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = [
    "--verbose",
    "--tb=short",
    "--strict-markers",
    "--cov=src",
    "--cov-report=term-missing"
]
```

### Testing Standards

- **Real Scenarios**: Prioritize actual FL workflows, SmolVLA interactions, SO-100 processing.
- **No Mocks in Production**: Tests use real dependencies; fail-fast on missing env.
- **Coverage**: 80% minimum; focus on integration points (Flower ↔ SmolVLA).
- **Zk0-Specific**: Test parameter exchange, multi-repo partitioning, hash validation.
- **Execution**: Always in Docker (`zk0`) or conda (`zk0`) for consistency.
- **Parallel**: `pytest -n auto` with coverage reporting.

For full test details, see [memory-bank/tasks.md](.kilocode/rules/memory-bank/tasks.md) (e.g., FedProx debugging workflows).

## Logging

### Unified Logging Architecture

Uses **Loguru** for structured logging across components. Logs unify into `outputs/<timestamp>/simulation.log` for debugging FL rounds.

#### Key Features

- **Coordination**: Server creates log file; passes path to clients via Flower config.
- **Safety**: `enqueue=True` for multi-process (Ray) compatibility.
- **Format**: Includes client_id, PID, round, VRAM/RAM diagnostics.
- **Rotation**: 500MB files, 10-day retention, zip compression.
- **Integration**: Bridges Flower/Ray logs to Loguru.

#### Log Format Example

```
2025-09-20 16:23:45 | INFO     | client_0 | PID:1234 | src.client_app:fit:48 - Fit start - VRAM allocated: 2.5 GB
2025-09-20 16:23:46 | INFO     | client_0 | PID:1234 | src.client_app:fit:52 - Fit start - Host RAM used: 85.2%
```

#### Configuration

Auto-configured on run:

```bash
# Logs appear in outputs/simulation.log
./train.sh  # Or Docker equivalent
```

#### Log Locations

- **Unified**: `outputs/YYYY-MM-DD_HH-MM-SS/simulation.log` (all messages).
- **Server**: `outputs/.../server/server.log`.
- **Clients**: `outputs/.../clients/client_N/client.log`.
- **WandB**: Unified run with prefixed metrics (if enabled).

#### Custom Logging in Code

```python
from loguru import logger

logger.info("Training started")
logger.bind(vram_gb="2.5").info("VRAM usage")
logger.error("Model loading failed: {error}", error=str(e))
```

#### Troubleshooting Logs

- **Missing**: Check permissions/volumes.
- **Conflicts**: Use serialized GPU mode.
- **Analysis**: Grep for client_id or round_N.

See [`src/logger.py`](src/logger.py) for implementation. For advanced integration, reference [memory-bank/tech.md](.kilocode/rules/memory-bank/tech.md).

## Development Guidelines

### Mandatory Project Constraints

From memory bank – **MANDATORY** for all work:

1. **Working Directory**: Only within `/home/ivelin/zk0` (project root).
2. **Environment**: Use conda "zk0" or Docker "zk0" via `train.sh`.
3. **Focus**: SmolVLA + Flower + SO-100 datasets.
4. **No External Changes**: No mods outside root; read access to lerobot/flower.
5. **Quality**: 80% test coverage; reproducible with seeds.
6. **No Mocks in Prod**: Real dependencies; fail-fast on issues.
7. **Testing**: Integrate into pytest suite; no standalone scripts.

Full list: [memory-bank/architecture.md](.kilocode/rules/memory-bank/architecture.md#project-constraints).

### Best Practices

- **Code Style**: PEP 8, type hints, docstrings.
- **Modularity**: Separate concerns (e.g., task.py for training).
- **Error Handling**: Raise RuntimeError for missing deps.
- **Reproducibility**: Pin deps in requirements.txt; use seeds (e.g., 42).
- **Tool Usage**: Batch file reads; diff format for edits.
- **Context**: Read memory bank at task start; update on changes.
- **Performance**: GPU optimization, AMP; monitor VRAM.

### Workflow

- **New Task**: Read memory bank; use [Memory Bank Instructions](.kilocode/rules/memory-bank/memory-bank-instructions.md).
- **Edits**: Use apply_diff for targeted changes; full content for new files.
- **Testing**: Run after changes: `pytest -n auto --cov=src`.
- **CI**: GitHub Actions for auto-testing on push/PR.

For repetitive tasks (e.g., FedProx tuning), see [memory-bank/tasks.md](.kilocode/rules/memory-bank/tasks.md).

## Contributing

We welcome contributions! At Beta stage, focus on:

- **Node Operators**: Join with SO-100 arm + RTX 3090+ GPU for data/compute.
- **Code**: Bug fixes, features, docs, tests.
- **Process**:
  1. Fork repository.
  2. Create feature branch.
  3. Commit changes (lint with Ruff).
  4. Run tests (`pytest`).
  5. Submit PR.

See [CONTRIBUTING.md](CONTRIBUTING.md) for details. Adhere to constraints; suggest memory bank updates for significant changes.

For questions, reference [memory-bank/brief.md](.kilocode/rules/memory-bank/brief.md) or open an issue.