# Development

This guide covers development workflows for the zk0 project, including testing, logging, guidelines, and contributing. It extracts key practices from the project setup.

For project status and roadmap, see [README.md](README.md). For architecture, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Testing

The project includes a comprehensive test suite built with pytest to ensure reliability of the SmolVLA federated learning implementation. Tests emphasize real scenarios over mocks, focusing on FL workflows and integrations.

### Test Structure

```
tests/
├── unit/                          # Unit tests for individual components
└── integration/                   # Integration tests for end-to-end workflows
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

# CI/CD mode (fail under 30% coverage)
pytest --cov=src --cov-fail-under=30
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
- **Coverage**: 30% minimum; focus on integration points (Flower ↔ SmolVLA).
- **Zk0-Specific**: Test parameter exchange, multi-repo partitioning, hash validation.
- **Execution**: Always in Docker (`zk0`) or conda (`zk0`) for consistency.
- **Parallel**: `pytest -n auto` with coverage reporting.

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

See [`src/logger.py`](src/logger.py) for implementation.

## Development Guidelines

### Mandatory Project Constraints

The following constraints are mandatory for all work:

1. **Working Directory**: Only within `/home/ivelin/zk0` (project root).
2. **Environment**: Use conda "zk0" or Docker "zk0" via `train.sh`.
3. **Focus**: SmolVLA + Flower + SO-100 datasets.
4. **No External Changes**: No mods outside root; read access to lerobot/flower.
5. **Quality**: 30% test coverage; reproducible with seeds.
6. **No Mocks in Prod**: Real dependencies; fail-fast on issues.
7. **Testing**: Integrate into pytest suite; no standalone scripts.

### Best Practices

- **Code Style**: PEP 8, type hints, docstrings.
- **Modularity**: Separate concerns (e.g., task.py for training).
- **Error Handling**: Raise RuntimeError for missing deps.
- **Reproducibility**: Pin deps in pyproject.toml; use seeds (e.g., 42).
- **Tool Usage**: Batch file reads; diff format for edits.
- **Context Management**: Maintain project context through documentation updates and reviews.
- **Performance**: GPU optimization, AMP; monitor VRAM.

### Workflow

- **New Task**: Follow established development practices.
- **Edits**: Use apply_diff for targeted changes; full content for new files.
- **Testing**: Run after changes: `pytest -n auto --cov=src`.
- **Prepare for Commit**: Standardized workflow for version management, full test suite with coverage, documentation consistency, and git operations to ensure quality.
- **CI**: GitHub Actions using Docker-based testing with the project's Dockerfile for isolated, reproducible runs; no caching to prevent space issues; auto-testing on push/PR.

## CI/CD Pipeline

### Docker-Based Testing

The CI pipeline uses the project's `Dockerfile.ci` (CPU-only base image) to build a containerized environment for testing, ensuring isolation and reproducibility. This avoids dependency conflicts and space issues from direct pip installs. CI runs on CPU-only due to GitHub-hosted runner limitations (no GPU hardware).

#### CI CPU-Only Mode

GitHub Actions runners (ubuntu-latest) lack NVIDIA GPU drivers, causing Docker `--gpus all` to fail with "could not select device driver". CI uses:
- **Base Image**: `huggingface/lerobot-cpu:latest` (CPU-only)
- **Environment**: `CUDA_VISIBLE_DEVICES=""` to force CPU in PyTorch/Lerobot
- **Tests**: All pass on CPU; mocks handle GPU-dependent code

Local GPU runs remain unaffected via `train.sh --docker` (uses main Dockerfile with GPU base).

#### Local Simulation

To test CI locally:

```bash
# Build the CI image (CPU-only)
docker build -f Dockerfile.ci -t zk0-ci .

# Run tests in container (CPU-only, as in CI)
docker run --rm \
  -v $(pwd):/workspace \
  -v /tmp/coverage:/coverage \
  -e CUDA_VISIBLE_DEVICES="" \
  -e COVERAGE_FILE=/coverage/.coverage \
  zk0-ci \
  bash -c "
    cd /workspace &&
    uv pip install -e '.[test]' &&
    pytest tests/ --cov=src --cov-report=term-missing -n auto &&
    coverage combine &&
    coverage xml -o /coverage/coverage.xml
  "

# For local GPU testing (uses main Dockerfile)
docker build -t zk0-gpu .
docker run --rm -v $(pwd):/workspace --gpus all zk0-gpu \
  bash -c "cd /workspace && uv pip install -e '.[test]' && pytest tests/ --cov=src --cov-report=xml -n auto"
```

#### Troubleshooting

- **Space Issues**: If Docker runs out of space, prune: `docker system prune -f`
- **GPU Not Available**: Use CPU mode (remove `--gpus all`, set `CUDA_VISIBLE_DEVICES=""`)
- **Build Failures**: Ensure Dockerfile.ci and requirements.txt are up-to-date
- **Test Failures**: Check logs for missing dependencies or environment issues
- **CI Failures**: Verify GitHub runners; no GPU passthrough available

#### Coverage Collection in Parallel Tests

CI uses parallel pytest execution (`-n auto`) with coverage collection, which can cause SQLite database locking issues in Docker due to concurrent writes to the `.coverage` file. This is resolved by:

- **Configuration**: `.coveragerc` enables parallel mode (`parallel = true`), directing each worker to write to separate files (e.g., `.coverage.pid.worker`).
- **Post-Processing**: After tests, `coverage combine` merges files, then `coverage xml` generates the final report.
- **Environment**: `COVERAGE_FILE=/coverage/.coverage` ensures files are written to the mounted `/coverage` directory, avoiding host permission issues.
- **Branch Coverage**: Disabled (`branch = false`) to reduce SQLite load and complexity.

If issues persist, disable parallel in CI by setting `parallel: 1` in the matrix (keeps local parallel via pyproject.toml).

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

See [CONTRIBUTING.md](CONTRIBUTING.md) for details. Adhere to constraints; suggest improvements for significant changes.

For questions, see [README.md](README.md) or open an issue.