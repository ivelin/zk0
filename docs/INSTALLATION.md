# Installation Guide

This guide provides detailed instructions for setting up the zk0 project, including cloning the repository, creating the environment, installing dependencies, and configuring environment variables.

## Prerequisites

- Python 3.10+
- Conda (recommended for environment management)
- Git
- NVIDIA GPU with CUDA support (optional but recommended for training)
- Docker (optional, for isolated execution)

## Clone the Project

Clone this project repository to your local machine:

```shell
git clone <repository-url> .
cd <project-directory>
```

This will set up the project in the current directory with the following structure:

```
project-root/
├── .env.example
├── .gitignore
├── LICENSE
├── pyproject.toml      # Project metadata like dependencies and configs
├── README.md
├── requirements.txt    # Pinned dependencies for reproducibility
├── train.sh
├── .kilocode/          # Memory bank and project constraints
├── .vscode/            # VS Code configuration
├── src/
│   ├── __init__.py
│   ├── client_app.py   # Defines your ClientApp
│   ├── server_app.py   # Defines your ServerApp
│   ├── task.py         # Training and evaluation logic
│   ├── utils.py        # Utility functions
│   ├── logger.py       # Logging configuration
│   ├── visualization.py # Evaluation visualization tools
│   ├── wandb_utils.py  # Weights & Biases integration
│   └── configs/        # Configuration management
│       ├── __init__.py
│       └── datasets.py # Dataset configuration
├── tests/              # Test suite
│   ├── __init__.py
│   ├── conftest.py     # Pytest configuration
│   ├── unit/           # Unit tests
│   │   ├── test_basic_functionality.py
│   │   ├── test_dataset_loading.py
│   │   ├── test_dataset_splitting.py
│   │   ├── test_dataset_validation.py
│   │   ├── test_error_handling.py
│   │   ├── test_logger.py
│   │   ├── test_model_loading.py
│   │   └── test_smolvla_client.py
│   └── integration/    # Integration tests
│       ├── __init__.py
│       └── test_integration.py
└── outputs/            # Runtime outputs (created during execution)
```

## Set Up Conda Environment

Create and activate the `zk0` conda environment:

```bash
# Create the zk0 environment (if it doesn't exist)
conda create -n zk0 python=3.10 -y

# Install required system dependencies
conda install ffmpeg=7.1.1 -c conda-forge

# Activate the zk0 environment
conda activate zk0
```

## Install Dependencies and Project

Install the pinned dependencies and the `zk0` package:

```bash
# Install dependencies
pip install -r requirements.txt

# Install the project in editable mode
pip install -e .
```

**Note**: The project uses Flower 1.21.0 (latest version), Ray 2.31.0, and LeRobot 0.3.3 for optimal performance.

## Environment Variables

Before running the project, set up your environment variables:

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit the `.env` file and configure the following variables:
   - `GITHUB_TOKEN`: Your GitHub personal access token for API access
   - `GITHUB_PERSONAL_ACCESS_TOKEN`: Alternative GitHub token (can be the same as GITHUB_TOKEN)
   - `GITHUB_TOOLSETS`: Comma-separated list of GitHub toolsets to use
   - `GITHUB_READ_ONLY`: Set to 'true' for read-only access, 'false' for full access
   - `HF_TOKEN`: Hugging Face token for model pushing (optional, for Hub integration)

These variables are used for GitHub integration, API access, and Hugging Face Hub throughout the federated learning workflow.

## Choose Training Parameters

You can leave the default parameters for an initial quick test. It will run for 100 rounds sampling 10 clients per round. SmolVLA is memory-efficient, allowing for more clients to participate. For best results, total number of training rounds should be over 100,000: `num-server-rounds` * `local_epochs` > 50,000.

Adjust these parameters in the `pyproject.toml` file under `[tool.flwr.app.config]` or via run-config overrides.

**✅ Successfully Tested**: The federated learning simulation has been tested and runs successfully for 100 rounds with 10 clients, completing in approximately 50 seconds.

## Docker Alternative (Optional)

For reproducible and isolated execution, build and use the Docker image:

```bash
# Build the Docker image
docker build -t zk0 .

# Run with Docker (example)
docker run --gpus all --shm-size=10.24gb \
  -v $(pwd):/workspace \
  -v $(pwd)/outputs:/workspace/outputs \
  -v /tmp:/tmp \
  -w /workspace \
  zk0 flwr run . local-simulation-serialized-gpu --run-config "num-server-rounds=2"
```

See [RUNNING.md](RUNNING.md) for detailed Docker usage.

## Verification

After installation, verify the setup:

```bash
# Activate environment
conda activate zk0

# Run a simple test
python -c "import flwr, lerobot; print('Setup successful!')"
```

If no errors occur, the installation is complete. Proceed to [RUNNING.md](RUNNING.md) for execution instructions.