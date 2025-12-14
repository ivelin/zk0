# zk0: Federated Learning for Robotics AI

Open-source federated learning platform for SmolVLA models on SO-100 datasets using Flower framework.

## Latest Model Release

- **Model**: [ivelin/zk0-smolvla-fl](https://huggingface.co/ivelin/zk0-smolvla-fl)
- **Training**: 250 rounds FedProx (μ=0.01), dynamic LR/MU scheduling
- **Final Policy Loss**: 0.495
- **Clients**: 4 clients on diverse SO-100 tasks
- **Framework**: Flower + SmolVLA + SO-100 datasets

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("ivelin/zk0-smolvla-fl")
```

## Quick Start

For detailed setup, see [docs/INSTALLATION](docs/INSTALLATION).

### Prerequisites

- Python 3.10+, Conda, Git.
- NVIDIA GPU recommended.

### Clone and Setup

```shell
git clone https://github.com/ivelin/zk0.git
cd zk0

# Create conda env
conda create -n zk0 python=3.10 -y
conda activate zk0
conda install ffmpeg=7.1.1 -c conda-forge

# Install deps
pip install -e .

# Env vars
cp .env.example .env  # Edit as needed (e.g., HF_TOKEN)
```

### Run the Simulation

See [docs/INSTALLATION](docs/INSTALLATION) for full instructions.

```bash
# Tiny test run (1 round)
./train-fl-simulation.sh --tiny

```

## Production Deployment

For production use with multiple distributed nodes, use the zk0bot CLI tool for orchestrated federated learning:

```bash
# Server operator: Start the central coordinator
zk0bot server start

# Client operators (each with their private dataset)
zk0bot client start hf:yourusername/your-private-dataset
# or
zk0bot client start local:/path/to/your/dataset

# Monitor status
zk0bot status

# Stop services
zk0bot server stop
zk0bot client stop
```

The CLI uses Flower's Deployment Engine (SuperLink, SuperNodes, SuperExec) for stateless, insecure-mode operation. Server runs continuously, automatically managing FL sessions based on connected clients. See [docs/NODE-OPERATORS.md](docs/NODE-OPERATORS.md) for detailed setup and security notes.

For run details, outputs, experiment tracking, and model pushing, see [docs/RUNNING](docs/RUNNING). For repository branches and contributing guidelines, see [CONTRIBUTING](CONTRIBUTING).

## Project Status

### Current Stage: Beta

The project is ready for local simulation testing with multiple clients and datasets.

#### In Progress
- Preparing client and server modules for production deployment
- ZK proofs, onchain coordination.

**Config**: 12 clients available (4 active); 500 rounds; policy loss metric; FedProx (μ=0.01); server-side evaluation.

For full details, see [docs/ARCHITECTURE](docs/ARCHITECTURE#project-status).


## Contributing

We welcome contributions from the community! At this Beta stage, we're particularly interested in:

### Node Operators

#### Requirements

- **Hardware**: LeRobot SO100 or SO101 robotic arm. Contributors can either:
  - Build a DIY arm using the official [LeRobot SO101 repository](https://huggingface.co/docs/lerobot/so101)
  - Or order a pre-built kit, for example [this one](https://www.ebay.com/str/ovobot) from Florin who runs the [Austin Robotics Meetup](https://austinrobotics.io/).
- **Compute**: Local machine with RTX 3090 GPU or better, compatible with LeRobot library
- **Network**: Stable internet connection for federated communication
- **Data**: Unique training data from your robotics setup

If you meet these requirements, we'd love for you to join as a node operator. Your unique training data and compute resources will help improve the federated learning system. For detailed setup instructions, see [CONTRIBUTING](CONTRIBUTING).

### Other Ways to Contribute

There are several ways you can contribute to this project:

1. **Node Operators**: Join the federated network with your hardware and data
2. **Code Contributors**: Improve the codebase, add features, fix bugs
3. **Documentation**: Help improve documentation and tutorials
4. **Testing**: Report bugs, test new features, improve test coverage
5. **Feedback**: Share your experience and suggestions

For more details on each, see [CONTRIBUTING](CONTRIBUTING).


# Test change for xAI delta sync
# Test change for xAI delta sync
# Test change for xAI delta sync
# Test change for xAI delta sync
# Test change for xAI delta sync
