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

## Join the Network (Self-Service)

Contributors join the zk0 federated network via the hosted coordinator and `zk0bot` CLI — no GitHub application required.

```bash
# Install zk0bot (one line)
curl -fsSL https://raw.githubusercontent.com/ivelin/zk0/main/website/get-zk0bot.sh | bash
cd ~/zk0

# Remote client: point at the hosted coordinator
export ZK0_SERVER_IP=coordinator.zk0.bot   # fleet API host

# Register your dataset-uri and start a SuperNode
zk0bot client start yourusername/your-private-dataset
# or local episodes:
zk0bot client start local:/path/to/your/dataset

zk0bot status
```

Coordinator operators start the always-on SuperLink:

```bash
zk0bot server start
ZK0_COORDINATOR_ADDRESS=coordinator.zk0.bot:9093 zk0bot run --rounds 20 --stream
```

Each run writes `contributor_registry.jsonl` under the run output directory with `{node_id, dataset_uri, timestamp}` for attribution (opt-out anonymity supported at join time).

The CLI uses Flower's Deployment Engine (SuperLink, SuperNodes, SuperExec). See [docs/NODE-OPERATORS.md](docs/NODE-OPERATORS.md) for the full self-service path, remote coordinator config, and security notes.

For run details, outputs, experiment tracking, and model pushing, see [docs/RUNNING](docs/RUNNING). For repository branches and contributing guidelines, see [CONTRIBUTING](CONTRIBUTING).

## Project Status

### Current Stage: Beta (Self-Service Network)

Local FL simulation and production deployment via `zk0bot` are supported. Contributors register a `dataset-uri` via CLI and connect to a hosted coordinator using `ZK0_SERVER_IP`.

#### In Progress
- zk0.bot MCP-guided onboarding and x402 network gate
- ZK proofs, onchain coordination (roadmap)

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

If you meet these requirements, join via the self-service path in [docs/NODE-OPERATORS.md](docs/NODE-OPERATORS.md) — install `zk0bot`, register your `dataset-uri`, and connect to the hosted coordinator. No application approval required.

### Other Ways to Contribute

There are several ways you can contribute to this project:

1. **Node Operators**: Join the federated network with your hardware and data
2. **Code Contributors**: Improve the codebase, add features, fix bugs
3. **Documentation**: Help improve documentation and tutorials
4. **Testing**: Report bugs, test new features, improve test coverage
5. **Feedback**: Share your experience and suggestions

For more details on each, see [CONTRIBUTING](CONTRIBUTING).
