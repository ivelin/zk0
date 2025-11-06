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

For detailed setup, see [docs/INSTALLATION.md](docs/INSTALLATION.md).

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

See [docs/INSTALLATION.md](docs/INSTALLATION.md) for full instructions.

```bash
# Quick test (1 round, serialized GPU)
./train-fl-simulation.sh

# Full run (5 rounds)
conda run -n zk0 flwr run . local-simulation-serialized-gpu --run-config "num-server-rounds=5"

# Docker alternative
./train-fl-simulation.sh --docker
```

### Push Model to Hugging Face Hub

**Prerequisites:**
- Ensure your Hugging Face token is set in `.env`: `HF_TOKEN=your_token_here`
- The conda environment "zk0" must be active for script execution

After training, your model checkpoint will be automatically pushed to Hugging Face Hub as a complete checkpoint directory.
However if the training stops early for any reason, you can still push a saved intermediate checkpoint directory to HF Hub:

```bash
# Push model checkpoint directory to HF Hub
conda run -n zk0 python -m zk0.push_to_hf outputs/2025-10-09_13-59-05/models/checkpoint_round_30

# Push to custom repository
conda run -n zk0 python -m zk0.push_to_hf outputs/2025-10-09_13-59-05/models/checkpoint_round_30 --repo-id your-username/your-model
```


- **Defaults**: 500 rounds, 4 clients, SO-100/SO-101 datasets.
- **Outputs**: `outputs/<timestamp>/` with logs, metrics, charts (`eval_policy_loss_chart.png`), checkpoint directories, videos.
- **HF Hub Push**: For tiny/debug runs (e.g., `num-server-rounds < checkpoint_interval=20`), the final model push to Hugging Face Hub is skipped to avoid repository clutter with incomplete checkpoints. Local checkpoints are always saved. Full runs (≥20 rounds) will push to the configured `hf_repo_id`.

### Experiment Tracking

zk0 integrates with Weights & Biases (WandB) for comprehensive experiment tracking and visualization:

- **Automatic Logging**: When `use-wandb=true` in `pyproject.toml`, training metrics, hyperparameters, and evaluation results are automatically logged to WandB.
- **Model Cards**: Generated README.md files in checkpoint directories include direct links to WandB experiment runs when WandB is enabled.
- **Visualization**: View detailed training curves, client performance, and federated learning metrics in real-time.
- **Setup**: Set `WANDB_API_KEY` in your `.env` file to enable WandB logging.

**Tested**: Completes 500 rounds in ~10-15 minutes; policy loss tracks convergence with early stopping.

## Repository Branches

- **main**: Stable releases. Use this for production setups and quick starts.
- **staging**: Final polish before merging with main. No new features. Only bug fixes and docs polish.
- **dev**: Active feature development. Pull requests should target dev. Clone or switch with `git checkout dev` for latest features (may be unstable).

## Project Status

### 🚀 Current Stage: Beta

Advanced development with core FL for SmolVLA on SO-100/SO-101. v0.4.15 updates: Modular architecture refinement with dedicated server utilities (parameter_validation.py, visualization.py, strategy.py, model_checkpointing.py, evaluation.py, server_utils.py, model_utils.py, fit_configuration.py). Added common utilities (parameter_utils.py, utils.py) and client core module. Fixed test inconsistencies and achieved 146 tests passing with 36.73% coverage. Enhanced security with bidirectional SHA256 parameter validation between client and server. Consolidated metrics implementation for unified reporting. Dynamic LR/MU scheduling with warm restarts, adaptive boosts, and spike detection. Prepare for commit workflow established for consistent code quality assurance.

#### Completed Milestones

- ✅ Core Infrastructure: Flower 1.20.0 + Ray 2.31.0 + LeRobot 0.3.0.
- ✅ Client Implementation: SmolVLA training, dataset partitioning.
- ✅ Testing: 30%+ coverage, unit/integration suites.
- ✅ CI/CD: GitHub Actions, auto-testing.
- ✅ Config/Tooling: YAML datasets, env management.
- ✅ Enhanced Security: Bidirectional SHA256 parameter validation.
- ✅ Consolidated Metrics: Server-side evaluation files now include both aggregated and individual client metrics with dataset identification (v0.1.19).

#### In Progress

- Preparing client and server modules for production deployment
- ZK proofs, onchain coordination.

Full status: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#project-status). Baselines: [docs/TECHNICAL-OVERVIEW.md](docs/TECHNICAL-OVERVIEW.md#federated-vs-centralized-training-comparison).

**Config**: 12 clients available (4 active); 500 rounds; policy loss metric; FedProx (μ=0.01); server-side evaluation.

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

If you meet these requirements, we'd love for you to join as a node operator. Your unique training data and compute resources will help improve the federated learning system. For detailed setup instructions, see [CONTRIBUTING.md](CONTRIBUTING.md).

### Other Ways to Contribute

There are several ways you can contribute to this project:

1. **Node Operators**: Join the federated network with your hardware and data
2. **Code Contributors**: Improve the codebase, add features, fix bugs
3. **Documentation**: Help improve documentation and tutorials
4. **Testing**: Report bugs, test new features, improve test coverage
5. **Feedback**: Share your experience and suggestions

For more details on each, see [CONTRIBUTING.md](CONTRIBUTING.md).


