---
title: "zk0 FAQ: Frequently Asked Questions about Federated Learning for Robotics AI"
description: "Common questions about zk0, the open-source federated learning platform for SmolVLA models on SO-100 datasets, including setup, hardware, privacy, and contributions."
---

# zk0 FAQ

## What is zk0?
zk0 is an open-source federated learning platform for training advanced vision-language-action (VLA) models like SmolVLA on real-world robotics datasets (SO-100/SO-101). It enables privacy-preserving collaborative training across distributed clients using the Flower framework, with future integrations for ZK proofs and blockchain incentives.

## How does federated learning work in zk0?
In zk0, clients (e.g., your local machine with a robotic arm) train models on private datasets locally. Only model updates (parameters) are sent to a central server for aggregation using strategies like FedProx. No raw data leaves your device, ensuring privacy while building a shared global model for robotics tasks like pick-place or stacking.

## What hardware is required to run zk0?
- **Development/Simulation**: A standard machine with Python 3.10+, NVIDIA GPU (RTX 30-series or better recommended for training), and 16GB+ RAM.
- **Node Operator**: LeRobot SO-100/SO-101 compatible robotic arm (DIY or pre-built), RTX 3090+ GPU, stable internet. See [Node Operators Guide](NODE-OPERATORS.md) for details.
- **Server**: Multi-GPU setup for aggregation; Docker for production.

## What datasets does zk0 use?
zk0 focuses on SO-100 real-world robotics manipulation tasks (e.g., pick-place, stacking, tool use). Clients use unique, non-overlapping subsets to promote diverse skill learning. Evaluation uses unseen SO-101 datasets for generalization testing. Custom LeRobot-compatible datasets are supported.

## How do I install and run zk0?
Follow the [Installation Guide](INSTALLATION.md):
1. `conda create -n zk0 python=3.10 && conda activate zk0`
2. `pip install -e .`
3. Run: `./train-fl-simulation.sh` for a quick test.
For production: Use `zk0bot` CLI after approval as a node operator.

## Is my data private in zk0?
Yes! zk0 uses federated learning: Raw videos, images, or states never leave your machine. Only anonymized model parameters are exchanged, secured with TLS and SHA256 validation. Dataset metadata is UUID-anonymized for node operators.

## How can I contribute as a node operator?
1. Apply via [GitHub Issue Template](https://github.com/ivelin/zk0/issues/new?template=node-operator-application.md).
2. Once approved, install `zk0bot` CLI: `curl -fsSL https://get.zk0.bot | bash`.
3. Start client: `zk0bot client start hf:your-dataset`.
Your unique robotics data helps improve the global model. Join [Discord](https://discord.gg/dhMnEne7RP) for support.

## What is the difference between simulation and production mode?
- **Simulation**: Local testing with fixed clients (e.g., `flwr run . local-simulation-gpu`). Ideal for development.
- **Production**: Multi-node via Docker Compose and `zk0bot` CLI. Supports dynamic clients, private datasets, and scaling. See [Node Operators](NODE-OPERATORS.md).

## How do I monitor training progress?
- **Logs**: Unified in `outputs/<timestamp>/simulation.log`; per-client in `clients/client_N/client.log`.
- **Metrics**: `federated_metrics.json` and auto-generated charts like `eval_policy_loss_chart.png`.
- **WandB**: Enable in `pyproject.toml` for real-time dashboards (server-side only).
- **Videos**: Rollout videos in `outputs/evaluate/` for qualitative assessment.

## Where can I find the trained models?
Models are saved as HF-compatible checkpoints in `outputs/<timestamp>/models/checkpoint_round_N/`. Full runs push to [Hugging Face Hub](https://huggingface.co/ivelin/zk0-smolvla-fl). Load with: `AutoModel.from_pretrained("ivelin/zk0-smolvla-fl")`.

## How do I report bugs or request features?
- Use [GitHub Issues](https://github.com/ivelin/zk0/issues) for bugs (include steps to reproduce, logs).
- Feature requests: Describe use case and impact.
- Join [Discord](https://discord.gg/dhMnEne7RP) for discussions.

For more, see [Contributing Guide](CONTRIBUTING.html) or contact us on Discord.