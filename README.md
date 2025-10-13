# zk0 [zee-ˈkō]

Open Source Humanoid AI trained collaboratively by a community of builders.

<a href="https://imagine-public.x.ai/imagine-public/share-videos/3332fbd2-7b73-4986-9ce5-6f4029569d89.mp4?cache=1"><img width="464" height="688" alt="image" src="docs/images/robots.png" /></a>

## 🚀 **Latest Model Release**

The zk0.bot SmolVLA Federated Learning model is now available on Hugging Face Hub!

- **Model**: [ivelin/zk0-smolvla-fl](https://huggingface.co/ivelin/zk0-smolvla-fl)
- **Training**: 30 rounds of federated learning with FedProx (μ=0.01)
- **Final Policy Loss**: 0.544
- **Clients**: 4 clients on diverse SO-100 robotics tasks
- **Framework**: Flower + SmolVLA + SO-100 datasets

```python
from transformers import AutoModel, AutoConfig
import torch

# Load the federated model
model = AutoModel.from_pretrained("ivelin/zk0-smolvla-fl")
config = AutoConfig.from_pretrained("ivelin/zk0-smolvla-fl")

# Ready for robotics manipulation tasks!
```

## Why

AI technology has [advanced enough to speculate](https://x.com/elonmusk/status/1786367513137233933) that within a decade most people will have their own humanoid buddy. By some estimates humanoids will become $100 Trillion market (5B humanoids * $20,000 per unit).

[Today's leading closed source humanoid](https://x.com/Tesla_Optimus/status/1846294753144361371) is trained on [100,000+ GPU farm](https://nvidianews.nvidia.com/news/spectrum-x-ethernet-networking-xai-colossus) with real world data collected from millions of cars labeled by able human drivers and a growing number of humanoid robot prototypes used in real world manufacturing environment. This is an enormous scale of compute and data that is hard to compete with as a centralized entity. However it would be interesting to see if a decentralized approach might produce useful results over time. On the chance that proprietary humanoids ever go rogue, it would be nice to have open source alternatives.

## Community Events

### Upcoming Events

- [Register now](https://lu.ma/embed/event/evt-udINVLo325xhKsG/simple) for the zk0 event at the upcoming DevConnect conference in Buenos Aires, Argentina on November 18, 2025.

### Past Events

- [Watch a recorded presentation](https://www.youtube.com/embed/fwAtTOZttWo?si=3d50oQtSvMvGxNg6) of the project at the Flower Monthly Webcast.

### Join the Community

Join our Discord server to connect with other contributors, ask questions, and stay updated on the latest developments:

[Join zk0 Discord](https://discord.gg/uue3uKSA)

For more detailed contribution guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md) and [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).

## How

zk0 is composed of several major building blocks:

- Physical Embodiment:
  * Open Source 3D printed robot parts
  * Base 3D model so100 series from [HuggingFace LeRobot](https://huggingface.co/lerobot)
- Generative AI:
  * End-to-end Vision Language Action models.
  * Base SmolVLA model from [HuggingFace LeRobot](https://huggingface.co/lerobot)
- Federated Learning:
  * Distributed network of nodes contributing local data and training compute to a shared model.
  * FL framework: [Flower](https://flower.ai/)

## Roadmap

- Zero Knowledge Proofs that allow quick verification and data privacy:
  * Quickly verifiable proofs that an FL node is making meaningful contributions.
  * Frameworks under consideration:
    * [SP1](https://github.com/succinctlabs/sp1)
    * [EZKL](https://github.com/zkonduit/ezkl)
- Onchain contributor coordination
  * Immutable contribution history
  * Programmable network participation rules, incentives and project governance
  * Hosting blockchain: TBD

## Share

![image](https://github.com/user-attachments/assets/e03913ec-62a0-4b05-a286-6fc18dfd433f)

## Quick Start

For detailed setup, see [docs/INSTALLATION.md](docs/INSTALLATION.md).

### Prerequisites

- Python 3.10+, Conda, Git.
- NVIDIA GPU recommended.

### Clone and Setup

```shell
git clone <repository-url> .
cd zk0

# Create conda env
conda create -n zk0 python=3.10 -y
conda activate zk0
conda install ffmpeg=7.1.1 -c conda-forge

# Install deps
pip install -r requirements.txt
pip install -e .

# Env vars
cp .env.example .env  # Edit as needed (e.g., HF_TOKEN)
```

### Run the Simulation

See [docs/RUNNING.md](docs/RUNNING.md) for full instructions.

```bash
# Quick test (1 round, serialized GPU)
./train.sh

# Full run (5 rounds)
conda run -n zk0 flwr run . local-simulation-serialized-gpu --run-config "num-server-rounds=5"

# Docker alternative
./train.sh --docker
```

- **Defaults**: 100 rounds, 10 clients, SO-100 datasets.
- **Outputs**: `outputs/<timestamp>/` with logs, metrics, charts (`eval_policy_loss_chart.png`), checkpoints (`.safetensors`), videos.

**Tested**: Completes 100 rounds in ~50s; policy loss tracks convergence.

## Repository Branches

- **main**: Stable releases. Use this for production setups and quick starts.
- **staging**: Final polish before merging with main. No new features. Only bug fixes and docs polish.
- **dev**: Active feature development. Pull requests should target dev. Clone or switch with `git checkout dev` for latest features (may be unstable).

## Project Status

### 🚀 Current Stage: Beta

Advanced development with core FL for SmolVLA on SO-100. Recent updates: Policy loss standardization, multi-repo partitioning, server eval.

#### Completed Milestones

- ✅ Core Infrastructure: Flower 1.21.0 + Ray 2.31.0 + LeRobot 0.3.3.
- ✅ Client Implementation: SmolVLA training, dataset partitioning.
- ✅ Testing: 80%+ coverage, unit/integration suites.
- ✅ CI/CD: GitHub Actions, auto-testing.
- ✅ Config/Tooling: YAML datasets, env management.

#### In Progress

- Multi-task learning, advanced strategies (FedProx+), hyperparam tuning.
- ZK proofs, onchain coordination.

Full status: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#project-status). Baselines: [docs/TECHNICAL-OVERVIEW.md](docs/TECHNICAL-OVERVIEW.md#federated-vs-centralized-training-comparison).

**Config**: 4 clients (pickplace, stacking, etc.); 30+ rounds; policy loss metric.

## Documentation

- [Installation](docs/INSTALLATION.md): Full setup.
- [Running](docs/RUNNING.md): Execution, outputs, troubleshooting.
- [Architecture](docs/ARCHITECTURE.md): FL design, components.
- [Development](docs/DEVELOPMENT.md): Testing, logging, guidelines.
- [Technical Overview](docs/TECHNICAL-OVERVIEW.md): Comparisons, reproducibility, videos.

Internal: `.kilocode/rules/memory-bank/` (e.g., [brief.md](.kilocode/rules/memory-bank/brief.md)).

## Social Media

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">It's time for a complete open-source stack for autonomy/robotics plus distributed learning. The first step is here: <a href="https://twitter.com/LeRobotHF?ref_src=twsrc%5Etfw">@LeRobotHF</a> + <a href="https://twitter.com/flwrlabs?ref_src=twsrc%5Etfw">@flwrlabs</a> LFG 🚀<a href="https://twitter.com/comma_ai?ref_src=twsrc%5Etfw">@comma_ai</a> <a href="https://twitter.com/wayve_ai?ref_src=twsrc%5Etfw">@wayve_ai</a> <a href="https://twitter.com/Figure_robot?ref_src=twsrc%5Etfw">@Figure_robot</a> <a href="https://twitter.com/Tesla?ref_src=twsrc%5Etfw">@Tesla</a> <a href="https://t.co/8O8cSD3SbO">https://t.co/8O8cSD3SbO</a> <a href="https://t.co/oVUOLTvwzm">https://t.co/oVUOLTvwzm</a></p>&mdash; nic lane (@niclane7) <a href="https://twitter.com/niclane7/status/1879597539676266726?ref_src=twsrc%5Etfw">January 15, 2025</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Open-source robots just got a boost. Frameworks like Flower FL enable faster learning, efficient scaling, and continuous knowledge sharing using real-world data. <a href="https://t.co/j8VSGiWF0W">https://t.co/j8VSGiWF0W</a></p>&mdash; 𝚐𝔪𝟾𝚡𝚡𝟾 (@gm8xx8) <a href="https://twitter.com/gm8xx8/status/1879633368427761785?ref_src=twsrc%5Etfw">January 15, 2025</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">We are not so far from a future where robots will be constantly learning by interacting with humans and their environments.<br><br>Frameworks like <a href="https://twitter.com/flwrlabs?ref_src=twsrc%5Etfw">@flwrlabs</a> will enable these robots to learn much faster by continuously sharing their learnings.<br><br>We really live in a sci-fi movie 😅 <a href="https://t.co/kAz3xZ2qvB">https://t.co/kAz3xZ2qvB</a></p>&mdash; Remi Cadene (@RemiCadene) <a href="https://twitter.com/RemiCadene/status/1879592068865282227?ref_src=twsrc%5Etfw">January 15, 2025</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Federated Learning Meets Robotics: 🤖 LeRobot + 🌼 Flower<br><br>This demo demonstrates how robots in remote environments can collaboratively train an AI model using their local data, which is then aggregated into a shared model. <br><br>In this quickstart, you will train a Diffusion policy… <a href="https://t.co/i32MkbxoPW">pic.twitter.com/i32MkbxoPW</a></p>&mdash; Flower (@flwrlabs) <a href="https://twitter.com/flwrlabs/status/1879571258532036739?ref_src=twsrc%5Etfw">January 15, 2025</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

## Contributing

We welcome contributions from the community! At this Beta stage, we're particularly interested in:

### Node Operators

If you have access to a LeRobot SO100 arm (or the newer SO101 version) and a local machine with an RTX 3090 GPU or better compatible with the LeRobot library, we'd love for you to join as a node operator. Your unique training data and compute resources will help improve the federated learning system.

### Code Contributors

We're also looking for developers to help with:
- Bug fixes and improvements
- Documentation enhancements
- New feature development
- Testing and quality assurance

### Getting Started

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request


---

**License**: [LICENSE](LICENSE)  
**Repository**: [GitHub](https://github.com/ivelin/zk0)  
**Memory Bank**: Internal docs in `.kilocode/rules/memory-bank/` for contributors.
