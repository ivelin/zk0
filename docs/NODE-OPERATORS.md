---
title: "zk0 Node Operators Guide: Join Decentralized Robotics AI Training"
description: "Complete guide for node operators to contribute to zk0 federated learning network with private SO-100 datasets, using zk0bot CLI for SmolVLA training in humanoid robotics."
---
# zk0 Node Operators Guide

Welcome to the zk0 Node Operators Guide! This document provides everything you need to know to participate in the zk0 federated learning network as a node operator.

## What is zk0?

[Installation Guide](INSTALLATION) | [Architecture Overview](ARCHITECTURE) | [Running Simulations](RUNNING)

zk0 is a federated learning platform for robotics AI, enabling privacy-preserving training of SmolVLA models across distributed clients using real-world SO-100/SO-101 datasets. Node operators contribute their private robotics datasets while maintaining full data privacy.

## Getting Started

### 1. Apply to Become a Node Operator

To join the zk0 network:

1. **Review Requirements**: Ensure you have:
   - A private robotics dataset (SO-100/SO-101 compatible)
   - GPU-enabled machine (recommended for training)
   - Stable internet connection
   - Basic familiarity with Conda and tmux

2. **Submit Application**: Create a new issue using our [Node Operator Application Template](https://github.com/ivelin/zk0/issues/new?template=node-operator-application.md)

3. **Wait for Approval**: Our team will review your application and contact you via Discord

### 2. Install zk0bot CLI

Once approved, install the zk0bot CLI tool:

```bash
# One-line installer
curl -fsSL https://raw.githubusercontent.com/ivelin/zk0/dev/get-zk0bot.sh | bash
```

This will:
- Clone/update zk0 repo to ~/zk0 (dev branch)
- Create conda zk0 env (Python 3.10)
- Install flwr[superexec], lerobot[smolvla], torch cu121, pip install -e .
- Optional Hugging Face login

### 3. Configure Your Environment

Set up required environment variables in `.env` (auto-sourced by zk0bot.sh):

```bash
# .env example (create in ~/zk0)
HF_TOKEN=your_huggingface_token_here
WANDB_API_KEY=your_wandb_key_here  # optional, server-side only
```

**Note**: zk0bot.sh automatically sources `.env` after conda activation, propagating HF_TOKEN/WANDB_API_KEY to tmux Flower subprocesses (SuperLink/SuperNode). No manual export needed.

### Full Production Session Example (Local Network)

**Server Machine:**
```bash
curl -fsSL https://raw.githubusercontent.com/ivelin/zk0/main/website/get-zk0bot.sh | bash
cd ~/zk0
zk0bot server start  # Auto-activates zk0 env; SuperLink ready
```

**Client Machines (same LAN, add to ~/.bashrc: export ZK0_SERVER_IP=server_ip):**
```bash
curl -fsSL https://raw.githubusercontent.com/ivelin/zk0/main/website/get-zk0bot.sh | bash
cd ~/zk0
zk0bot client start shaunkirby/record-test  # Auto-activates zk0 env; or your private dataset
zk0bot client start ethanCSL/direction_test
```

**On Server (submit run):**
```bash
zk0bot run --rounds 20 --stream  # Full FL session, stateless; auto-zk0 env
```

**Remote Clients:** Set `ZK0_SERVER_IP=public_server_ip` (insecure=true for dev; TLS for prod).

Note: WandB logging is handled server-side only. Client training does not require WandB credentials.

### 4. Start Your Client

**Light Test Production Run (Recommended):**
1. Server: zk0bot server start
2. Clients: zk0bot client start shaunkirby/record-test
3. Submit run: zk0bot run --rounds 3 --stream

Examples:
  zk0bot client start shaunkirby/record-test
  zk0bot client start ethanCSL/direction_test

**Production Run (Stateless):**
```bash
# Standard (pyproject.toml defaults) - runs all server rounds
zk0bot client start yourusername/your-private-dataset
zk0bot client start local:/path/to/your/dataset
```

Your client will:
- Connect to zk0 server (auto-starts at min_fit_clients=2)
- Train locally for all server rounds (stateless, no persistence)
- Send only model updates (no raw data leaves machine)

## Server Operations (For Server Operators)

If you're running a zk0 server:

```bash
# Start the server
zk0bot server start

# Check status
zk0bot status

# View logs
zk0bot server log

# Stop the server
zk0bot server stop
```

## Monitoring and Troubleshooting

### Check Status
```bash
zk0bot status
```

### View Logs
```bash
# Server logs
zk0bot server log

# Client logs
zk0bot client log
```

### Common Issues

**tmux not found**: sudo apt install tmux (Linux) or brew install tmux (macOS)
**Conda zk0 not active**: conda activate zk0
**Dataset not found**: Verify dataset path/URL and credentials
**Connection failed**: Check internet connection and server availability
**Installer fails**: Check GitHub status, ensure curl available, or git clone https://github.com/ivelin/zk0.git ~/zk0; cd ~/zk0; ./get-zk0bot.sh

## Dataset Requirements

### Supported Formats
- SO-100: Real-world robotics manipulation tasks
- SO-101: Extended robotics tasks
- Custom: LeRobot-compatible datasets

### Quality Guidelines
- Clear, well-annotated episodes
- Consistent task definitions
- No overlap with existing network datasets
- Minimum 100 episodes recommended

### Privacy Considerations
- All training happens locally
- Only model gradients are shared
- Raw data never leaves your environment
- Dataset metadata is anonymized

## Dynamic Client Joining (Stateless)

### Server Behavior
- **Always-On Operation**: Server runs continuously via SuperExec-Server.
- **Automatic Start**: Sessions start at min_fit_clients=2.
- **Idle Handling**: Idles below min_clients.

### Client Lifecycle (Stateless)
- **Full Participation**: Clients run ALL server rounds (num_server_rounds).
- **Manual Stop**: Use zk0bot client stop to disconnect.
- **Clean Restarts**: No state; always fresh.

### Flower Deployment Engine
- **Stateless SuperExec**: Clean restarts, no persistence.
- **Insecure Mode**: Dev; TLS for prod.

## Federation Flow

<div class="mermaid">
sequenceDiagram
    participant Admin as Admin/Submission\n(zk0bot.sh run)
    participant SuperLink as SuperLink\n(flower-superlink)
    participant SuperNode1 as SuperNode 1\n(flower-supernode,\ndataset-uri=uri1)
    participant SuperNode2 as SuperNode 2\n(flower-supernode,\ndataset-uri=uri2)
    participant ServerApp as ServerApp\n(SuperExec process\non SuperLink host)
    participant ClientApp1 as ClientApp 1\n(SuperExec process\non SuperNode 1)
    participant ClientApp2 as ClientApp 2\n(SuperExec process\non SuperNode 2)

    Note over SuperLink,SuperNode2: Persistent Infrastructure (started first)

    Admin->>+SuperLink: Start SuperLink\n(zk0bot.sh server start)
    Note right of SuperLink: Listens on gRPC Fleet API\n(ports 9091-9093)

    Admin->>+SuperNode1: Start SuperNode 1\n(zk0bot.sh client start &lt;dataset-uri1&gt;)
    Note right of SuperNode1: e.g., dataset-uri1 = "shaunkirby/record-test"\nor "local:/data/client1_episodes"
    SuperNode1->>+SuperLink: Register via gRPC\n(Fleet API handshake)
    Note right of SuperNode1: Passes node-config\n(dataset-uri=uri1 → unique/private dataset)

    Admin->>+SuperNode2: Start SuperNode 2\n(zk0bot.sh client start &lt;dataset-uri2&gt;)
    Note right of SuperNode2: e.g., dataset-uri2 = "ethanCSL/direction_test"\nor private HF repo / local path
    SuperNode2->>+SuperLink: Register via gRPC\n(Fleet API handshake)
    Note right of SuperNode2: Passes node-config\n(dataset-uri=uri2 → unique/private dataset)

    Note over SuperLink,SuperNode2: SuperNodes now visible/registered in SuperLink logs

    Admin->>+SuperLink: Submit Run\n(zk0bot.sh run → flwr run)
    Note right of Admin: Uploads Flower App Bundle (FAB)\ncontaining ServerApp + ClientApp code

    SuperLink->>ServerApp: Spawn SuperExec process\nfor ServerApp execution
    Note right of ServerApp: ServerApp starts (strategy, rounds, etc.)

    SuperLink->>SuperNode1: Instruct to execute ClientApp\n(via registered Fleet API, sends FAB + config)
    SuperNode1->>ClientApp1: Spawn SuperExec process\nfor ClientApp
    Note over ClientApp1: ClientApp loads private/unique dataset\n(from injected node-config dataset-uri=uri1)\ne.g., HF dataset download or local path

    SuperLink->>SuperNode2: Instruct to execute ClientApp\n(via registered Fleet API, sends FAB + config)
    SuperNode2->>ClientApp2: Spawn SuperExec process\nfor ClientApp
    Note over ClientApp2: ClientApp loads private/unique dataset\n(from injected node-config dataset-uri=uri2)\ne.g., different HF repo or local episodes

    Note over ServerApp,ClientApp2: Federation begins (gRPC message passing via SuperLink/SuperNodes)

    loop For each federation round (e.g., Fit)
        ServerApp->>SuperLink: Send FitIns (parameters)\nto selected SuperNodes
        SuperLink->>SuperNode1: Forward FitIns (gRPC)
        SuperNode1->>ClientApp1: Forward to local SuperExec
        ClientApp1->>ClientApp1: Local training\non private unique dataset (from uri1)
        ClientApp1->>SuperNode1: Return FitRes (updated parameters)
        SuperNode1->>SuperLink: Forward FitRes
        SuperLink->>ServerApp: Deliver FitRes

        SuperLink->>SuperNode2: Forward FitIns (gRPC)
        SuperNode2->>ClientApp2: Forward to local SuperExec
        ClientApp2->>ClientApp2: Local training\non private unique dataset (from uri2)
        ClientApp2->>SuperNode2: Return FitRes
        SuperNode2->>SuperLink: Forward FitRes
        SuperLink->>ServerApp: Deliver FitRes

        ServerApp->>ServerApp: Aggregate updates\n(e.g., FedAvg)
    end

    Note over ServerApp,ClientApp2: Similar flow for Evaluate rounds\n(Server sends EvaluateIns, clients evaluate locally on their private datasets from uri1/uri2)

    Note over SuperLink, SuperNode2: Run completes → SuperExec processes terminate. SuperLink & SuperNodes remain running for next Run
</div>


### Updated Explanation of the Sequence (Dataset URI Flow)

The core architecture uses Flower's Deployment Engine, with client data privacy via **positional `<dataset-uri>`** in `zk0bot.sh client start <dataset-uri>`.

#### CLI Change
- `zk0bot.sh client start <dataset-uri>`
  - Examples:
    - `./zk0bot.sh client start shaunkirby/record-test`
    - `./zk0bot.sh client start ethanCSL/direction_test`
    - Private: `./zk0bot.sh client start yourusername/private-so100`
    - Local: `./zk0bot.sh client start local:/home/user/robot_episodes`
- Passes `--node-config '{"dataset-uri": "<uri>"}'` to `flower-supernode`.

#### Persistent Infrastructure First
- Start SuperLink (`zk0bot server start`).
- SuperNodes register with SuperLink, injecting unique `dataset-uri` in node-config.

#### Dynamic App Execution on Run Submission
- `zk0bot run` submits FAB to SuperLink.
- SuperLink spawns ServerApp; instructs SuperNodes to spawn ClientApps with pre-registered config.
- Each ClientApp loads **exclusive dataset** from its `node_config["dataset-uri"]` (HF download/local).

#### Message Passing During Federation
- gRPC via SuperLink/SuperNodes: parameters/metrics only.
- Local training/eval on private datasets.
- No raw data leaves client.

## Community and Support

### Discord
Join our Discord community for support and updates: [zk0 Discord](https://discord.gg/zk0)

### GitHub
- Report issues: [GitHub Issues](https://github.com/ivelin/zk0/issues)
- Documentation: [zk0 Docs](https://github.com/ivelin/zk0/tree/main/docs)

### Contact
- Email: operators@zk0.ai
- Discord: @zk0-team

## Technical Details

### System Requirements
- **OS**: Linux, macOS, Windows (WSL2)
- **RAM**: 16GB minimum, 32GB recommended
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)
- **Storage**: 50GB free space for datasets and models
- **Network**: Stable broadband connection

### Security
- Communications use insecure mode for development (no TLS encryption)
- Data remains on your local machine
- Secure parameter validation and hashing
- No external access to your datasets
- **Note**: TLS can be enabled for production deployments

### Performance
- Training time: 10-30 minutes per round
- Network transfer: Minimal (only model updates)
- GPU utilization: Automatic detection and optimization

## Advanced Configuration

### Advanced tmux/Conda Configuration
zk0bot uses native Flower CLI + tmux for persistence. For custom setups:
- Edit [`zk0bot.sh`](zk0bot.sh) ports/node-config.
- Env vars: `DATASET_URI`, `HF_TOKEN`.

### Environment Variables
- `DATASET_URI`: Dataset location (hf:repo/name or local:/path)
- `HF_TOKEN`: Hugging Face API token
- `WANDB_API_KEY`: Weights & Biases API key
- `ZK0_SERVER_URL`: Custom server URL (default: auto-discovery)

## Contributing

We welcome contributions to improve the zk0 platform:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request
4. Join our Discord for discussion

## License

zk0 is open-source software licensed under the Apache 2.0 License.

---

*Last updated: 2025-12-17*