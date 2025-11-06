---
title: "zk0 Node Operators Guide: Join Decentralized Robotics AI Training"
description: "Complete guide for node operators to contribute to zk0 federated learning network with private SO-100 datasets, using zk0bot CLI for SmolVLA training in humanoid robotics."
---
# zk0 Node Operators Guide

Welcome to the zk0 Node Operators Guide! This document provides everything you need to know to participate in the zk0 federated learning network as a node operator.

## What is zk0?

[Installation Guide](INSTALLATION.md) | [Architecture Overview](ARCHITECTURE.md) | [Running Simulations](RUNNING.md)

zk0 is a federated learning platform for robotics AI, enabling privacy-preserving training of SmolVLA models across distributed clients using real-world SO-100/SO-101 datasets. Node operators contribute their private robotics datasets while maintaining full data privacy.

## Getting Started

### 1. Apply to Become a Node Operator

To join the zk0 network:

1. **Review Requirements**: Ensure you have:
   - A private robotics dataset (SO-100/SO-101 compatible)
   - GPU-enabled machine (recommended for training)
   - Stable internet connection
   - Basic familiarity with Docker

2. **Submit Application**: Create a new issue using our [Node Operator Application Template](https://github.com/ivelin/zk0/issues/new?template=node-operator-application.md)

3. **Wait for Approval**: Our team will review your application and contact you via Discord

### 2. Install zk0bot CLI

Once approved, install the zk0bot CLI tool:

```bash
# One-line installer
curl -fsSL https://get.zk0.bot | bash
```

This will:
- Download and install zk0bot to your PATH
- Verify Docker and Docker Compose installation
- Set up necessary dependencies

### 3. Configure Your Environment

Set up required environment variables:

```bash
# For Hugging Face datasets (if using HF-hosted private datasets)
export HF_TOKEN="your_huggingface_token"
```

Note: WandB logging is handled server-side only. Client training does not require WandB credentials.

### 4. Start Your Client

Launch your zk0 client with your private dataset:

```bash
# For Hugging Face datasets
zk0bot client start hf:yourusername/your-private-dataset

# For local datasets
zk0bot client start local:/path/to/your/dataset
```

Your client will:
- Connect to the zk0 server
- Participate in federated learning rounds
- Train on your private data locally
- Send only model updates (no raw data leaves your machine)

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

**Docker not found**: Install Docker Desktop or Docker Engine
**Permission denied**: Ensure Docker daemon is running and you have permissions
**Dataset not found**: Verify dataset path/URL and credentials
**Connection failed**: Check internet connection and server availability

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
- All communications use TLS encryption
- Data remains on your local machine
- Secure parameter validation and hashing
- No external access to your datasets

### Performance
- Training time: 10-30 minutes per round
- Network transfer: Minimal (only model updates)
- GPU utilization: Automatic detection and optimization

## Advanced Configuration

### Custom Docker Compose
You can modify the Docker Compose files for advanced setups:

```yaml
# docker-compose.client.yml
version: '3.8'
services:
  zk0-client:
    image: ghcr.io/ivelin/zk0:v0.4.0
    environment:
      - DATASET_URI=${DATASET_URI}
      - HF_TOKEN=${HF_TOKEN}
    volumes:
      - ./datasets:/app/datasets:ro
      - ./outputs:/app/outputs
```

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

*Last updated: 2025-10-31*