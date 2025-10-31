# Running the Project

This guide covers executing the zk0 federated learning simulation, including default and alternative methods, output details, and troubleshooting.

## Default: Conda Environment Execution

By default, the training script uses the conda `zk0` environment for **fast and flexible execution**. This provides direct access to host resources while maintaining reproducibility.

### Quick Start with Conda

```bash
# Activate environment (if not already)
conda activate zk0

# Run federated learning (uses pyproject.toml defaults: 1 round, 2 steps/epochs, serialized GPU)
./train.sh
ls
# Or direct Flower run with overrides
conda run -n zk0 flwr run . local-simulation-serialized-gpu --run-config "num-server-rounds=5 local-epochs=10"

# Activate first, then run
conda activate zk0
flwr run . local-simulation-serialized-gpu --run-config "num-server-rounds=5 local-epochs=10"
```

**✅ Validated Alternative**: Conda execution has been tested and works reliably for federated learning runs, providing a simpler setup for development environments compared to Docker.

## Alternative: Docker-Based Execution

For **reproducible and isolated execution**, use the `--docker` flag or run directly with Docker. This ensures consistent environments and eliminates SafeTensors multiprocessing issues.

### Training Script Usage

The `train.sh` script runs with configuration from `pyproject.toml` (defaults: 1 round, 2 steps/epochs for quick tests). Uses conda by default, with `--docker` flag for Docker execution.

```bash
# Basic usage with conda (default)
./train.sh

# Detached mode (anti-hang rule - prevents VSCode client crashes from stopping training)
./train.sh --detached

# Use Docker instead of conda
./train.sh --docker

# Detached mode with Docker
./train.sh --docker --detached

# For custom config, use direct Flower run with overrides
flwr run . local-simulation-serialized-gpu --run-config "num-server-rounds=5 local-epochs=10"

# Or with Docker directly (example with overrides)
docker run --gpus all --shm-size=10.24gb \
  -v $(pwd):/workspace \
  -v $(pwd)/outputs:/workspace/outputs \
  -v /tmp:/tmp \
  -v $HOME/.cache/huggingface:/home/user_lerobot/.cache/huggingface \
  -w /workspace \
  zk0 flwr run . local-simulation-serialized-gpu --run-config "num-server-rounds=5"
```

### Configuration Notes

- Edit `[tool.flwr.app.config]` in `pyproject.toml` for defaults (e.g., num-server-rounds=1, local-epochs=2).
- Use `local-simulation-serialized-gpu` for reliable execution (prevents SafeTensors issues; max-parallelism=1).
- `local-simulation-gpu` for parallel execution (may encounter SafeTensors issues).
- Evaluation frequency: Set via `eval-frequency` in pyproject.toml (0 = every round).

### ⚠️ Important Notes

- **Default execution uses conda** for fast development iteration.
- **Use `--detached` flag** to prevent VSCode client crashes from stopping training (anti-hang rule).
- **Use `--docker` flag** for reproducible, isolated execution when needed.
- **Use `local-simulation-serialized-gpu`** for reliable execution (prevents SafeTensors multiprocessing conflicts).
- **GPU support** requires NVIDIA drivers (conda) or `--gpus all` flag (Docker).
- **Conda provides flexibility** with direct host resource access.
- **Docker provides isolation** and eliminates environment-specific issues.
- **Detached mode** uses tmux sessions for process isolation (critical for remote VSCode connections).

## Result Output

Results of training steps for each client and server logs will be under the `outputs/` directory. For each run there will be a subdirectory corresponding to the date and time of the run. For example:

```
outputs/date_time/
├── simulation.log       # Unified logging output (all clients, server, Flower, Ray)
├── server/              # Server-side outputs
│   ├── server.log       # Server-specific logs
│   ├── eval_policy_loss_chart.png      # 📊 AUTOMATIC: Line chart of per-client and server avg policy loss over rounds
│   ├── eval_policy_loss_history.json   # 📊 AUTOMATIC: Historical policy loss data for reproducibility
│   ├── round_N_server_eval.json        # Server evaluation results
│   ├── federated_metrics.json          # Aggregated FL metrics
│   └── federated_metrics.png           # Metrics visualization
├── clients/             # Client-side outputs
│   └── client_N/        # Per-client directories
│       ├── client.log   # Client-specific logs
│       └── round_N.json # Client evaluation metrics (policy_loss, etc.)
└── models/              # Saved model checkpoints
    └── checkpoint_round_N.safetensors
```

### 📊 Automatic Evaluation Chart Generation

The system automatically generates comprehensive evaluation charts at the end of each training session:

- **📈 `eval_policy_loss_chart.png`**: Interactive line chart showing:
  - Individual client policy loss progression over rounds (Client 0, 1, 2, 3)
  - Server average policy loss across all clients
  - Clear visualization of federated learning convergence

- **📋 `eval_policy_loss_history.json`**: Raw data for reproducibility and analysis:
  - Per-round policy loss values for each client
  - Server aggregated metrics
  - Timestamp and metadata for each evaluation

**No manual steps required** - charts appear automatically after training completion. The charts use intuitive client IDs (0-3) instead of long Ray/Flower identifiers for better readability.

### 💾 Automatic Model Checkpoint Saving

The system automatically saves model checkpoints during federated learning to preserve trained models for deployment and analysis:

#### Checkpoint Saving Configuration

- **Interval-based saving**: Checkpoints saved every N rounds based on `checkpoint_interval` in `pyproject.toml` (default: 5)
- **Final model saving**: Always saves the final model at the end of training regardless of interval
- **Format**: Models saved as `.safetensors` files for efficient storage and loading
- **Location**: `outputs/YYYY-MM-DD_HH-MM-SS/models/` directory

#### Example Checkpoint Files

```
outputs/2025-01-01_12-00-00/models/
├── checkpoint_round_5.safetensors     # After round 5
├── checkpoint_round_10.safetensors    # After round 10
└── checkpoint_round_20.safetensors    # Final model (end of training)
```

#### Configuration Options

```toml
[tool.flwr.app.config]
checkpoint_interval = 5  # Save checkpoint every 5 rounds (0 = disabled)
hf_repo_id = "username/zk0-smolvla-federated"  # Optional: Push final model to Hugging Face Hub
```

#### Hugging Face Hub Integration

- **Automatic pushing**: Final model automatically pushed to Hugging Face Hub if `hf_repo_id` is configured
- **Authentication**: Requires `HF_TOKEN` environment variable for Hub access
- **Model format**: Compatible with Hugging Face model repositories
- **Sharing**: Enables easy model sharing and deployment across different environments

#### Using Saved Models

```python
# Load a saved checkpoint for inference
from safetensors.torch import load_file
from src.task import get_model  # Assuming get_model is available

# Load model architecture
checkpoint_path = "outputs/2025-01-01_12-00-00/models/checkpoint_round_20.safetensors"
state_dict = load_file(checkpoint_path)

# Create model and load weights
model = get_model(dataset_meta)  # dataset_meta from your config
model.load_state_dict(state_dict)
model.eval()

# Use for inference
with torch.no_grad():
    predictions = model(input_data)
```

**No manual intervention required** - model checkpoints are saved automatically during training and can be used for deployment, analysis, or continued training.

## Troubleshooting

- **Training Appears Hung/Stuck**: Use `./train.sh --detached` to isolate training in tmux sessions (anti-hang rule). VSCode client crashes won't stop training processes.
- **Detached Session Management**: `tmux ls` to list sessions, `tmux attach -t <session-name>` to monitor, `tmux kill-session -t <session-name>` to stop.
- **Missing Logs**: Ensure output directory permissions (conda) or Docker volume mounting (`-v $(pwd)/outputs:/workspace/outputs`).
- **Permission Issues**: Check user permissions for log file creation in both conda and Docker environments.
- **Multi-Process Conflicts**: Use `local-simulation-serialized-gpu` for reliable execution.
- **Log Rotation**: Large simulations automatically rotate logs to prevent disk space issues.
- **Dataset Issues**: System uses 0.0001s tolerance (1/fps) for accurate timestamp sync. See [ARCHITECTURE.md](ARCHITECTURE.md) for details.
- **Doubled Datasets**: Automatic hotfix for GitHub issue #1875 applied during loading.
- **Model Loading**: Automatic fallback to simulated training if issues arise.
- **Performance**: Use `pytest -n auto` for parallel testing (see [DEVELOPMENT.md](DEVELOPMENT.md)).
- **SafeTensors Errors**: Switch to `local-simulation-serialized-gpu` or Docker for isolation.
- **GPU Not Detected**: Verify CUDA installation and `nvidia-smi` output.

For advanced troubleshooting, check `simulation.log` in outputs or consult [TECHNICAL-OVERVIEW.md](TECHNICAL-OVERVIEW.md).

If issues persist, ensure you're following the constraints in [INSTALLATION.md](INSTALLATION.md) and the memory bank in `.kilocode/rules/memory-bank/`.