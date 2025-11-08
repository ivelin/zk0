---
title: "Running zk0: Execute Federated Learning Simulations for SmolVLA"
description: "Instructions to run zk0 federated learning simulations with conda or Docker, training SmolVLA models on SO-100 datasets for decentralized robotics AI development."
---
# Running the Project

This guide covers executing the zk0 federated learning simulation, including default and alternative methods, output details, and troubleshooting.

## Default: Conda Environment Execution

[Installation Guide](INSTALLATION) | [Architecture Overview](ARCHITECTURE) | [Node Operators](NODE-OPERATORS)

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

- **Defaults**: 500 rounds, 4 clients, SO-100/SO-101 datasets.
- **Outputs**: `outputs/<timestamp>/` with logs, metrics, charts (`eval_policy_loss_chart.png`), checkpoint directories, videos.
- **HF Hub Push**: For tiny/debug runs (e.g., `num-server-rounds < checkpoint_interval=10`), the final model push to Hugging Face Hub is skipped to avoid repository clutter with incomplete checkpoints. Local checkpoints are always saved. Full runs (≥10 rounds) will push to the configured `hf_repo_id`.

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

The system automatically saves model checkpoints during federated learning to preserve trained models for deployment and analysis. To optimize disk usage, local checkpoints are gated by intervals, and HF Hub pushes are restricted to substantial runs.

#### Checkpoint Saving Configuration

- **Local Interval-based saving**: Checkpoints saved every N rounds based on `checkpoint_interval` in `pyproject.toml` (default: 10), plus final round always saved
- **HF Hub Push Gating**: Pushes to Hugging Face Hub only occur if `num_server_rounds >= checkpoint_interval` (avoids cluttering repos with tiny/debug runs)
- **Final model saving**: Always saves the final model locally at the end of training regardless of interval
- **Format**: Complete directories with `.safetensors` weights, config, README, and metrics for HF Hub compatibility
- **Location**: `outputs/YYYY-MM-DD_HH-MM-SS/models/` directory

#### Example Checkpoint Files (Full Run: 250 rounds, interval=10)

```
outputs/2025-01-01_12-00-00/models/
├── checkpoint_round_10/     # After round 10 (interval hit)
├── checkpoint_round_20/     # After round 20 (interval hit)
├── checkpoint_round_30/     # After round 30 (interval hit)
...
├── checkpoint_round_250/    # Final model (always saved)
```

#### Example: Tiny Run (2 rounds, interval=10)

```
outputs/2025-01-01_12-00-00/models/
├── checkpoint_round_2/      # Final model only (always saved)
# No intermediate checkpoints; no HF push (2 < 10)
```

#### Configuration Options

```toml
[tool.flwr.app.config]
checkpoint_interval = 10  # Save local checkpoint every N rounds + final (default: 10)
hf_repo_id = "username/zk0-smolvla-fl"  # Optional: Push final model to Hugging Face Hub (only if num_server_rounds >= checkpoint_interval)
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

#### Hugging Face Hub Integration
- **Conditional pushing**: Final model pushed to Hugging Face Hub only if `hf_repo_id` configured AND `num_server_rounds >= checkpoint_interval`
- **Authentication**: Requires `HF_TOKEN` environment variable for Hub access
- **Model format**: Complete directories with safetensors, config, README, and metrics
- **Sharing**: Enables easy model sharing and deployment for meaningful training runs
- **Tiny run protection**: Prevents repo clutter from short validation runs

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

### Experiment Tracking

zk0 integrates with Weights & Biases (WandB) for comprehensive experiment tracking and visualization:

- **Automatic Logging**: When `use-wandb=true` in `pyproject.toml`, training metrics, hyperparameters, and evaluation results are automatically logged to WandB.
- **Model Cards**: Generated README.md files in checkpoint directories include direct links to WandB experiment runs when WandB is enabled.
- **Visualization**: View detailed training curves, client performance, and federated learning metrics in real-time.
- **Setup**: Set `WANDB_API_KEY` in your `.env` file to enable WandB logging.

**Tested**: Completes 500 rounds in ~10-15 minutes; policy loss tracks convergence with early stopping.

## Advanced: Monitoring Runs

To troubleshoot restarts (e.g., PSU overload), use sys_monitor_logs.sh:

- Run `./sys_monitor_logs.sh` before training.
- Logs: gpu_monitor.log (nvidia-smi), system_temps.log (sensors/CPU).
- Post-restart: tail -n 100 gpu_monitor.log | grep power to check spikes.

## Troubleshooting

- **Training Appears Hung/Stuck**: Use `./train.sh --detached` to isolate training in tmux sessions (anti-hang rule). VSCode client crashes won't stop training processes.
- **Detached Session Management**: `tmux ls` to list sessions, `tmux attach -t <session-name>` to monitor, `tmux kill-session -t <session-name>` to stop.
- **Missing Logs**: Ensure output directory permissions (conda) or Docker volume mounting (`-v $(pwd)/outputs:/workspace/outputs`).
- **Permission Issues**: Check user permissions for log file creation in both conda and Docker environments.
- **Multi-Process Conflicts**: Use `local-simulation-serialized-gpu` for reliable execution.
- **Log Rotation**: Large simulations automatically rotate logs to prevent disk space issues.
- **Dataset Issues**: System uses 0.0001s tolerance (1/fps) for accurate timestamp sync. See [ARCHITECTURE](ARCHITECTURE) for details.
- **Doubled Datasets**: Automatic hotfix for GitHub issue #1875 applied during loading.
- **Model Loading**: Automatic fallback to simulated training if issues arise.
- **Performance**: Use `pytest -n auto` for parallel testing (see [DEVELOPMENT](DEVELOPMENT)).
- **SafeTensors Errors**: Switch to `local-simulation-serialized-gpu` or Docker for isolation.
- **GPU Not Detected**: Verify CUDA installation and `nvidia-smi` output.

For advanced troubleshooting, check `simulation.log` in outputs or consult [TECHNICAL-OVERVIEW](TECHNICAL-OVERVIEW).

If issues persist, ensure you're following the constraints in [INSTALLATION](INSTALLATION) and the memory bank in `.kilocode/rules/memory-bank/`.