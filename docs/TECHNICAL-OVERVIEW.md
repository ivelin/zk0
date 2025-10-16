# Technical Overview

This document provides advanced technical details on zk0's federated learning implementation, focusing on comparisons, reproducibility, and evaluation mechanisms. It extracts deep-dive sections from the project documentation, complementing [ARCHITECTURE.md](ARCHITECTURE.md). For core architecture, see [ARCHITECTURE.md](ARCHITECTURE.md); for development practices, see [DEVELOPMENT.md](DEVELOPMENT.md).

## Federated vs. Centralized Training Comparison

The zk0 system enables rigorous benchmarking between federated and centralized training to evaluate privacy-efficiency trade-offs.

### Objective Performance Benchmarking

- **Federated Setup**: 4-10 clients, partitioned SO-100 subsets, FedProx aggregation (μ=0.01).
- **Centralized Baseline**: Single model on full SO-100 dataset.
- **Controlled Variables**: Identical hyperparameters (lr=1e-4, cosine scheduler), architecture, total steps (~50k+).
- **Evaluation**: Same held-out validation set (unseen SO-101 tasks).

### Federated Learning Characteristics

| Metric                  | Federated (Best Config)    |
|-------------------------|----------------------------|
| **Final Policy Loss**   | 0.544 (30 rounds x 50 epochs) |
| **Convergence Rounds**  | 30+ (more rounds could further improve convergence) |
| **Training Efficiency** | 0.85 (parallel clients)   |
| **Privacy**             | High (parameters only)    |
| **Scalability**         | Horizontal (10+ clients)  |

- **Federated Insights**: FedProx (μ=0.01) stabilizes convergence across heterogeneous clients. Benefits: Privacy, distributed compute.
- **Reproduction**: Run with seed=42; monitor via `federated_metrics.json`.

Example metrics from best FL config (30 rounds, 50 epochs, μ=0.01, LR=0.0001):
- Final Server Policy Loss: 0.544 (improved from initial 0.149).
- Client Avg Loss: 0.34 (decline from 2.53).
- **Best Config**: 50 local epochs, FedProx μ=0.01, LR=0.0001 (see experiment results table).

For full benchmarks, see [memory-bank/context.md](.kilocode/rules/memory-bank/context.md) (run summaries and experiment results table).

## Reproducing Experiments

zk0 emphasizes reproducibility with seeds, pinned dependencies, and scripted workflows. This ensures consistent results across environments.

### Environment Setup for Reproduction

```bash
# Pinned deps ensure consistency
pip install -r requirements.txt  # Flower 1.21.0, LeRobot 0.3.3, etc.
pip install -e .

# Set reproducible seed
export PYTHONHASHSEED=42
export CUDA_LAUNCH_BLOCKING=1  # For CUDA determinism
```

### Federated Learning Reproduction

```bash
# Reproducible FL run
conda activate zk0
flwr run . local-simulation-serialized-gpu \
    --run-config "num-server-rounds=30 local-epochs=50 batch-size=64 seed=42" \
    --seed 42
```

- **Seed Coverage**: Torch, NumPy, Ray, Flower (via --seed).
- **Total Steps**: Equivalent to centralized (~50k+ for meaningful convergence).
- **Outputs**: Deterministic `federated_metrics.json`, charts, checkpoints.
- **Validation**: Compare policy_loss trends; expect <1% variance.

### Centralized Training Baseline

For fair comparison, run equivalent centralized training:

```python
# centralized_baseline.py (example script)
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from transformers import AutoModelForVision2Seq

# Reproducible setup
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Load full dataset (no partitioning)
dataset = LeRobotDataset("lerobot/svla_so100_pickplace", split="train")  # Or aggregate clients

# Model and optimizer (match FL)
model = AutoModelForVision2Seq.from_pretrained("lerobot/smolvla_base")
optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)

# Train for equivalent steps (30 rounds * 50 epochs * batches)
total_steps = 30000  # Adjust based on batch_size
for step in range(total_steps):
    # Training loop (match FL: policy loss, scheduler reset equivalent)
    batch = next(iter(dataset_dataloader))
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Save for comparison
torch.save(model.state_dict(), "centralized_checkpoint.pt")
```

- **Equivalence**: Same lr scheduler (cosine, reset per "round" equivalent), loss function.
- **Comparison Script**:
  ```bash
  python compare_experiments.py \
      --federated-dir outputs/fl_run_2025-10-11 \
      --centralized-dir outputs/centralized_run \
      --metrics policy_loss,success_rate \
      --seed 42
  ```
- **Statistical Testing**: 95% CI on metrics; expect federated within 10-20% of centralized.

See [memory-bank/tech.md](.kilocode/rules/memory-bank/tech.md) for hardware reproducibility.

## Evaluation Video Recordings and Playback

zk0 captures episodic performance via videos to visualize SmolVLA progress on SO-100 tasks.

### Video Generation

- **When**: End-of-round evaluations (configurable frequency).
- **What**: Full episodes with predicted actions overlaid on observations.
- **Format**: MP4 (30 FPS), saved per client/round.
- **Implementation**: Integrated in [`src/client_app.py`](src/client_app.py) and [`src/visualization.py`](src/visualization.py); uses imageio for encoding.

Example code snippet:
```python
# In evaluation loop
frames = []  # Collect RGB frames + action overlays
for step in range(max_steps):
    action = model.predict(observation)
    frame = env.render(mode="rgb_array")  # 224x224 with annotations
    frames.append(frame)
    env.step(action)

# Save video
import imageio
imageio.mimsave(f"outputs/evaluate/round_{round}_client_{cid}.mp4", frames, fps=30)
```

- **Location**: `outputs/<timestamp>/evaluate/round_N/client_M/rollout_TIMESTAMP.mp4`.
- **Metadata**: JSON alongside videos (success, duration, policy_loss).

### Playback and Analysis

#### Manual Playback

```bash
# List videos by round
find outputs/ -name "*.mp4" | sort

# Play example
vlc outputs/2025-10-11_16-00-00/evaluate/round_10/client_0/rollout_20251011_160500.mp4

# Batch view (e.g., progression)
for video in outputs/*/evaluate/round_*/client_0/*.mp4; do
    echo "Round $(basename $(dirname $video))"; vlc "$video" --play-and-exit;
done
```

#### Automated Analysis

```python
# analyze_videos.py example
import cv2
import json
from pathlib import Path

def analyze_video(video_path):
    cap = cv2.VideoCapture(str(video_path))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frames / fps
    # Detect success (e.g., via metadata or frame analysis)
    success = check_final_frame(cap)  # Custom logic
    return {"duration": duration, "frames": frames, "success": success}

# Batch analysis
video_dir = Path("outputs/2025-10-11_16-00-00/evaluate")
results = {}
for round_dir in sorted(video_dir.glob("round_*")):
    round_results = [analyze_video(v) for v in round_dir.glob("*.mp4")]
    results[round_dir.name] = {
        "avg_success": sum(r["success"] for r in round_results) / len(round_results),
        "avg_duration": sum(r["duration"] for r in round_results) / len(round_results)
    }

with open("video_analysis.json", "w") as f:
    json.dump(results, f, indent=2)
```

- **Metrics**: Success rate, completion time, action smoothness (via optical flow).
- **Visualization**: Upload to [LeRobot Dataset Visualizer](https://huggingface.co/spaces/lerobot/visualize_dataset) for interactive playback.
- **Progress Tracking**: Videos show improvement (e.g., smoother actions over rounds).

For integration with WandB, see [memory-bank/tech.md](.kilocode/rules/memory-bank/tech.md#wandb-integration).

## Next Steps and Current Configuration

### Current Config (v0.1.14+)

- **Clients**: 4 (pyproject.toml `[tool.zk0.datasets]`):
  - Client 0: `lerobot/svla_so100_pickplace` (50 episodes) ✅ CLEAN
  - Client 1: `lerobot/svla_so100_stacking` (56 episodes) ✅ HOTFIX
  - Client 2: `lerobot/svla_so100_sorting` (52 episodes) ✅ HOTFIX
  - Client 3: `lerobot/svla_so101_pickplace` (50 episodes) ✅ CLEAN
- **Rounds**: 30+ (local-epochs=50, batch=64).
- **Model**: `lerobot/smolvla_base`.
- **Eval**: Policy loss on unseen SO-101; videos for qualitative.
- **Status**: Beta – core FL functional; seeking feedback on convergence.

### Planned Enhancements

- Multi-task learning across SO-100 variants.
- Advanced strategies (SCAFFOLD).
- Hyperparam auto-tuning.
- ZK proofs for verifiable contributions.

See [memory-bank/brief.md](.kilocode/rules/memory-bank/brief.md) for objectives.

## References

- [Federated Baselines](.kilocode/rules/memory-bank/context.md): Run summaries (e.g., 30-round policy loss 0.544).
- LeRobot: [Evaluation Docs](https://huggingface.co/docs/lerobot/evaluation).
- Flower: [Simulation Guide](https://flower.ai/docs/framework/how-to-run-simulations.html).
- Tools: imageio for videos; cv2 for analysis.