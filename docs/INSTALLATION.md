# Installation

## Environment Preferences
- **Conda (Recommended for Development)**: Preferred for fast iteration and direct host GPU access. Use for local development and testing.
- **Docker (Recommended for Production/Reproducibility)**: Preferred for isolated, reproducible runs. Use `--docker` flag in train.sh or direct Docker commands for consistent environments across machines.

## Standard Installation

1. Create the zk0 environment:
   ```
   conda create -n zk0 python=3.10 -y
   conda activate zk0
   ```

2. Install CUDA-enabled PyTorch (for GPU support):
   ```
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130 --no-cache-dir
   ```

3. Install LeRobot (latest version, manually before project install):
   ```
   pip install lerobot[smolvla]==0.3.3
   ```

4. Install project dependencies from pyproject.toml:
   ```
   pip install -e .
   ```

5. Verify GPU:
   ```
   python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
   ```
   - Expected: `True`.

## Troubleshooting GPU Issues

- **CUDA Not Available**: Run `nvidia-smi` to confirm GPU. If True, reinstall PyTorch with the cu130 index.
- **Slow Execution**: Check logs for "Running test() on device 'cpu'". Ensure `model.to(device)` is called in code (added in src/server_app.py and src/task.py).
- **Dependency Conflicts**: Comment out "torch>=2.5.0" in pyproject.toml to avoid reinstalls; install manually with CUDA index.
- **Video Decoding**: If "No accelerated backend detected", install CUDA toolkit: `conda install cudatoolkit=13.0 -c nvidia` and set `export VIDEO_BACKEND=torchcodec`.

## Running

- Tiny test: `./train.sh --tiny`
- Full training: `./train.sh`
- Docker (GPU): `./train.sh --docker`

For other environments with torch CUDA issues, use the same pip install command with the appropriate CUDA version (e.g., cu121 for CUDA 12.1).

For full execution instructions, see [RUNNING.md](RUNNING.md).