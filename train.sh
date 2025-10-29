#!/bin/bash

# SmolVLA Federated Learning Training Script
# This script runs federated learning simulations using conda by default, with Docker as an option

set -e  # Exit on any error

# Configuration managed via pyproject.toml
FEDERATION=${FEDERATION:-local-simulation-serialized-gpu}
DOCKER_IMAGE=zk0
USE_DOCKER=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --docker)
            USE_DOCKER=true
            shift
            ;;
        --tiny)
            TINY_TRAIN=true
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            print_info "Usage: $0 [--docker] [--tiny]"
            exit 1
            ;;
    esac
done

# Configuration managed via environment variables and pyproject.toml
# Use --docker flag to run with Docker instead of conda

print_info "Starting SmolVLA Federated Learning Training"
print_info "=========================================="
print_info "Federation: $FEDERATION"
if [ "$USE_DOCKER" = true ]; then
    print_info "Runtime: Docker ($DOCKER_IMAGE)"
else
    print_info "Runtime: Conda (zk0 environment)"
fi
print_info "(Configuration loaded from pyproject.toml)"
print_info ""

# Check runtime availability and GPU support
if [ "$USE_DOCKER" = true ]; then
    # Docker-specific checks
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed or not in PATH"
        exit 1
    fi

    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running"
        exit 1
    fi

    # Check if GPU support is available (for GPU federations)
    if [[ "$FEDERATION" == *"gpu"* ]]; then
        print_info "Checking GPU availability for federation: $FEDERATION"
        if docker run --rm --gpus all $DOCKER_IMAGE nvidia-smi &> /dev/null; then
            print_info "GPU support confirmed available in Docker"
        else
            print_warning "GPU support not available in Docker. Falling back to CPU federation."
            FEDERATION="local-simulation"
        fi
    fi
else
    # Conda-specific checks
    if ! command -v conda &> /dev/null; then
        print_error "Conda is not installed or not in PATH"
        exit 1
    fi

    # Check if zk0 environment exists
    if ! conda env list | grep -q "^zk0"; then
        print_error "Conda environment 'zk0' does not exist. Run: conda create -n zk0 python=3.10 -y"
        exit 1
    fi

    # Check if GPU support is available (for GPU federations)
    if [[ "$FEDERATION" == *"gpu"* ]]; then
        print_info "Checking GPU availability for federation: $FEDERATION"
        if nvidia-smi &> /dev/null; then
            print_info "GPU support confirmed available on host"
        else
            print_warning "GPU support not available on host. Falling back to CPU federation."
            FEDERATION="local-simulation"
        fi
    fi
fi


# Set Hugging Face verbosity for cache hit visibility
export HF_HUB_VERBOSITY=info
export HF_DATASETS_VERBOSITY=info
export TRANSFORMERS_VERBOSITY=info

if [ "$USE_DOCKER" = true ]; then
    # Set PyTorch CUDA allocator config to reduce fragmentation (pass as env var to container)
    DOCKER_CMD="docker run --gpus all --shm-size=10.24gb -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    DOCKER_CMD="$DOCKER_CMD -v $(pwd):/workspace"
    DOCKER_CMD="$DOCKER_CMD -v /tmp:/tmp"
    # Mount Hugging Face cache directory for model persistence
    DOCKER_CMD="$DOCKER_CMD -v $HOME/.cache/huggingface:/home/user_lerobot/.cache/huggingface"
    DOCKER_CMD="$DOCKER_CMD -w /workspace"
    # Pass Ray environment variables to Docker container (must be set before ray.init())
    DOCKER_CMD="$DOCKER_CMD -e RAY_LOGGING_CONFIG_LOG_LEVEL=INFO"
    DOCKER_CMD="$DOCKER_CMD -e RAY_LOGGING_CONFIG_LOGGER_NAMES=ray,ray.worker,ray.actor"
    DOCKER_CMD="$DOCKER_CMD -e RAY_DEDUP_LOGS=0"
    DOCKER_CMD="$DOCKER_CMD -e RAY_COLOR_PREFIX=1"
    DOCKER_CMD="$DOCKER_CMD $DOCKER_IMAGE"
    DOCKER_CMD="$DOCKER_CMD sh -c 'uv pip install --no-cache-dir --no-build-isolation -r requirements.txt && pip install -e . && PYTHONPATH=/workspace:$PYTHONPATH flwr run . $FEDERATION' 2>&1"

    print_info "Executing Docker command:"
    print_info "$DOCKER_CMD"
    print_info ""

    # Execute the Docker command
    eval "$DOCKER_CMD"
else
    # Set PyTorch CUDA allocator config to reduce fragmentation
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    # Build the conda command with explicit channel configuration
    if [ "$TINY_TRAIN" = true ]; then
        export RAY_LOG_TO_STDERR=1
        export RAY_LOG_LEVEL=DEBUG
        CONDA_CMD="conda run --live-stream -n zk0 sh -c 'pip install -e . && flwr run . $FEDERATION --run-config \"num-server-rounds=2 local-epochs=2 batch_size=2 eval_batches=2 fraction-fit=0.2 fraction-evaluate=0.2\"'"
    else
        CONDA_CMD="conda run --live-stream -n zk0 sh -c 'pip install -e . && flwr run . $FEDERATION'"
    fi

    print_info "Executing conda command:"
    print_info "$CONDA_CMD"
    print_info ""

    # Execute the conda command
    eval "$CONDA_CMD"
fi

# Check exit status
if [ $? -eq 0 ]; then
    print_success "Federated learning training completed successfully!"
    print_info "Check the 'outputs' directory for results and logs."
else
    print_error "Federated learning training failed!"
    exit 1
fi