#!/bin/bash

# SmolVLA Standalone Training Script
# This script runs standalone LeRobot training using conda (default) or Docker for reproducible execution

set -e  # Exit on any error

# Default values
STEPS=${STEPS:-2}
MODE=${MODE:-conda}  # Default to conda, can be set to 'docker'
DOCKER_IMAGE=${DOCKER_IMAGE:-zk0}
DATASET_REPO_ID=""  # Must be provided by user

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

# Function to show usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Run SmolVLA standalone training using conda (default) or Docker"
    echo ""
    echo "Options:"
    echo "  -d, --dataset REPO     Hugging Face dataset repo ID (required)"
    echo "  -s, --steps NUM        Number of training steps (default: 200)"
    echo "  -m, --mode MODE        Execution mode: 'conda' or 'docker' (default: conda)"
    echo "  -i, --image NAME       Docker image name (default: zk0, used only with --mode docker)"
    echo "  -h, --help            Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  MODE=docker           Use Docker instead of conda"
    echo "  DOCKER_IMAGE=zk0      Docker image to use"
    echo ""
    echo "Examples:"
    echo "  $0                     # Run with defaults (conda mode, 200 steps)"
    echo "  $0 -s 1000             # Run 1000 steps with conda"
    echo "  $0 -m docker           # Use Docker instead of conda"
    echo "  $0 -m docker -i custom-image  # Use custom Docker image"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dataset)
            DATASET_REPO_ID="$2"
            shift 2
            ;;
        -s|--steps)
            STEPS="$2"
            shift 2
            ;;
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        -i|--image)
            DOCKER_IMAGE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate inputs
if [[ -z "$DATASET_REPO_ID" ]]; then
    print_error "Dataset repo ID is required. Use -d or --dataset"
    usage
    exit 1
fi

if ! [[ "$STEPS" =~ ^[0-9]+$ ]] || [ "$STEPS" -lt 1 ]; then
    print_error "Steps must be a positive integer"
    exit 1
fi

if [[ "$MODE" != "conda" && "$MODE" != "docker" ]]; then
    print_error "Mode must be either 'conda' or 'docker'"
    exit 1
fi

print_info "Starting SmolVLA Standalone Training"
print_info "===================================="
print_info "Dataset: $DATASET_REPO_ID"
print_info "Steps: $STEPS"
print_info "Mode: $MODE"
if [[ "$MODE" == "docker" ]]; then
    print_info "Docker Image: $DOCKER_IMAGE"
fi
print_info ""

# Function to execute conda training
execute_conda_training() {
    print_info "Using conda environment: zk0"

    # Check if conda is available
    if ! command -v conda &> /dev/null; then
        print_error "Conda is not installed or not in PATH"
        exit 1
    fi

    # Check if the zk0 environment exists
    if ! conda info --envs | grep -q "zk0"; then
        print_error "Conda environment 'zk0' not found"
        print_error "Please create it first with: conda create -n zk0 python=3.10"
        exit 1
    fi

    # Check if GPU is available
    if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
        print_info "GPU detected and available"
    else
        print_warning "GPU not detected. Training will use CPU."
    fi


    # Execute the training command
    conda run -n zk0 python -m lerobot.scripts.train \
      --policy.path=lerobot/smolvla_base \
      --dataset.repo_id="$DATASET_REPO_ID" \
      --batch_size=64 \
      --steps=$STEPS \
      --output_dir=outputs/train/my_smolvla \
      --job_name=my_smolvla_training \
      --policy.device=cuda
}

# Function to execute docker training
execute_docker_training() {
    print_info "Using Docker image: $DOCKER_IMAGE"

    # Check if Docker is available
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed or not in PATH"
        exit 1
    fi

    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running"
        exit 1
    fi

    # Check if GPU support is available
    if ! docker run --rm --gpus all $DOCKER_IMAGE nvidia-smi &> /dev/null; then
        print_warning "GPU support not available in Docker. Training will use CPU."
    fi

    # Build the Docker command
    DOCKER_CMD="docker run --gpus all --shm-size=10.24gb"
    DOCKER_CMD="$DOCKER_CMD -v $(pwd):/workspace"
    DOCKER_CMD="$DOCKER_CMD -v $(pwd)/outputs:/workspace/outputs"
    DOCKER_CMD="$DOCKER_CMD -v /tmp:/tmp"
    # Mount Hugging Face cache directory for model persistence
    DOCKER_CMD="$DOCKER_CMD -v $HOME/.cache/huggingface:/home/user_lerobot/.cache/huggingface"
    DOCKER_CMD="$DOCKER_CMD -w /workspace"
    DOCKER_CMD="$DOCKER_CMD $DOCKER_IMAGE"
    DOCKER_CMD="$DOCKER_CMD sh -c \"uv pip install --no-cache-dir --no-build-isolation -r requirements.txt && PYTHONPATH=/workspace python -m lerobot.scripts.train \
      --policy.path=lerobot/smolvla_base \
      --dataset.repo_id=tinkhireeva/so101_duck_sort \
      --batch_size=64 \
      --steps=$STEPS \
      --output_dir=outputs/train/my_smolvla \
      --job_name=my_smolvla_training \
      --policy.device=cuda \
      --policy.repo_id=ivelin/smolvla_test
      \""

    print_info "Executing Docker command:"
    print_info "$DOCKER_CMD"
    print_info ""

    # Execute the command
    eval "$DOCKER_CMD"
}

# Set Hugging Face verbosity for cache hit visibility
export HF_HUB_VERBOSITY=info
export HF_DATASETS_VERBOSITY=info
export TRANSFORMERS_VERBOSITY=info

# Execute based on mode
if [[ "$MODE" == "conda" ]]; then
    execute_conda_training
elif [[ "$MODE" == "docker" ]]; then
    execute_docker_training
fi

# Check exit status
if [ $? -eq 0 ]; then
    print_success "SmolVLA standalone training completed successfully!"
    print_info "Check the 'outputs/train/my_smolvla' directory for results and logs."
else
    print_error "SmolVLA standalone training failed!"
    exit 1
fi