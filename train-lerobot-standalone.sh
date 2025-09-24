#!/bin/bash

# SmolVLA Standalone Training Script
# This script runs standalone LeRobot training using Docker for reproducible execution

set -e  # Exit on any error

# Default values
STEPS=${STEPS:-2}
DOCKER_IMAGE=${DOCKER_IMAGE:-zk0}

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
    echo "Run SmolVLA standalone training using Docker"
    echo ""
    echo "Options:"
    echo "  -s, --steps NUM        Number of training steps (default: 200)"
    echo "  -i, --image NAME       Docker image name (default: zk0)"
    echo "  -h, --help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                     # Run with defaults (200 steps)"
    echo "  $0 -s 1000             # Run 1000 steps"
    echo "  $0 -i custom-image     # Use custom Docker image"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--steps)
            STEPS="$2"
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
if ! [[ "$STEPS" =~ ^[0-9]+$ ]] || [ "$STEPS" -lt 1 ]; then
    print_error "Steps must be a positive integer"
    exit 1
fi

print_info "Starting SmolVLA Standalone Training"
print_info "===================================="
print_info "Steps: $STEPS"
print_info "Docker Image: $DOCKER_IMAGE"
print_info ""

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
  --dataset.repo_id=sergiov2000/so100_movella_ball_usability_u3_stack1 \
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

# Set Hugging Face verbosity for cache hit visibility
export HF_HUB_VERBOSITY=info
export HF_DATASETS_VERBOSITY=info
export TRANSFORMERS_VERBOSITY=info

# Execute the command
eval "$DOCKER_CMD"

# Check exit status
if [ $? -eq 0 ]; then
    print_success "SmolVLA standalone training completed successfully!"
    print_info "Check the 'outputs/train/my_smolvla' directory for results and logs."
else
    print_error "SmolVLA standalone training failed!"
    exit 1
fi