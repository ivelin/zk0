#!/bin/bash
# zk0bot: CLI wrapper for Flower Deployment Engine (conda zk0 env, no Docker)
# Usage: conda activate zk0; ./zk0bot.sh --help
# Server: ./zk0bot.sh server start
# Client: ./zk0bot.sh client start shaunkirby/record-test
# Run: ./zk0bot.sh run --rounds 3 --stream

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check conda zk0 env (safe for unbound vars)
if [ "${CONDA_DEFAULT_ENV:-}" != "zk0" ]; then
    log_info "Auto-activating conda zk0 env..."
    exec conda run -n zk0 --live-stream bash "$0" "$@"
fi

# Prereq checks only for start commands
if [[ "$1" == @(server|client) && "$2" == "start" ]]; then
    # GPU check
    log_info "Checking GPU/CUDA health..."
    if command -v nvidia-smi > /dev/null 2>&1; then
        GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | wc -l)
        log_success "NVIDIA GPU(s): $GPU_COUNT"
    else
        log_warning "nvidia-smi not found. Install NVIDIA drivers/CUDA toolkit."
    fi

    CUDA_AVAILABLE=$(python3 -c "import torch; print('True' if torch.cuda.is_available() else 'False')" 2>/dev/null || echo "False")
    if [ "$CUDA_AVAILABLE" = "True" ]; then
        log_success "PyTorch CUDA ready"
    else
        log_warning "PyTorch CUDA not available - training on CPU (much slower)."
        log_info "Fix: https://pytorch.org/get-started/locally/ (match CUDA version)"
    fi

    # Check tmux (required for SuperLink/SuperNode sessions)
    if ! command -v tmux >/dev/null 2>&1; then
        log_error "tmux required for persistent SuperLink/SuperNode sessions. Install: sudo apt install tmux"
        exit 1
    fi
    log_success "tmux ready"
fi

# Config
SUPERLINK_FLEET_PORT=9092
SUPERLINK_CONTROL_PORT=9093
DEFAULT_CLIENTAPP_PORT=8080
LOG_DIR="$(pwd)/outputs/zk0bot-cli-logs"
mkdir -p "$LOG_DIR"

get_server_ip() {
    if [ -z "${ZK0_SERVER_IP:-}" ]; then
        read -p "Enter Server IP (default: localhost): " ip
        ZK0_SERVER_IP="${ip:-localhost}"
        export ZK0_SERVER_IP
        log_info "ZK0_SERVER_IP set to $ZK0_SERVER_IP (add to ~/.bashrc for persistence)"
    fi
}

usage() {
    cat << EOF
${GREEN}zk0bot.sh${NC} - zk0 Federation CLI (Flower native, conda zk0)
Usage: ./zk0bot.sh <command> [options]

Commands:
  server start|stop|status|logs     Manage SuperLink (ServerApp spawns dynamically)
  client start|stop|status|logs <dataset-uri>  Manage SuperNode (ClientApp spawns dynamically)
  stop                              Kill all zk0-* tmux sessions safely
  run [--rounds <N>] [--stream]     Submit FL run
  status                            Show all zk0 processes/tmux

Examples:
  ./zk0bot.sh server start
  ./zk0bot.sh client start shaunkirby/record-test
  ./zk0bot.sh run --rounds 3 --stream

Logs: $LOG_DIR/
EOF
}

case "${1:-}" in
    server)
        case "$2" in
            start)
                if tmux has-session -t zk0-superlink 2>/dev/null; then
                    log_warning "SuperLink already running"
                else
                    log_info "Starting SuperLink..."
                    tmux new-session -d -s zk0-superlink "flower-superlink --insecure > \"$LOG_DIR/superlink.log\" 2>&1"
                    sleep 2
                    log_success "Server started (ServerApp spawns dynamically on run)"
                    log_info "Fleet API: localhost:$SUPERLINK_FLEET_PORT"
                    log_info "Control API: localhost:$SUPERLINK_CONTROL_PORT"
                fi
                ;;
            stop)
                log_info "Stopping server..."
                tmux kill-session -t zk0-superlink &>/dev/null || true
                log_success "Server stopped"
                ;;
            status)
                log_info "Server status:"
                tmux ls 2>/dev/null | grep zk0-superlink || echo "No tmux zk0-superlink"
                ps aux | grep flower-superlink | grep -v grep || true
                ;;
            logs|log)
                log_info "Server logs (tail -f):"
                tail -f "$LOG_DIR/superlink.log"
                ;;
            *)
                log_error "server <start|stop|status|logs>"
                usage
                exit 1
                ;;
        esac
        ;;
    client)
        ACTION="$2"
        if [[ "$ACTION" == "start" ]]; then
            get_server_ip
        fi
        DATASET_NAME="${3:-}"
        if [[ -z "$DATASET_NAME" && "$ACTION" != "stop" ]]; then
            log_error "dataset-name required for start/status/logs"
            usage
            exit 1
        fi
        ID="$(echo "$DATASET_NAME" | sed 's/[^a-zA-Z0-9-]/-/g' | cut -c1-20)"
        HASH=$(echo -n "$ID" | cksum | cut -d' ' -f1)
        CLIENT_PORT=$((DEFAULT_CLIENTAPP_PORT + (HASH % 100)))
        TMUX_SUPERNODE="zk0-supernode-$ID"
        case "$ACTION" in
            start)
                if tmux has-session -t "$TMUX_SUPERNODE" 2>/dev/null; then
                    log_warning "Client $ID already running"
                else
                    log_info "Starting SuperNode for $DATASET_NAME (ID: $ID)"
                    tmux new-session -d -s "$TMUX_SUPERNODE" "env DATASET_NAME=\"$DATASET_NAME\" flower-supernode --insecure --superlink ${ZK0_SERVER_IP}:${SUPERLINK_FLEET_PORT} --clientappio-api-address 0.0.0.0:${CLIENT_PORT} --isolation subprocess > \"$LOG_DIR/supernode-$ID.log\" 2>&1"
                    log_success "SuperNode $ID started (ClientApp spawns dynamically on run, inherits conda env + DATASET_NAME env var)"
                fi
                ;;
            stop)
                log_info "Stopping client $ID..."
                tmux kill-session -t "$TMUX_SUPERNODE" &>/dev/null || true
                log_success "Client $ID stopped"
                ;;
            status)
                log_info "Client $ID status:"
                tmux ls 2>/dev/null | grep "$TMUX_SUPERNODE" || echo "No session for $ID"
                ;;
            logs|log)
                log_info "Client $ID logs:"
                tail -f "$LOG_DIR/supernode-$ID.log"
                ;;
            *)
                log_error "client <start|stop|status|logs> <dataset-name>"
                exit 1
                ;;
        esac
        ;;
    run)
        ROUNDS=50
        STREAM=""
        shift
        while [[ $# -gt 0 ]]; do
            case $1 in
                --rounds=*) ROUNDS="${1#*=}"; shift ;;
                --stream) STREAM="--stream"; shift ;;
                *) shift ;;
            esac
        done
        log_info "Submitting FL run (rounds: $ROUNDS) via prod-deployment federation (SuperLink control localhost:9093)"
        flwr run . prod-deployment --run-config "num-server-rounds=${ROUNDS}" $STREAM
        ;;
    status)
        log_info "zk0 Status (tmux + processes):"
        tmux ls 2>/dev/null | grep zk0 || echo "No zk0 tmux sessions"
        echo ""
        ps aux | grep -E "(flower-superlink|flower-supernode)" | grep -v grep || echo "No flower processes"
        ;;
    stop)
        log_info "Stopping ALL zk0 tmux sessions (global cleanup)..."
        for session in $(tmux ls 2>/dev/null | grep '^zk0-' | cut -d: -f1); do
            tmux kill-session -t "$session" &>/dev/null || true
        done
        log_success "All zk0 tmux sessions stopped"
        ;;
    --help|-h|"")
        usage
        ;;
    *)
        log_error "Unknown command: $1"
        usage
        exit 1
        ;;
esac
