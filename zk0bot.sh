#!/bin/bash

# zk0bot: CLI tool for zk0 federated learning operations
# Inspired by RocketPool CLI and OpenCode installer

set -e

# Configuration
ZK0_VERSION="v0.7.0"
DOCKER_COMPOSE_SERVER="docker-compose.server.yml"
DOCKER_COMPOSE_CLIENT="docker-compose.client.yml"
SUPEREXEC_IMAGE="zk0-superexec:${ZK0_VERSION}"
NETWORK_NAME="flwr-network"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
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

# Check if Docker and Docker Compose are available
check_dependencies() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not installed or not in PATH"
        exit 1
    fi

    log_success "Dependencies check passed"
}

# Detect Docker Compose command
detect_compose() {
    if docker compose version &> /dev/null 2>&1; then
        COMPOSE_CMD="docker compose"
        log_info "Using docker compose plugin"
    elif command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
        log_info "Using legacy docker-compose"
    else
        log_error "No valid Docker Compose command found"
        exit 1
    fi
}

# Create Docker network
create_network() {
    if ! docker network ls --format "{{.Name}}" | grep -q "^${NETWORK_NAME}$"; then
        log_info "Creating Docker network: ${NETWORK_NAME}"
        docker network create --driver bridge "${NETWORK_NAME}"
        log_success "Network created successfully"
    else
        log_info "Network ${NETWORK_NAME} already exists"
    fi
}

# Build SuperExec image
build_superexec() {
    log_info "Building SuperExec image: ${SUPEREXEC_IMAGE}"
    docker build -f superexec.Dockerfile -t "${SUPEREXEC_IMAGE}" .
    log_success "SuperExec image built successfully"
}

# Pull or build SuperExec image
pull_image() {
    log_info "Checking/pulling SuperExec image: ${SUPEREXEC_IMAGE}"
    if docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "^${SUPEREXEC_IMAGE}$"; then
        log_info "Image ${SUPEREXEC_IMAGE} already exists locally"
    else
        log_info "Image not found, building..."
        build_superexec
    fi
    log_success "SuperExec image ready"
}

# Server commands
server_start() {
    log_info "Starting zk0 server..."
    check_dependencies
    detect_compose
    create_network
    pull_image

    if [ -f "${DOCKER_COMPOSE_SERVER}" ]; then
        ${COMPOSE_CMD} -f "${DOCKER_COMPOSE_SERVER}" up -d
        log_info "Clearing LeRobot HF cache for fresh dataset downloads..."
        docker exec zk0-superexec-server-1 rm -rf /home/user_lerobot/.cache/huggingface/lerobot/* || true
        log_success "Server started successfully (SuperLink + SuperExec, cache cleared)"
        log_info "Server APIs available on:"
        log_info "  - Fleet API: http://localhost:9092"
        log_info "  - ServerApp API: http://localhost:9091"
        log_info "  - Control API: http://localhost:9093"
    else
        log_error "Server compose file not found: ${DOCKER_COMPOSE_SERVER}"
        exit 1
    fi
}

server_stop() {
    log_info "Stopping zk0 server..."
    detect_compose
    if [ -f "${DOCKER_COMPOSE_SERVER}" ]; then
        ${COMPOSE_CMD} -f "${DOCKER_COMPOSE_SERVER}" down
        docker network rm "${NETWORK_NAME}" 2>/dev/null || true
        log_success "Server stopped successfully"
    else
        log_error "Server compose file not found: ${DOCKER_COMPOSE_SERVER}"
        exit 1
    fi
}

server_log() {
    log_info "Showing server logs (SuperExec + FL runner)..."
    detect_compose
    if [ -f "${DOCKER_COMPOSE_SERVER}" ]; then
        ${COMPOSE_CMD} -f "${DOCKER_COMPOSE_SERVER}" logs -f &
        COMPOSE_PID=$!
        sleep 1
        if [ -f /tmp/zk0-flwr.pid ]; then
            FLWR_CONTAINER=$(cat /tmp/zk0-flwr.pid)
            docker logs -f ${FLWR_CONTAINER} || true
        else
            wait $COMPOSE_PID
        fi
    else
        log_error "Server compose file not found: ${DOCKER_COMPOSE_SERVER}"
        exit 1
    fi
}

server_status() {
    log_info "Checking server status..."
    detect_compose
    if [ -f "${DOCKER_COMPOSE_SERVER}" ]; then
        ${COMPOSE_CMD} -f "${DOCKER_COMPOSE_SERVER}" ps
    else
        log_error "Server compose file not found: ${DOCKER_COMPOSE_SERVER}"
        exit 1
    fi
}

server_run() {
    log_info "Starting federated learning run on server (flwr run . local-deployment)..."
    CONTAINER=$(docker ps --filter "ancestor=zk0-superexec:v0.7.0" --filter "name=superexec-server" --format "{{.Names}}" | head -1)
    if [ -z "$CONTAINER" ]; then
        log_error "No zk0-superexec server container running. Run 'zk0bot server start' first."
        exit 1
    fi
    log_info "Executing 'flwr run . local-deployment' in container $CONTAINER"
    docker exec -it "$CONTAINER" flwr run . local-deployment
}

# Client commands
client_start() {
    DATASET_URI="$1"
    if [ -z "${DATASET_URI}" ]; then
        log_error "Dataset URI required. Usage: zk0bot client start <dataset-uri>"
        log_info "Examples:"
        log_info "  zk0bot server start"
        log_info "  zk0bot client start hf:user/my-dataset"
        log_info "  zk0bot client start local:/path/to/dataset"
        exit 1
    fi

    # NOTE: In production, server runs remotely. Local validation for development only.
    # Users must ensure SuperLink (port 9092) is reachable before starting clients.

    log_info "Starting zk0 client with dataset: ${DATASET_URI}..."
    check_dependencies
    detect_compose
    pull_image

    log_info "Using DATASET_URI=${DATASET_URI} for production SuperNode"

    if [ -f "${DOCKER_COMPOSE_CLIENT}" ]; then
        export DATASET_URI="${DATASET_URI}"
        PROJECT_NAME="zk0-client-$(basename "${DATASET_URI}" | sed 's/[^a-zA-Z0-9]/-/g')"
        ${COMPOSE_CMD} --project-name "${PROJECT_NAME}" -f "${DOCKER_COMPOSE_CLIENT}" up -d
        log_success "Client started successfully"
        log_info "Client will connect to server and begin federated learning"
    else
        log_error "Client compose file not found: ${DOCKER_COMPOSE_CLIENT}"
        exit 1
    fi
}

client_stop() {
    DATASET_URI="$1"
    if [ -z "${DATASET_URI}" ]; then
        log_error "Dataset URI required. Usage: zk0bot client stop <dataset-uri>"
        exit 1
    fi

    log_info "Stopping zk0 client with dataset: ${DATASET_URI}..."
    detect_compose
    export DATASET_URI="${DATASET_URI}"
    if [ -f "${DOCKER_COMPOSE_CLIENT}" ]; then
        PROJECT_NAME="zk0-client-$(basename "${DATASET_URI}" | sed 's/[^a-zA-Z0-9]/-/g')"
        ${COMPOSE_CMD} --project-name "${PROJECT_NAME}" -f "${DOCKER_COMPOSE_CLIENT}" down
        log_success "Client stopped successfully (stateless - no state cleanup)"
    else
        log_error "Client compose file not found: ${DOCKER_COMPOSE_CLIENT}"
        exit 1
    fi
}

client_log() {
    DATASET_URI="$1"
    if [ -z "${DATASET_URI}" ]; then
        log_error "Dataset URI required. Usage: zk0bot client log <dataset-uri>"
        exit 1
    fi
    log_info "Showing client logs for dataset: ${DATASET_URI}..."
    detect_compose
    export DATASET_URI="${DATASET_URI}"
    PROJECT_NAME="zk0-client-$(basename "${DATASET_URI}" | sed 's/[^a-zA-Z0-9]/-/g')"
    if [ -f "${DOCKER_COMPOSE_CLIENT}" ]; then
        ${COMPOSE_CMD} --project-name "${PROJECT_NAME}" -f "${DOCKER_COMPOSE_CLIENT}" logs -f
    else
        log_error "Client compose file not found: ${DOCKER_COMPOSE_CLIENT}"
        exit 1
    fi
}

# Config command
config() {
    log_info "zk0 Configuration:"
    echo "Version: ${ZK0_VERSION}"
    echo "Server Compose: ${DOCKER_COMPOSE_SERVER}"
    echo "Client Compose: ${DOCKER_COMPOSE_CLIENT}"
    echo "Image: ${GHCR_IMAGE}"
    echo ""
    log_info "Environment Variables:"
    echo "DATASET_URI: ${DATASET_URI:-not set}"
    echo "HF_TOKEN: ${HF_TOKEN:+set}"
    echo "WANDB_API_KEY: ${WANDB_API_KEY:+set}"
}

# Status command
status() {
    log_info "zk0 Status:"

    echo "Network:"
    if docker network ls --format "{{.Name}}" | grep -q "^${NETWORK_NAME}$"; then
        echo "  ${NETWORK_NAME} exists"
    else
        echo "  ${NETWORK_NAME} not found"
    fi

    echo ""
    echo "Server:"
    detect_compose
    if [ -f "${DOCKER_COMPOSE_SERVER}" ]; then
        ${COMPOSE_CMD} -f "${DOCKER_COMPOSE_SERVER}" ps
    else
        echo "  Compose file not found"
    fi

    echo ""
    echo "Client (dynamic zk0-client-*) :"
    docker ps -a --filter "name=^zk0-client-" --format "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null || echo "  No zk0-client-* containers found"
}

# Global stop - server + all clients
stop() {
    log_info "Stopping zk0 server and all clients..."
    server_stop
    log_info "Stopping all zk0-client-* containers (force clean)..."
    docker rm -f $(docker ps -aq --filter "name=^zk0-client-") 2>/dev/null || true
    log_success "All zk0-client-* containers removed"
    log_success "All zk0 stopped"
}


# Main command dispatcher
main() {
    COMMAND="$1"
    SUBCOMMAND="$2"

    case "${COMMAND}" in
        stop)
            stop
            ;;
        server)
            case "${SUBCOMMAND}" in
                start)
                    server_start
                    ;;
                stop)
                    server_stop
                    ;;
                log)
                    server_log
                    ;;
                run)
                    server_run
                    ;;
                status)
                    server_status
                    ;;
                *)
                    log_error "Invalid server subcommand: ${SUBCOMMAND}"
                    echo "Usage: zk0bot server {start|stop|log|status}"
                    exit 1
                    ;;
            esac
            ;;
        client)
            case "${SUBCOMMAND}" in
                start)
                    client_start "$3"
                    ;;
                stop)
                    client_stop "$3"
                    ;;
                log)
                    client_log "$3"
                    ;;
                *)
                    log_error "Invalid client subcommand: ${SUBCOMMAND}"
                    echo "Usage: zk0bot client {start|stop|log} <dataset-uri>"
                    exit 1
                    ;;
            esac
            ;;
        config)
            config
            ;;
        status)
            status
            ;;
        --help|-h)
            echo "zk0bot: CLI tool for zk0 federated learning operations"
            echo ""
            echo "Usage:"
            echo "  zk0bot server {start|stop|log|run|status}"
            echo "  zk0bot client {start|stop|log} <dataset-uri>"
            echo "  zk0bot config"
            echo "  zk0bot stop                          # Stop server + ALL clients"
            echo "  zk0bot status"
            echo "  zk0bot --help"
            echo ""
            echo "Deployment sequence (MANDATORY):"
            echo "  SERVER FIRST: Ensure SuperLink (port 9092) is running"
            echo "  1. zk0bot server start          # Local dev server"
            echo "  2. zk0bot client start hf:test  # Connects to SuperLink"
            echo ""
            echo "PRODUCTION: Server runs remotely. Clients connect to zk0.superlink:9092"
            echo ""
            echo "Examples:"
            echo "  zk0bot server start"
            echo "  zk0bot client start hf:user/my-dataset"
            echo "  zk0bot status"
            ;;
        *)
            log_error "Invalid command: ${COMMAND}"
            echo "Run 'zk0bot --help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"