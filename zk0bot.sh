#!/bin/bash

# zk0bot: CLI tool for zk0 federated learning operations
# Inspired by RocketPool CLI and OpenCode installer

set -e

# Configuration
ZK0_VERSION="v0.4.0"
DOCKER_COMPOSE_SERVER="docker-compose.server.yml"
DOCKER_COMPOSE_CLIENT="docker-compose.client.yml"
GHCR_IMAGE="ghcr.io/ivelin/zk0:${ZK0_VERSION}"

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

# Pull latest zk0 image
pull_image() {
    log_info "Pulling zk0 image: ${GHCR_IMAGE}"
    docker pull "${GHCR_IMAGE}"
    log_success "Image pulled successfully"
}

# Server commands
server_start() {
    log_info "Starting zk0 server..."
    check_dependencies
    pull_image

    if [ -f "${DOCKER_COMPOSE_SERVER}" ]; then
        docker-compose -f "${DOCKER_COMPOSE_SERVER}" up -d
        log_success "Server started successfully"
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
    if [ -f "${DOCKER_COMPOSE_SERVER}" ]; then
        docker-compose -f "${DOCKER_COMPOSE_SERVER}" down
        log_success "Server stopped successfully"
    else
        log_error "Server compose file not found: ${DOCKER_COMPOSE_SERVER}"
        exit 1
    fi
}

server_log() {
    log_info "Showing server logs..."
    if [ -f "${DOCKER_COMPOSE_SERVER}" ]; then
        docker-compose -f "${DOCKER_COMPOSE_SERVER}" logs -f
    else
        log_error "Server compose file not found: ${DOCKER_COMPOSE_SERVER}"
        exit 1
    fi
}

server_status() {
    log_info "Checking server status..."
    if [ -f "${DOCKER_COMPOSE_SERVER}" ]; then
        docker-compose -f "${DOCKER_COMPOSE_SERVER}" ps
    else
        log_error "Server compose file not found: ${DOCKER_COMPOSE_SERVER}"
        exit 1
    fi
}

# Client commands
client_start() {
    DATASET_URI="$1"
    if [ -z "${DATASET_URI}" ]; then
        log_error "Dataset URI required. Usage: zk0bot client start <dataset-uri>"
        log_info "Examples:"
        log_info "  zk0bot client start hf:user/my-dataset"
        log_info "  zk0bot client start local:/path/to/dataset"
        exit 1
    fi

    log_info "Starting zk0 client with dataset: ${DATASET_URI}..."
    check_dependencies
    pull_image

    if [ -f "${DOCKER_COMPOSE_CLIENT}" ]; then
        export DATASET_URI="${DATASET_URI}"
        docker-compose -f "${DOCKER_COMPOSE_CLIENT}" up -d
        log_success "Client started successfully"
        log_info "Client will connect to server and begin federated learning"
    else
        log_error "Client compose file not found: ${DOCKER_COMPOSE_CLIENT}"
        exit 1
    fi
}

client_stop() {
    log_info "Stopping zk0 client..."
    if [ -f "${DOCKER_COMPOSE_CLIENT}" ]; then
        docker-compose -f "${DOCKER_COMPOSE_CLIENT}" down
        log_success "Client stopped successfully"
    else
        log_error "Client compose file not found: ${DOCKER_COMPOSE_CLIENT}"
        exit 1
    fi
}

client_log() {
    log_info "Showing client logs..."
    if [ -f "${DOCKER_COMPOSE_CLIENT}" ]; then
        docker-compose -f "${DOCKER_COMPOSE_CLIENT}" logs -f
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

    echo "Server:"
    if [ -f "${DOCKER_COMPOSE_SERVER}" ]; then
        docker-compose -f "${DOCKER_COMPOSE_SERVER}" ps
    else
        echo "  Compose file not found"
    fi

    echo ""
    echo "Client:"
    if [ -f "${DOCKER_COMPOSE_CLIENT}" ]; then
        docker-compose -f "${DOCKER_COMPOSE_CLIENT}" ps
    else
        echo "  Compose file not found"
    fi
}

# Main command dispatcher
main() {
    COMMAND="$1"
    SUBCOMMAND="$2"

    case "${COMMAND}" in
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
                    client_stop
                    ;;
                log)
                    client_log
                    ;;
                *)
                    log_error "Invalid client subcommand: ${SUBCOMMAND}"
                    echo "Usage: zk0bot client {start|stop|log} [dataset-uri]"
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
            echo "  zk0bot server {start|stop|log|status}"
            echo "  zk0bot client {start|stop|log} [dataset-uri]"
            echo "  zk0bot config"
            echo "  zk0bot status"
            echo "  zk0bot --help"
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