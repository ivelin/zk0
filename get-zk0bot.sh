#!/bin/bash

# zk0bot Installer
# One-line installer for zk0bot CLI tool
# Usage: curl -fsSL https://get.zk0.bot | bash

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging
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

# Detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        echo "windows"
    else
        echo "unknown"
    fi
}

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first: https://docs.docker.com/get-docker/"
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi

    log_success "Docker and Docker Compose are installed"
}

# Download zk0bot
download_zk0bot() {
    local os="$1"
    local version="v0.4.0"
    local repo="ivelin/zk0"
    local binary_name="zk0bot.sh"

    log_info "Downloading zk0bot ${version} for ${os}..."

    # For now, assume the script is in the same repo
    # In production, this would download from GitHub releases
    local download_url="https://raw.githubusercontent.com/${repo}/${version}/zk0bot.sh"

    if command -v curl &> /dev/null; then
        curl -fsSL "${download_url}" -o "${binary_name}"
    elif command -v wget &> /dev/null; then
        wget -q "${download_url}" -O "${binary_name}"
    else
        log_error "Neither curl nor wget found. Please install one of them."
        exit 1
    fi

    chmod +x "${binary_name}"
    log_success "Downloaded and made executable: ${binary_name}"
}

# Install zk0bot to PATH
install_zk0bot() {
    local install_dir=""
    local binary_name="zk0bot.sh"

    # Determine install directory
    if [[ -w "/usr/local/bin" ]]; then
        install_dir="/usr/local/bin"
    elif [[ -w "/usr/bin" ]]; then
        install_dir="/usr/bin"
    else
        # User install
        install_dir="${HOME}/.local/bin"
        mkdir -p "${install_dir}"
        # Add to PATH if not already there
        if [[ ":$PATH:" != *":${install_dir}:"* ]]; then
            log_warning "Adding ${install_dir} to PATH. You may need to restart your shell or run: export PATH=\"${install_dir}:\$PATH\""
            echo "export PATH=\"${install_dir}:\$PATH\"" >> "${HOME}/.bashrc"
            echo "export PATH=\"${install_dir}:\$PATH\"" >> "${HOME}/.zshrc" 2>/dev/null || true
        fi
    fi

    log_info "Installing zk0bot to ${install_dir}..."
    mv "${binary_name}" "${install_dir}/zk0bot"
    log_success "zk0bot installed successfully!"
    log_info "Run 'zk0bot --help' to get started"
}

# Main installation
main() {
    log_info "Welcome to zk0bot installer!"
    log_info "This will install the zk0 federated learning CLI tool."

    local os
    os=$(detect_os)

    if [[ "${os}" == "unknown" ]]; then
        log_error "Unsupported OS: ${OSTYPE}"
        exit 1
    fi

    log_info "Detected OS: ${os}"

    check_docker
    download_zk0bot "${os}"
    install_zk0bot

    log_success "Installation complete!"
    echo ""
    log_info "Next steps:"
    echo "  1. Ensure Docker is running: docker --version"
    echo "  2. Start a server: zk0bot server start"
    echo "  3. Or start a client: zk0bot client start hf:user/dataset"
    echo ""
    log_info "For more info, visit: https://github.com/ivelin/zk0"
}

main "$@"