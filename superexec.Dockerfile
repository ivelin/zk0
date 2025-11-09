FROM huggingface/lerobot-gpu:latest

# Set working directory
WORKDIR /workspace

# Install Flower with superexec support using uv
RUN uv pip install "flwr[superexec]==1.23.0"

# Copy and install zk0
COPY pyproject.toml /workspace/pyproject.toml
COPY src /workspace/src

# Install zk0 using uv (includes simulation extras for simplicity)
RUN uv pip install -e .

# Add venv bin to PATH if needed (lerobot uses /lerobot/.venv)
ENV PATH="/lerobot/.venv/bin:$PATH"

ENTRYPOINT ["flower-superexec"]