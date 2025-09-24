# Use the official LeRobot GPU base image
FROM huggingface/lerobot-gpu:latest

# Install OpenCV for video processing in LeRobot datasets

# Set environment variables for better Python behavior
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV UV_LINK_MODE=copy

# Create a working directory for the application
# The app code will be mounted at runtime
WORKDIR /workspace

# Copy requirements file for dependency installation
COPY requirements.txt .

# Install Python dependencies using uv (same as LeRobot)
RUN uv pip install --no-cache-dir --no-build-isolation -r requirements.txt

# Default command (can be overridden)
# The actual app code will be mounted at /workspace
CMD ["echo", "Container ready. Mount your app code to /workspace and run commands."]