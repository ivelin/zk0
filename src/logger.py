from loguru import logger
from pathlib import Path
import sys
import os
import logging


def get_config():
    """Load logging configuration from pyproject.toml or return defaults."""
    try:
        # Try to load from pyproject.toml
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from src.core.utils import get_tool_config

        config = get_tool_config("zk0.logging")
        return config
    except Exception:
        # Fallback to defaults if config loading fails
        return {
            "level": "DEBUG",  # Changed to DEBUG for FL diagnostics
            "flwr_log_level": "INFO",  # Flower framework logging level
            "lerobot_log_level": "INFO",  # LeRobot framework logging level
            "enable_grpc_logging": False,  # Disabled to reduce low-level noise
            "enable_ray_logging": True,
            "enable_audit_logging": False,
            "log_format": "detailed",
            "file_rotation": "500 MB",
            "file_retention": "10 days",
            "ray_log_to_driver": True,
            "ray_dedup_logs": True,
            "ray_color_prefix": True,
        }


def get_formats(config):
    """Get console and file log formats from config."""
    if config.get("log_format") == "json":
        console_format = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {extra[client_id]:<10} | PID:{extra[process_id]} | {name}:{function}:{line} - {message}"
        file_format = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {extra[client_id]:<10} | PID:{extra[process_id]} | {name}:{function}:{line} - {message}"
    else:
        console_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        file_format = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {extra[client_id]:<10} | PID:{extra[process_id]} | {name}:{function}:{line} - {message}"
    return console_format, file_format


def get_rotation_mb(config):
    """Get rotation size in MB from config."""
    rotation_size = config.get("file_rotation", "500 MB")
    if "MB" in rotation_size:
        return int(rotation_size.split()[0])
    elif "GB" in rotation_size:
        return int(rotation_size.split()[0]) * 1024
    else:
        return 500  # Default 500MB


def setup_common_logging(
    log_file: Path, level: str = "DEBUG", client_id: str = None, process_id: int = None
):
    """Setup common Loguru configuration for console sinks and framework bridges.

    Handles console output and Flower/Ray logging propagation. Conditional server file sink.

    Args:
        log_file: Path to simulation.log (used for server sink path)
        level: Logging level (DEBUG, INFO, etc.)
        client_id: Optional client identifier for federated learning
        process_id: Optional process ID for multi-process identification
    """

    # clean loguru from any previous context polution
    # required due to the way loguru handles global state while flower / ray recycle client instances withing the same python process
    logger.remove()
    logger.configure(extra={})  # Reset extra contextual data

    # Load configuration from pyproject.toml
    config = get_config()

    # Override level from config if available
    if level is None and "level" in config:
        level = config["level"]

    # Get specific framework log levels, fallback to general level
    flwr_level = config.get("flwr_log_level", level).upper()
    lerobot_level = config.get("lerobot_log_level", level).upper()

    # Set framework environment variables for logging integration
    os.environ["FLWR_LOG_LEVEL"] = flwr_level

    # LeRobot logging (if supported)
    if "LEROBOT_LOG_LEVEL" in os.environ:
        pass  # Keep existing setting
    else:
        os.environ["LEROBOT_LOG_LEVEL"] = lerobot_level

    # Configure gRPC logging if enabled
    if config.get("enable_grpc_logging", False):  # Disabled by default to reduce noise
        os.environ["GRPC_VERBOSITY"] = "INFO"
        os.environ["GRPC_TRACE"] = "all"
    else:
        # Disable gRPC verbose logging
        os.environ["GRPC_VERBOSITY"] = "ERROR"
        os.environ["GRPC_TRACE"] = ""

    # Note: Ray logging configuration is handled via environment variables in train.sh
    # for simulation mode, as Ray must be configured before ray.init() is called by Flower
    # In deployment mode, Ray logging can be configured programmatically here

    # Keep default handler to stderr for console output in case stdout is redirected
    logger.remove()

    # Set default extras
    extras = {}
    extras["client_id"] = client_id if client_id is not None else "server"
    extras["process_id"] = process_id if process_id is not None else os.getpid()

    # Choose format based on config
    if config.get("log_format") == "json":
        # JSON structured logging
        console_format = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {extra[client_id]:<10} | PID:{extra[process_id]} | {name}:{function}:{line} - {message}"
        file_format = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {extra[client_id]:<10} | PID:{extra[process_id]} | {name}:{function}:{line} - {message}"
    else:
        # Enhanced format with client and process info
        console_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        file_format = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {extra[client_id]:<10} | PID:{extra[process_id]} | {name}:{function}:{line} - {message}"

    # Get rotation_mb from config
    rotation_mb = get_rotation_mb(config)

    # Console sink
    logger.add(
        sys.stdout,
        level=level,
        format=console_format,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # Stderr sink to capture Ray worker output and other stderr
    logger.add(
        sys.stderr,
        level=level,
        format=console_format,
        colorize=False,
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )

    retention = config.get("file_retention", "10 days")

    # Add direct file sink for simulation.log (for all processes including server)
    logger.add(
        str(log_file),
        level=level,
        rotation=f"{rotation_mb} MB",
        retention=retention,
        compression="zip",
        format=file_format,
        enqueue=True,
        catch=True,
    )

    # Bind default extras to logger
    if extras:
        logger.configure(extra=extras)

    logger.info(f"Unified logging setup complete for {log_file} at level {level}")

    # Configure Flower logger to prevent duplicates
    flwr_logger = logging.getLogger("flwr")
    flwr_logger.propagate = False  # Prevent propagation to root logger
    flwr_logger.setLevel(getattr(logging, flwr_level))

    # Suppress noisy logging from standard library and third-party libraries
    logging.getLogger("logging").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("lerobot").setLevel(getattr(logging, lerobot_level))
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

    return (
        level,
        file_format,
        retention,
        rotation_mb,
    )


def setup_server_logging(log_file: Path):
    level, file_format, retention, rotation_mb = setup_common_logging(
        log_file=log_file, client_id="server"
    )
    # Create server-specific log if this is the server process
    server_dir = log_file.parent / "server"
    server_dir.mkdir(parents=True, exist_ok=True)
    server_log_file = server_dir / "server.log"
    logger.add(
        str(server_log_file),
        level=level,
        rotation=f"{rotation_mb} MB",
        retention=retention,
        compression="zip",
        format=file_format,
        enqueue=True,
        catch=True,
    )
    logger.info(f"Server-specific logging to {server_log_file}")


def setup_client_logging(log_file: Path, partition_id: int):
    """Setup client-specific logging for individual client processes.

    Args:
        log_file_path: Path to the main simulation log file
        partition_id: Client partition ID for logging
    """

    level, file_format, retention, rotation_mb = setup_common_logging(
        log_file, client_id=f"client_{partition_id}"
    )

    # Create client-specific log file using the provided partition_id (always fresh)
    timestamp_dir = log_file.parent
    client_dir = timestamp_dir / "clients" / f"client_{partition_id}"
    client_dir.mkdir(parents=True, exist_ok=True)
    client_log_file = client_dir / "client.log"

    # Add client-specific sink with enqueue=True for multiprocess safety
    logger.add(
        str(client_log_file),
        level=level,
        rotation=f"{rotation_mb} MB",
        retention=retention,
        compression="zip",
        format=file_format,
        enqueue=True,
        catch=True,
    )
    logger.info(f"Client {partition_id}: Client-specific logging to {client_log_file}")


# Global logger instance
log = logger
