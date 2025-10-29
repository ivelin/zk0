"""Dataset visualization utilities for SmolVLA federated learning.

This module leverages LeRobot's existing visualization tools instead of reinventing the wheel.
It provides a simple interface to LeRobot's comprehensive HTML visualization system.
"""

from pathlib import Path
from typing import List, Optional, Dict
import json
from loguru import logger

try:
    from lerobot.scripts.visualize_dataset_html import visualize_dataset_html
    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False
    print("Warning: LeRobot not available for visualization")

# Import the new helper
from src.utils import load_lerobot_dataset

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for Docker containers
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

try:
    from lerobot.scripts.record_video import record_video
    RECORD_VIDEO_AVAILABLE = True
except ImportError:
    RECORD_VIDEO_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


class SmolVLAVisualizer:
    """Visualizer for SmolVLA evaluation results using LeRobot's existing tools."""

    def __init__(self):
        self.logger = logger

    def create_local_visualization(self, dataset_repo_id: str,
                                  output_dir: Path,
                                  episodes: Optional[List[int]] = None,
                                  host: str = "127.0.0.1",
                                  port: int = 9090,
                                  tolerance_s: float = 1e-4) -> bool:
        """Create local visualization using LeRobot's existing HTML visualizer.

        This leverages LeRobot's comprehensive visualization system that provides:
        - Interactive episode browsing
        - Video playback with frame-by-frame navigation
        - Real-time data plotting (states, actions, timestamps)
        - Task instruction display
        - Episode metadata and statistics

        Args:
            dataset_repo_id: Hugging Face dataset ID (e.g., 'lerobot/svla_so100_stacking')
            output_dir: Directory to save visualization files
            episodes: List of episode indices to visualize (None for all)
            host: Host for the web server
            port: Port for the web server
            tolerance_s: Timestamp tolerance for dataset loading

        Returns:
            True if visualization server started successfully
        """
        if not LEROBOT_AVAILABLE:
            self.logger.error("LeRobot not available for visualization")
            return False

        try:
            self.logger.info(f"Creating local visualization for dataset: {dataset_repo_id}")

            # Load the dataset using centralized helper
            dataset = load_lerobot_dataset(
                repo_id=dataset_repo_id,
                tolerance_s=tolerance_s
            )

            # Create visualization using LeRobot's tool
            # Try different ports if the default is in use
            max_port_attempts = 10
            current_port = port

            for attempt in range(max_port_attempts):
                try:
                    visualize_dataset_html(
                        dataset=dataset,
                        episodes=episodes,
                        output_dir=output_dir,
                        serve=True,
                        host=host,
                        port=current_port,
                        force_override=True
                    )
                    port = current_port  # Successfully started on this port
                    break
                except SystemExit:
                    # SystemExit is raised by Flask when port is in use
                    current_port += 1
                    self.logger.warning(f"Port {current_port-1} in use (SystemExit), trying port {current_port}")
                    continue
                except OSError as e:
                    # Port is in use or other system error, try next port
                    error_msg = str(e).lower()
                    if ("address already in use" in error_msg or
                        "port" in error_msg or
                        "address in use" in error_msg):
                        current_port += 1
                        self.logger.warning(f"Port {current_port-1} in use, trying port {current_port}")
                        continue
                    else:
                        # Different error, re-raise
                        raise e
                except Exception as e:
                    # Other exceptions, try to continue with different port
                    error_msg = str(e).lower()
                    if ("address already in use" in error_msg or
                        "port" in error_msg or
                        "address in use" in error_msg):
                        current_port += 1
                        self.logger.warning(f"Port {current_port-1} in use, trying port {current_port}")
                        continue
                    else:
                        raise e
            else:
                # All ports tried, skip server start but still create files
                self.logger.warning(f"Could not find available port after {max_port_attempts} attempts, skipping server start")
                try:
                    visualize_dataset_html(
                        dataset=dataset,
                        episodes=episodes,
                        output_dir=output_dir,
                        serve=False,  # Just create files, don't start server
                        host=host,
                        port=port,
                        force_override=True
                    )
                except Exception as e:
                    self.logger.error(f"Failed to create visualization files: {e}")
                    return False

            self.logger.info(f"Local visualization server started at http://{host}:{port}")
            self.logger.info(f"Visualization files saved to: {output_dir}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to create local visualization: {e}")
            return False

    def create_action_chart(self, recorded_actions, predicted_actions, save_path):
        """Create chart comparing recorded vs predicted actions."""
        if not MATPLOTLIB_AVAILABLE:
            self.logger.warning("Matplotlib not available for charting")
            return

        recorded = recorded_actions.detach().cpu().numpy() if hasattr(recorded_actions, 'detach') else recorded_actions
        predicted = predicted_actions.detach().cpu().numpy() if hasattr(predicted_actions, 'detach') else predicted_actions

        min_len = min(len(recorded), len(predicted))
        fig, axes = plt.subplots(recorded.shape[1], 1, figsize=(12, 8))

        for i in range(recorded.shape[1]):
            axes[i].plot(recorded[:min_len, i], label='Recorded', color='blue')
            axes[i].plot(predicted[:min_len, i], label='Predicted', color='red', linestyle='--')
            axes[i].set_title(f'Action Dimension {i+1}')
            axes[i].set_xlabel('Time Step')
            axes[i].set_ylabel('Action Value')
            axes[i].legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Action chart saved: {save_path}")

    def save_episode_video(self, dataset, episode_idx, save_path):
        """Save video for the episode using LeRobot tools."""
        if not RECORD_VIDEO_AVAILABLE:
            self.logger.warning("Record video not available")
            return

        try:
            record_video(
                dataset=dataset,
                episode=episode_idx,
                output_path=save_path
            )
            self.logger.info(f"Episode video saved: {save_path}")
        except Exception as e:
            self.logger.error(f"Failed to save video: {e}")

    def plot_federated_metrics(self, round_metrics: List[Dict], save_dir: Path, wandb_run=None):
        """Plot federated learning metrics across rounds."""
        if not MATPLOTLIB_AVAILABLE:
            self.logger.warning("Matplotlib not available for plotting")
            return

        save_dir.mkdir(exist_ok=True)

        # Extract data
        rounds = [m['round'] for m in round_metrics]
        num_clients = [m['num_clients'] for m in round_metrics]
        avg_losses = [m.get('avg_client_loss', 0) for m in round_metrics]

        # Plot federated metrics
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.plot(rounds, num_clients, marker='s')
        plt.title('Active Clients per Round')
        plt.xlabel('Round')
        plt.ylabel('Number of Clients')

        plt.subplot(2, 2, 2)
        plt.plot(rounds, avg_losses, marker='^')
        plt.title('Average Client Loss')
        plt.xlabel('Round')
        plt.ylabel('Loss')

        plt.tight_layout()
        plt.savefig(save_dir / "federated_metrics.png", dpi=150, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Federated metrics plot saved: {save_dir / 'federated_metrics.png'}")

        # Log to wandb if available and run is initialized and not finished
        if WANDB_AVAILABLE and wandb_run is not None:
            try:
                # Check if wandb run is still active (not finished)
                if hasattr(wandb_run, '_state') and wandb_run._state == 'finished':
                    self.logger.warning("WandB run is already finished, skipping federated metrics logging")
                else:
                    for metrics in round_metrics:
                        wandb_run.log({
                            "round": metrics['round'],
                            "num_clients": metrics['num_clients'],
                            "avg_client_loss": metrics.get('avg_client_loss', 0)
                        }, step=metrics['round'])
            except Exception as e:
                self.logger.warning(f"Failed to log federated metrics to wandb: {e}")

        # Save metrics to JSON
        metrics_file = save_dir / "federated_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(round_metrics, f, indent=2)
        self.logger.info(f"Federated metrics saved: {metrics_file}")

    def plot_eval_policy_loss_chart(self, policy_loss_history: Dict[int, Dict[str, float]], save_dir: Path, wandb_run=None):
        """Plot evaluation policy loss lines for each client and server average over rounds.

        Args:
            policy_loss_history: Dict where keys are round numbers, values are dicts with
                        'client_0', 'client_1', ..., 'server_policy_loss' policy loss values.
                        Can also include aggregated metrics from consolidated server eval files.
            save_dir: Directory to save the chart and history JSON.
            wandb_run: Optional wandb run object for logging metrics.
        """
        if not MATPLOTLIB_AVAILABLE:
            self.logger.warning("Matplotlib not available for MSE chart generation")
            return

        if not policy_loss_history:
            self.logger.warning("No policy loss history data provided for chart generation")
            return

        save_dir.mkdir(exist_ok=True)

        # Extract rounds and sort them
        rounds = sorted(policy_loss_history.keys())
        if not rounds:
            self.logger.warning("No rounds found in policy loss history")
            return

        # Get all client keys (excluding server_policy_loss)
        sample_round_data = policy_loss_history[rounds[0]]
        client_keys = [k for k in sample_round_data.keys() if k.startswith('client_')]
        client_ids = sorted([int(k.split('_')[1]) for k in client_keys])

        # Prepare data for plotting
        client_policy_losses = {cid: [] for cid in client_ids}
        server_policy_losses = []

        for round_num in rounds:
            round_data = policy_loss_history[round_num]
            for cid in client_ids:
                client_key = f'client_{cid}'
                policy_loss_val = round_data.get(client_key, float('nan'))  # Use NaN if missing
                client_policy_losses[cid].append(policy_loss_val)
            server_policy_loss = round_data.get('server_policy_loss', float('nan'))
            server_policy_losses.append(server_policy_loss)

        # Create plot
        plt.figure(figsize=(12, 8))

        # Color cycle for clients
        colors = plt.cm.tab10.colors  # Use matplotlib's default color cycle

        # Plot each client line
        for i, cid in enumerate(client_ids):
            color = colors[i % len(colors)]
            plt.plot(rounds, client_policy_losses[cid], label=f'Client {cid}', color=color, marker='o', linewidth=2)

        # Plot server average with bold line
        plt.plot(rounds, server_policy_losses, label='Server Avg', color='black', linewidth=4, marker='s')

        # Add labels, title, legend
        plt.xlabel('Round Number')
        plt.ylabel('Policy Loss')
        plt.title('Federated Learning Evaluation Policy Loss Over Rounds')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)

        # Save chart
        chart_path = save_dir / "policy_loss_chart.png"
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Policy loss chart saved: {chart_path}")

        # Log final metrics to wandb if run is available and not finished
        if WANDB_AVAILABLE and wandb_run is not None:
            try:
                # Check if wandb run is still active (not finished)
                if hasattr(wandb_run, '_state') and wandb_run._state == 'finished':
                    self.logger.warning("WandB run is already finished, skipping final metrics logging")
                else:
                    # Log final policy loss values for each client and server average
                    final_round = max(policy_loss_history.keys())
                    final_data = policy_loss_history[final_round]

                    wandb_metrics = {"final_round": final_round}
                    for key, value in final_data.items():
                        wandb_metrics[f"final_{key}_policy_loss"] = value

                    wandb_run.log(wandb_metrics)
                    self.logger.info(f"Final policy loss metrics logged to wandb for round {final_round}")
            except Exception as e:
                self.logger.warning(f"Failed to log final metrics to wandb: {e}")

        # Save history to JSON for reproducibility
        history_file = save_dir / "policy_loss_history.json"
        with open(history_file, 'w') as f:
            json.dump(policy_loss_history, f, indent=2)
        self.logger.info(f"Policy loss history saved: {history_file}")

