"""lerobot_example: A Flower / Hugging Face LeRobot app."""

from pathlib import Path
from typing import Any
from collections import OrderedDict

import torch
from evaluate import load as load_metric
from torch.optim import AdamW
from torch.utils.data import DataLoader

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

import gym_pusht  # noqa: F401
import gymnasium as gym
import numpy
import imageio
import time

from datasets.utils.logging import disable_progress_bar
from .lerobot_dataset_partitioner import LeRobotDatasetPartitioner
from .lerobot_federated_dataset import FederatedLeRobotDataset

disable_progress_bar()
fds = None  # Cache FederatedDataset


# Create a directory to store the training checkpoint.
timestr = time.strftime("%Y%m%d/%H/%M")
output_directory = Path("outputs/train/lerobot_federated_example") / timestr
output_directory.mkdir(parents=True, exist_ok=True)

# Number of offline training steps (we'll only do offline training for this example.)
# Adjust as you prefer. 5000 steps are needed to get something worth evaluating.
# training_steps = 5000

log_freq = 250

# Set up the dataset.
delta_timestamps = {
    # Load the previous image and state at -0.1 seconds before current frame,
    # then load current image and state corresponding to 0.0 second.
    "observation.image": [-0.1, 0.0],
    "observation.state": [-0.1, 0.0],
    # Load the previous action (-0.1), the next action to be executed (0.0),
    # and 14 future actions with a 0.1 seconds spacing. All these actions will be
    # used to supervise the policy.
    "action": [
        -0.1,
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        1.1,
        1.2,
        1.3,
        1.4,
    ],
}

def get_dataset():
    dataset = LeRobotDataset("lerobot/pusht", delta_timestamps=delta_timestamps)
    return dataset

def load_data(
    partition_id: int, num_partitions: int, model_name: str, device = None
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    """Load pusht data (training and eval)"""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        # Partition the pusht dataset into N partitions
        partitioner = LeRobotDatasetPartitioner(num_partitions=num_partitions)
        fds = FederatedLeRobotDataset(
            dataset="lerobot/pusht",
            partitioners={"train": partitioner},
            delta_timestamps=delta_timestamps,
        )
    partition = fds.load_partition(partition_id)

    # Create dataloader for offline training.
    trainloader = torch.utils.data.DataLoader(
        partition,
        num_workers=4,
        batch_size=64,
        shuffle=True,
        pin_memory=device != torch.device("cpu"),
        drop_last=True,
    )

    # LeRobot pusht test samples are not loaded but rather generated via gym
    testloader = None

    return trainloader, testloader


def get_model(model_name: str = None, dataset = None):
    # Set up the the policy.
    # Policies are initialized with a configuration class, in this case `DiffusionConfig`.
    # For this example, no arguments need to be passed because the defaults are set up for PushT.
    # If you're doing something different, you will likely need to change at least some of the defaults.
    cfg = DiffusionConfig()
    policy = DiffusionPolicy(cfg, dataset_stats=dataset.stats)
    return policy


def get_params(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_params(model, parameters) -> None:
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def train(partition_id: int = None, net = None, trainloader = None, epochs = None, device = None) -> None:
    # in lerobot terminology policy is the neural network
    policy = net
    policy.train()
    # policy.to(device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)

    # Run training loop.
    step = 0
    done = False
    while not done:
        for batch in trainloader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            output_dict = policy.forward(batch)
            loss = output_dict["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % log_freq == 0:
                print(f"train step: {step} train loss: {loss.item():.3f}")
            step += 1
            assert isinstance(epochs, int), f"epochs value: {epochs} , type: {epochs.__class__}"
            if step >= epochs:
                done = True
                break

    # Save a policy checkpoint.
    timestr = time.strftime("%Y%m%d-%H%M%S")
    net.save_pretrained(
        output_directory / f"client_{partition_id}" / f"checkpoint_{timestr}"
    )


def test(partition_id: int, net, device) -> tuple[Any | float, Any]:
    # in lerobot terminology policy is the neural network
    policy = net
    policy.eval()
    # Reset the policy and environmens to prepare for rollout
    policy.reset()

    # Initialize evaluation environment to render two observation types:
    # an image of the scene and state/position of the agent. The environment
    # also automatically stops running after 300 interactions/steps.
    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels_agent_pos",
        max_episode_steps=300,
    )

    numpy_observation, info = env.reset(seed=42)

    # Prepare to collect every rewards and all the frames of the episode,
    # from initial state to final state.
    rewards = []
    frames = []
    successes = []

    # Render frame of the initial state
    frames.append(env.render())

    step = 0
    done = False
    while not done:
        # Prepare observation for the policy running in Pytorch
        state = torch.from_numpy(numpy_observation["agent_pos"])
        image = torch.from_numpy(numpy_observation["pixels"])

        # Convert to float32 with image from channel first in [0,255]
        # to channel last in [0,1]
        state = state.to(torch.float32)
        image = image.to(torch.float32) / 255
        image = image.permute(2, 0, 1)

        # Send data tensors from CPU to GPU
        state = state.to(device, non_blocking=True)
        image = image.to(device, non_blocking=True)

        # Add extra (empty) batch dimension, required to forward the policy
        state = state.unsqueeze(0)
        image = image.unsqueeze(0)

        # Create the policy input dictionary
        observation = {
            "observation.state": state,
            "observation.image": image,
        }

        # Predict the next action with respect to the current observation
        with torch.inference_mode():
            action = policy.select_action(observation)

        # Prepare the action for the environment
        numpy_action = action.squeeze(0).to("cpu").numpy()

        # Step through the environment and receive a new observation
        numpy_observation, reward, terminated, truncated, info = env.step(numpy_action)
        print(f"{step=} {reward=} {terminated=}")

        # Keep track of all the rewards and frames
        rewards.append(reward)
        frames.append(env.render())

        # The rollout is considered done when the success state is reach (i.e. terminated is True),
        # or the maximum number of iterations is reached (i.e. truncated is True)
        done = terminated | truncated | done
        step += 1

    if terminated:
        successes.append(1)
        print("Success! Task completed.")
    else:
        successes.append(0)
        print("Failure! Task incomplete.")

    # Get the speed of environment (i.e. its number of frames per second).
    fps = env.metadata["render_fps"]

    timestr = time.strftime("%Y%m%d-%H%M%S")
    # Encode all frames into a mp4 video.
    video_dir = output_directory / f"client_{partition_id}"
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path =  video_dir / f"rollout_{timestr}.mp4"
    imageio.mimsave(str(video_path), numpy.stack(frames), fps=fps)

    env.close()

    print(f"Video of the evaluation is available in '{video_path}'.")

    return successes, rewards