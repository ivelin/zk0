---
tags: [robotics, ai, zk0]
dataset: [pusht]
framework: [lerobot]
---

# Federated Learning with HuggingFace LeRobot and Flower (Quickstart Example)

This introductory example to using [🤗LeRobot](https://huggingface.co/lerobot) with Flower. 

In this example, we will federate the training of a [Diffusion](https://arxiv.org/abs/2303.04137) policy on the [PushT](https://huggingface.co/datasets/lerobot/pusht/tree/v1.3) dataset. The data will be downloaded and partitioned using [Flower Datasets](https://flower.ai/docs/datasets/). This example runs best when a GPU is available.

https://github.com/user-attachments/assets/a9771310-e48d-4426-9cc5-850b2efabae3

## Set up the project

### Clone the project

Start by cloning the example project. We prepared a single-line command that you can copy into your shell which will checkout the example for you:

```shell
git clone https://github.com/ivelin/zk0
cd federate
```

This will create a new directory called `quickstart-lerobot` containing the following files:

```shell
quickstart-huggingface
├── lerobot_example
│   ├── __init__.py
│   ├── client_app.py   # Defines your ClientApp
│   ├── server_app.py   # Defines your ServerApp
│   └── task.py         # Defines your model, training and data loading
├── pyproject.toml      # Project metadata like dependencies and configs
└── README.md
```

### Install dependencies and project

Install the dependencies defined in `pyproject.toml` as well as the `lerobot_example` package.

```bash
pip install -e .
```

## Run the Example

You can run your Flower project in both _simulation_ and _deployment_ mode without making changes to the code. If you are starting with Flower, we recommend you using the _simulation_ mode as it requires fewer components to be launched manually. By default, `flwr run` will make use of the Simulation Engine.

### Run with the Simulation Engine

> \[!TIP\]
> This example runs faster when the `ClientApp`s have access to a GPU. If your system has one, you can make use of it by configuring the `backend.client-resources` component in `pyproject.toml`. If you want to try running the example with GPU right away, use the `local-simulation-gpu` federation as shown below.

```bash
# Run with the default federation (CPU only)
flwr run .
```

Run the project in the `local-simulation-gpu` federation that gives CPU and GPU resources to each `ClientApp`. By default, at most 4x`ClientApp` (using ~1 GB of VRAM each) will run in parallel in each available GPU. Note you can adjust the degree of paralellism but modifying the `client-resources` specification.

```bash
# Run with the `local-simulation-gpu` federation
flwr run . local-simulation-gpu
```

You can also override some of the settings for your `ClientApp` and `ServerApp` defined in `pyproject.toml`. For example

```bash
flwr run --run-config "num-server-rounds=5 fraction-fit=0.1"
```
