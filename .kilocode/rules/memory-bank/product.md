# Product Vision

**Created**: 2025-09-06
**Last Updated**: 2025-09-07
**Version**: 1.0.0
**Author**: Kilo Code

## Why This Project Exists
Federated learning enables training AI models on distributed data without compromising privacy. In robotics, this is crucial for sharing manipulation skills across different environments while keeping sensitive data local.

## Problems Solved
- Privacy concerns in robotics data sharing
- Scalability issues with centralized training
- Data heterogeneity across different robotic setups
- Resource constraints in distributed environments

## How It Should Work
Clients train SmolVLA models locally on their SO-100 datasets, send model updates to a central server, which aggregates them using Flower framework strategies. The aggregated model is then distributed back to clients for improved performance on robotics manipulation tasks.