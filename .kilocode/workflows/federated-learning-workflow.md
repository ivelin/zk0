# Federated Learning Workflow

## Overview
This workflow guides the setup and execution of a federated learning round using Flower framework with SmolVLA models and SO-100 datasets.

## Prerequisites
- Conda environment "zk0" activated
- SmolVLA model and SO-100 dataset access
- Flower framework installed
- At least one client configured
- Server hardware with sufficient resources

## Steps

### 1. Server Initialization
- Configure ServerApp with FedAvg or FedProx strategy
- Set number of rounds and client participation fraction
- Initialize global SmolVLA model parameters

### 2. Client Registration
- Accept client connections via secure channels
- Validate client authentication and data privacy compliance
- Distribute initial global model to connected clients

### 3. Federated Round Execution
- Orchestrate training rounds across clients
- Clients train locally on partitioned SO-100 datasets
- Collect model updates from participating clients

### 4. Parameter Aggregation
- Aggregate client updates using selected strategy
- Update global model parameters
- Broadcast improved model to all clients

### 5. Performance Monitoring
- Track convergence metrics and training progress
- Monitor communication efficiency and resource usage
- Log round completion and client participation

### 6. Round Completion
- Validate aggregated model performance
- Prepare for next round or terminate federation
- Document results and lessons learned

## Checkpoints
- [ ] Server successfully initialized and listening
- [ ] Minimum number of clients connected
- [ ] All clients received initial model
- [ ] Training rounds completed without errors
- [ ] Global model updated and distributed
- [ ] Performance metrics logged and reviewed