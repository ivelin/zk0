# Integration Requirements

**Created**: 2025-09-06
**Last Updated**: 2025-09-06
**Version**: 1.0.0
**Author**: Kilo Code

## SmolVLA + Flower Integration
- **Framework Compatibility**: Flower 1.20.0 with Ray 2.31.0
- **Dataset Format**: Flower Datasets for partitioning
- **Model Loading**: Direct integration with LeRobot SmolVLA models
- **Federated Dataset**: FederatedLeRobotDataset for distributed data

## Key Integration Points
1. **Client Implementation**: Extend NumPyClient for SmolVLA
2. **Server Strategy**: Use FedAvg or custom strategies
3. **Data Partitioning**: LeRobotDatasetPartitioner for episode-based splitting
4. **Model Aggregation**: Flower's parameter aggregation mechanisms