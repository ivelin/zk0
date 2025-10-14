# System Context

**Project**: zk0 - Federated Learning with SmolVLA on SO-100 Datasets

**Latest Update (2025-10-14)**: Training run comparison analysis completed. Identified early stopping configuration as root cause of divergence in 500-round run. Tiny training validation successful. Memory bank and documentation updated to align with latest code and configuration. Project version updated to 0.1.20.

**Consolidated Metrics Implementation**:
- **Server Eval Files**: round_X_server_eval.json contains aggregated_client_metrics and individual_client_metrics
- **Client Identification**: Each client metric includes client_id and dataset_name for traceability
- **Aggregated Metrics**: avg_client_loss, std_client_loss, avg_client_proximal_loss, avg_client_grad_norm, num_clients, param_update_norm
- **Individual Metrics**: Per-client loss, fedprox_loss, grad_norm, param_hash, dataset_name
- **Benefits**: Single source of truth for federated learning metrics, easier remote client analysis

**Current Technical Status**:
- **FedProx Implementation**: Proximal loss correctly integrated before backprop with mu/2 formula
- **Server-Side Evaluation**: Policy loss as primary metric for SmolVLA flow-matching objective
- **Early Stopping**: Configurable server-side early stopping (default patience=10 rounds)
- **Parameter Safety**: Critical fixes ensure valid parameters always returned to prevent Flower crashes
- **Dataset Configuration**: 4 validated clients with diverse SO-100/SO-101 tasks
- **Enhanced Security**: Bidirectional SHA256 parameter validation between client and server
- **Consolidated Metrics**: Unified server evaluation files with aggregated and individual client metrics
- **Dynamic Learning Rate**: Optional adaptive learning rate adjustment based on evaluation trends

**Key Recent Fixes**:
- Client parameter type handling for Flower compatibility
- Server eval_mode passing for proper full/quick evaluation modes
- Full evaluation episode limits using dataset.episodes count
- Early stopping parameter safety (no None returns)
- Aggregate fit parameter safety for edge cases
- Enhanced security with bidirectional SHA256 parameter validation
- Consolidated metrics implementation for unified reporting
- Dynamic learning rate adjustment capability

**Federated Learning Experiment Results Table**

| Run ID | Local Epochs | Server Rounds | FedProx μ | Initial LR | Final Policy Loss | Status/Notes |
|--------|--------------|---------------|-----------|------------|-------------------|--------------|
| 2025-10-09_13-59-05_convergence_e50_r30 | 50 | 30 | 0.01 | 0.0005 | 0.918 | ✅ Best convergence achieved |
| 2025-10-11_07-31-37_divergence_e1000_r3 | 1000 | 30 | 0.01 | 0.0005 | 1.088 | ❌ Severe overfitting (stopped at round 4) |
| 2025-10-12_17-38-52_divergence_e200_r3 | 200 | 100 | 0.01 | 0.0005 | 0.570 | ❌ Divergence observed (stopped at round 3) |
| 2025-10-14_00-17-44_e50_r500 | 50 | 500 | 0.01 | 0.0005 | N/A | ❌ Early stopping triggered (round 16) due to aggressive patience=10 |

**Performance Insights**:
- **Optimal Range**: 50 local epochs per round for stable convergence
- **FedProx μ Impact**: 0.01 provides moderate regularization for heterogeneous SO-100 data
- **Convergence Target**: Policy Loss <0.4 considered good performance
- **Early Stopping**: Default patience=10 appropriate for 500-round experiments; prevents wasted computation
- **Evaluation Scope**: eval_batches=0 (full) preferred for proper convergence monitoring
- **Current Configuration**: 4 clients, 500 rounds, early stopping enabled, consolidated metrics
- **Key Lessons**:
  - Excessive local training (1000+ epochs) causes overfitting and FL divergence
  - Early stopping with full evaluation provides stable training termination
  - Core FL pipeline is stable; divergence typically due to configuration issues
  - Parameter validation prevents corrupted training from affecting global model