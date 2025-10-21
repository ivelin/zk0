# System Context

**Project**: zk0 - Federated Learning with SmolVLA on SO-100 Datasets

**Latest Update (2025-10-21)**: ✅ **Server evaluation refactoring completed!** Successfully broke down bloated `_server_evaluate` method into 6 atomic, testable functions: `should_skip_evaluation`, `prepare_evaluation_model`, `process_evaluation_metrics`, `log_evaluation_to_wandb`, `save_evaluation_results`, and `generate_evaluation_charts`. All tests pass (46/46), functionality preserved. Improved code maintainability and testability.

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
- Advanced LR/MU scheduling with warm restarts, adaptive boosts, dynamic mu, and spike detection

**Federated Learning Experiment Results Table**

| Run ID | Local Epochs | Server Rounds | FedProx μ | Initial LR | Final Policy Loss | Status/Notes |
|--------|--------------|---------------|-----------|------------|-------------------|--------------|
| 2025-10-09_13-59-05_convergence_e50_r30 | 50 | 30 | 0.01 | 0.0005 | 0.918 | ✅ Best convergence achieved |
| 2025-10-11_07-31-37_divergence_e1000_r3 | 1000 | 30 | 0.01 | 0.0005 | 1.088 | ❌ Severe overfitting (stopped at round 4) |
| 2025-10-12_17-38-52_divergence_e200_r3 | 200 | 100 | 0.01 | 0.0005 | 0.570 | ❌ Divergence observed (stopped at round 3) |
| 2025-10-14_00-17-44_e50_r500 | 50 | 500 | 0.01 | 0.0005 | N/A | ❌ Early stopping triggered (round 16) due to aggressive patience=10 |
| 2025-10-17_08-02-19_convergence_dynamicDecay_e20_r50 | 20 | 50 | 0.01 | 0.0005 | 0.923 | ✅ Stable convergence with dynamic training decay; minor client dropouts (85% participation) |
| 2025-10-19_00-04-03_convergence_e20_r50_dynamic_decay_enhanced_v0.2.7_lr_5e-4 | 20 | 50 | 0.01 (dynamic) | 0.0005 | 0.997 | ✅ Volatile; high initial loss (9.17), oscillates ~1.0; std=1.82 |
| 2025-10-19_09-05-34_convergence_e20_r50_dynamic_decay_enhanced_v0.2.7_lr_1e-4 | 20 | 50 | 0.01 (dynamic) | 0.0001 | 0.532 | ✅ Stable; smooth to 0.53; 47% better final, std=0.11 |

**Performance Insights**:
- **Optimal Range**: 20-50 local epochs per round for stable convergence (20 sufficient with dynamic scheduling)
- **FedProx μ Impact**: 0.01 moderate for SO-100; dynamic decay reduces to ~0.001 by r50
- **Convergence Target**: Policy Loss <0.4 good; <0.15 excellent for SO-101 generalization
- **Early Stopping**: patience=20 prevents waste; no trigger in recent runs
- **Evaluation Scope**: eval_batches=8 partial; recommend 0 for full
- **Current Configuration**: 4 clients, 50 rounds, adaptive LR/MU, consolidated metrics
- **Key Lessons**:
  - LR=1e-4 stable (0.53 final, low std); 5e-4 volatile (1.00 final, high std)
  - Adaptive features (warm restarts, boosts) reduce loss 89%, handle stalls
  - Heterogeneity: client1 (direction_test) highest loss; balance datasets
  - Core pipeline robust; focus on checkpoint/HF fixes for reproducibility