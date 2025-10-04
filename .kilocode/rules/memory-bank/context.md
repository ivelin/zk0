# System Context

**Project**: zk0 - Federated Learning with SmolVLA on SO-100 Datasets

**Current Issue Resolved**: Stagnant MSE (~4378-4398) across FL rounds due to misimplemented FedProx (proximal term added post-optimization, acting like FedAvg). Fixed by integrating proximal loss before backprop in `src/task.py`'s `run_training_step`, ensuring cumulative global model improvements.

**Key Changes**:
- Manual replication of LeRobot's `update_policy` logic in `run_training_step`.
- Compute `total_loss = main_loss + (mu / 2) * ||w - w_global||^2` before backward.
- Preserves AMP, clipping, scheduler from LeRobot for compatibility.

**Validation**: Code linted (Ruff clean), imports successful. Test with 5 rounds to confirm progressive MSE decrease.

**Next**: Run full FL experiment (20 rounds) to verify convergence (<3000 MSE).