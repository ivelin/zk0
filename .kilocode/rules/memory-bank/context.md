# System Context

**Project**: zk0 - Federated Learning with SmolVLA on SO-100 Datasets

**Latest Update (2025-11-06)**: ✅ **v0.5.0 SEO Quick Wins Implementation** Enhanced zk0.bot Jekyll site discoverability with auto-generated sitemaps, Open Graph/Twitter Cards, schema markup, front matter optimizations, and internal linking. Jekyll build verified with 36.73% test coverage maintained.

**Directory Structure Audit (2025-10-29)**: ✅ Audited workspace against memory bank and docs/ARCHITECTURE.md. Memory bank architecture.md was partially outdated (missing recent docs subfiles, src modules like logger.py/push_to_hf.py, expanded tests); updated to full structure matching v0.3.11 workspace state (version 1.0.6). Docs/ARCHITECTURE.md remains current.

**Recent Updates Summary (v0.3.13)**:
- CI Docker overhaul: Switched to containerized testing with project's Dockerfile for isolation, disabled caching to prevent space issues, updated documentation with CI/CD pipeline section.
- CI workflow consolidation with single matrix job for cleaner testing, lerobot CI fixes, Python 3.10 standardization.
- Enhanced security with bidirectional SHA256 parameter validation.
- Consolidated metrics (aggregated + individual client metrics in server eval files).
- Dynamic LR/MU scheduling with warm restarts, adaptive boosts, and spike detection.
- Code refactoring for modularity (70% reduction in aggregate_fit method size).
- Conditional HF push logic to avoid incomplete model uploads.
- Advanced LR/MU scheduling enhancements with warm restarts, per-client adaptive LR boosts, dynamic mu adjustment, and spike detection. New scheduler types (cosine_warm_restarts, reduce_on_plateau), configurable adaptive parameters, and comprehensive validation. Targets <0.15 server policy loss with 100% client engagement through heterogeneity-aware scheduling.

**Sprint v0.4.1 Progress (Production Deployment)**:
- ✅ Docker Infrastructure: Custom zk0 image (Dockerfile.zk0), compose files for server/client, local testing completed.
- ✅ Runtime Modes: Simulation via train-fl-simulation.sh; Prod via zk0bot CLI subcommands.
- ✅ Code Refactoring: Mode guards in server/client_app.py, utils moved to src/core/, <500 LOC enforced.
- ✅ zk0bot CLI: Bash-based tool implemented with installer, commands for server/client management.
- ✅ Documentation: NODE-OPERATORS.md created, GitHub issue template, README/ARCHITECTURE/DEVELOPMENT updated.
- ✅ Testing: Full pytest in Docker passing at 36.67% coverage; prod mocks included.
- ⏳ Pending: zk0bot cross-OS testing, E2E tests, security scan, performance benchmarks, GHCR push, PR merge/tag v0.4.1.

**Consolidated Metrics Implementation**:
- **Server Eval Files**: round_X_server_eval.json contains aggregated_client_metrics and individual_client_metrics
- **Client Identification**: Each client metric includes client_id and dataset_name for traceability
- **Aggregated Metrics**: avg_client_loss, std_client_loss, avg_client_proximal_loss, avg_client_grad_norm, num_clients, param_update_norm
- **Individual Metrics**: Per-client loss, fedprox_loss, grad_norm, param_hash, dataset_name
- **Benefits**: Single source of truth for federated learning metrics, easier remote client analysis

**Current Technical Status**:
- **FedProx Implementation**: Proximal loss correctly integrated before backprop with mu/2 formula
- **Server-Side Evaluation**: Policy loss as primary metric for SmolVLA flow-matching objective
- **Parameter Safety**: Critical fixes ensure valid parameters always returned to prevent Flower crashes
- **Dataset Configuration**: 12 validated clients with diverse SO-100/SO-101 tasks (expanded from 4)
- **Enhanced Security**: Bidirectional SHA256 parameter validation between client and server
- **Consolidated Metrics**: Unified server evaluation files with aggregated and individual client metrics
- **Dynamic Learning Rate**: Advanced LR/MU scheduling with warm restarts, adaptive boosts, dynamic mu, and spike detection
- **Production Readiness**: Docker Compose for multi-node FL, zk0bot for node operators, privacy via UUID-anonymized metrics.
- **WandB Integration**: Model cards now include direct links to WandB experiment runs when WandB is enabled


**Performance Insights**:
- Optimal configurations achieve stable convergence with policy loss <0.5, using LR=1e-4 and dynamic scheduling for 89% loss reduction and 100% client engagement.
- Key lessons: Lower initial LR preferred for stability; adaptive features handle stalls effectively; focus on dataset balance for heterogeneity.

For core architecture details, see [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md).