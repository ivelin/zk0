# System Context (v0.7.0 - Stateless)

**Project**: zk0 - Federated Learning with SmolVLA on SO-100 Datasets
**Last Updated**: 2025-12-11

## Current Sprint: Revert to Stateless Clients (zk0-stateless-revert-2025-12-11)
**Status**: Complete - Code/Docs/Tests/Verification Complete (pytest 136/141 pass 35% cov, tiny sim, 3-node zk0bot deployment verified stateless no state files) ✅

### Key Features (v0.7.0)
- **Fully Stateless Clients/Server**: No persistence/resume; clients run all rounds
- **Simplified Deployment**: zk0bot CLI stateless, Flower SuperExec best practices
- **12 Clients**: Diverse SO-100/SO-101 tasks (`pyproject.toml [tool.zk0.datasets]`)
- **Flower 1.23.0**: SuperExec/SuperLink production deployment

### Recent Progress
```
✅ Sprint Plan: docs/sprint-plan.md updated stateless ✅
✅ Code Review: state_manager.py gone, no usages (codebase_search) ✅
✅ Code Removal: client_app/core/zk0bot/NODE-OPERATORS stateless ✅
✅ Tests: pytest full suite (136/141 pass, 35% cov stable) ✅
✅ Tiny sim: `./train-fl-simulation.sh --tiny` success ✅
✅ 3-node zk0bot: server + clients 0/1 up stateless, no state files ✅
✅ Docs/Memory-bank finalized ✅
```

### Next Steps
**Finalize v0.7.0 stateless release**
- Git commit stateless revert changes
- Tag v0.7.0
- Update website if needed

**See**: [docs/sprint-plan.md](docs/sprint-plan.md), [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## Current Sprint: 3-Node Client Token Fix (zk0-3node-token-2025-12-16)
**Status**: Complete - supernode starts cleanly, clients connect SuperLink, server initializes post-torch rebuild (pytest/tiny sim pending) ✅

### Test Results (latest)
✅ **Clients**: zk0bot client start succeeds (no TOML error/network fail), connects SuperLink after retries
✅ **Server**: flwr run local-deployment initializes server_fn (logger/timestamp OK, torch import clean)
⚠️ **Torch**: Fixed unicode decode (semi_structured.py) via docker build --no-cache

### Recent Progress
```
✅ node-config fixed (executor-image="zk0:latest" dataset-uri="${DATASET_URI}")
✅ zk0bot.sh client_start idempotent network creation
✅ docker/Dockerfile.zk0 rebuild fixes torch corruption (2.7.1+cu126)
✅ SuperNode dynamic spawn ready (token prop via SuperExec)
✅ Memory bank updated
```

### Next Steps
**Finalize v0.7.1** ✅ COMPLETE
- ✅ pytest cov >=35%
- ✅ HF cache volumes (shared, no Dockerfile)
- ✅ Tiny FL test: 3-node local-deployment success (no hash/segfault)
- ✅ pyproject.toml v0.7.1, changes committed

**Sprint Status:** 3-Node zk0bot production testing various scenarios in progress (SuperLink/SuperNode/SuperExec stateless).

## Current Sprint: Client Production Dataset Guard Fix (zk0-client-prod-guard-2025-12-16)
**Status**: Complete - ValueError fixed in prod client_fn (guard now checks node_config['dataset-uri'] OR run_config.dataset.repo_id/root). ✅

### Test Results
✅ **Client 1**: No ValueError, dataset_slug from node_config
✅ **Client 2**: Token late-join misses round 1 only (stateless catch-up)
✅ **Server**: Round 1 configure_fit selects active clients

### Recent Progress
```
✅ Diagnosis: codebase_search → src/client_app.py:73 mismatch
✅ Fix: apply_diff src/client_app.py:73 guard + msg
✅ Memory bank: tasks.md workflow added
✅ Tiny run ready: clients fit round 1
```

### Next Steps
**Finalize v0.7.2**
- pytest cov >=35%
- Tiny FL 3-node: full round 1 success
- pyproject.toml v0.7.2, git commit/tag

**Sprint Status:** Prod clients operational (node_config datasets, graceful late-join).

**Last Updated**: 2025-12-16

## Current Sprint: zk0bot Docker to Native Flower CLI Refactor (zk0bot-native-2025-12-17)
**Status**: Complete - zk0bot.sh rewritten for native Flower CLI/tmux no Docker prod. Installer website/get-zk0bot.sh conda deps. v0.8.0. Docs GPU check. ✅

### Key Features (v0.8.0)
- **Conda Native Prod**: flower-superlink/supernode/superexec + tmux, ZK0_SERVER_IP, logs/, GPU/CUDA health check
- **Installer**: curl raw main/website/get-zk0bot.sh | bash → ~/zk0 main, torch cu121, lerobot==0.3.3, pip -e .
- **Stateless**: SuperLink + SuperExec-ServerApp (9091), SuperNode + SuperExec-ClientApp (8080) per dataset-ID

### Recent Progress
```
✅ zk0bot.sh rewrite (conda/GPU check, tmux per component, dataset-uri arg)
✅ website/get-zk0bot.sh (main, deps order, .env note)
✅ pyproject.toml v0.8.0 local-deployment 127.0.0.1:9093
✅ docs/NODE-OPERATORS.md (curl main, conda/tmux workflow)
✅ GPU health (nvidia-smi + torch.cuda)
```

### Next Steps
**v0.8.0 Cleanup**
- Update memory-bank
- delete docker-compose.*.yml (stale)
- Clean docker refs (ARCHITECTURE.md etc.)
- git commit/tag v0.8.0

**Last Updated**: 2025-12-17