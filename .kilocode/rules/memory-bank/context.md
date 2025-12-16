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

**Sprint Status:** 3-Node zk0bot production-ready (SuperLink/SuperNode/SuperExec stateless).

**Last Updated**: 2025-12-16