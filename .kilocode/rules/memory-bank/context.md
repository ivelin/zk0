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
**Status**: Diagnosed/Plan ready - static SuperExec flwr-clientapp "Invalid token" PERMISSION_DENIED; fix SuperNode dynamic spawn (--isolation process + executor-image=zk0:latest)

### Test Results (outputs/2025-12-15_21-42-03)
✅ **Server**: r0 eval loss=0.395, r1=0.399 (stable; 5 unseen SO-101 datasets, 256 ex/dataset)
❌ **Clients**: 0/2 fits r1 (RunNotRunningException → Invalid token PullClientAppInputs)
⚠️ HF DNS retry ok; heartbeat post-stop non-blocking

### Recent Progress
```
✅ 3-node logs/files analysis (server evals good, client token fail)
✅ Todo plan: dynamic SuperExec, HF cache, verify 3+ rounds
✅ Memory-bank queued
```

### Next Steps
**Finalize 3-node fix → v0.7.1**
- docker/docker-compose.client.yml: Remove zk0-client static; SuperNode node-config executor-image
- Test: zk0bot server/client + flwr local-deployment --tiny (fits succeed, losses ↓)
- HF offline cache docker/Dockerfile.zk0
- pytest cov 35%+, git tag/push

**Last Updated**: 2025-12-16