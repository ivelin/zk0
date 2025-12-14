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