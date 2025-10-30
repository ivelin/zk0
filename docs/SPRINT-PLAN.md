# Sprint Planning Document: zk0 v0.4.0 - Production Deployment Support

## Executive Summary
This sprint plan outlines the development of zk0 v0.4.0, a major release introducing production-ready deployments alongside existing local simulation capabilities. The focus is on enabling scalable, secure federated learning (FL) for SmolVLA models on SO-100/SO-101 datasets using Docker and Docker Compose, following Flower AI's best practices (e.g., [Quickstart with Docker Compose](https://flower.ai/docs/framework/docker/tutorial-quickstart-docker-compose.html), [Run Flower using Docker](https://flower.ai/docs/framework/docker/index.html), and [Deploy on Multiple Machines](https://flower.ai/docs/framework/docker/tutorial-deploy-on-multiple-machines.html)).

Key innovations:
- **Runtime Modes**: Local simulation (current), production server (SuperLink + ServerApp), production client (SuperNode + ClientApp).
- **Custom Docker Ecosystem**: A zk0-specific image combining LeRobot and Flower, tagged with project version (e.g., `ghcr.io/ivelin/zk0:v0.4.0`), pushed to GitHub Container Registry (GHCR).
- **Node Operator Onboarding**: Invitation-only via GitHub issues (new template) and Discord; detailed docs for private dataset integration.
- **Simulation vs. Production Differentiation**: Addressed in architecture (e.g., local/private datasets, UUID-anonymized metrics, server-assigned node IDs).
- **Code Hygiene**: Reorganization for maintainability; enforce <500 LOC per new file (added to memory-bank/tech.md).

**Sprint Goals**:
- Achieve production viability: Secure, multi-node FL with privacy-preserving client training.
- Maintain backward compatibility: Simulation mode unchanged.
- Community Focus: Enable node operators to contribute unique datasets while protecting privacy.
- Measurable Outcomes: 100% test coverage for new features (>80% overall), Docker images built/pushed, docs live on GitHub.

**Estimated Sprint Duration**: 2-3 weeks (10-15 working days), assuming 1-2 developers. Prioritize core Docker/Compose setup in Week 1, mode differentiation and docs in Week 2, testing/reorg in Week 3.

**Risks & Mitigations**:
- Docker Image Complexity: Mitigate by basing on official Flower images; test with multi-stage builds.
- Mode Conflicts: Use separate scripts to isolate logic; unit tests for each mode.
- Security: Enforce TLS (from Flower docs); validate private datasets don't leak.
- Scope Creep: Lock to listed features; defer advanced auth (e.g., full OAuth) to v0.5.

**Versioning**: Bump from 0.3.16 to 0.4.0 (minor for production support). Use semantic versioning; tag Docker images accordingly.

**Branch Strategy**: Create `feat-prod-deployment` from `main`; merge via PR after sprint.

## Research Insights
Deep research via Brave Search on Flower Docker best practices confirms:
- **Core Components**: Use `flwr/superlink` (fleet management), `flwr/serverapp` (FL server), `flwr/clientapp` (FL clients). zk0 custom image will extend `flwr/serverapp` with LeRobot/SmolVLA code.
- **Docker Compose**: Single `docker compose up` for quickstarts; separate `compose.yml` for server/SuperLink and clients/SuperNodes. Ports: 9091 (ServerAppIO API), 9092 (Fleet API), 9093 (Control API). Security handled via p2p network protocols (Tailscale/WebRTC) at IP level, allowing Flower to run in insecure mode.
- **Production Deployment**: Multi-machine via SSH/SCP (e.g., copy `server/compose.yml` to remotes); build custom images with `ADD` for zk0 code. Use `--insecure` for all modes since network-level security protects nodes.
- **Heterogeneity**: Supports device diversity (e.g., GPU clients); aligns with zk0's private datasets.
- **Best Practices**: Pre-built images for reproducibility; volumes for datasets/models; health checks in Compose. For zk0: Mount private dataset dirs at runtime; env vars for dataset IDs.
- **Gaps Filled**: zk0 needs custom entrypoints (e.g., `flower-server-app server:zk0.server_app` with zk0 params); integrate LeRobot via multi-stage Dockerfile (install deps, copy src).

Additional Considerations from zk0 Context (Memory Bank):
- **Simulation Mode**: Server assigns partitions via pyproject.toml; clients use shared SO-100 splits.
- **Production Mode**: Clients specify local/HF dataset at startup (e.g., `--dataset=local:/path/to/private` or `--dataset=hf:user/private-so100`); server assigns node_id dynamically (no dataset knowledge). Clients report `dataset_label_uuid` in metrics (e.g., `{"dataset": "private-so100-v1-uuid123", "loss": 0.5}`) for WandB logging (public aggregates only).
- **Other Differentiations**:
  - **Networking**: Simulation: Localhost/loopback. Prod: External IPs, TLS, firewall rules (e.g., expose 9091-9093).
  - **Persistence**: Simulation: Ephemeral. Prod: Volumes for models/checkpoints/datasets; backups via cron.
  - **Scaling**: Simulation: Fixed clients (e.g., 4). Prod: Dynamic via SuperNodes; auto-scale with Kubernetes later.
  - **Monitoring**: Both: WandB/Loguru. Prod: Add Prometheus/Grafana for node health.
  - **Auth**: Simulation: None. Prod: Invitation keys in env (e.g., `NODE_INVITE_KEY`); validate on connect.
  - **Eval**: Server uses fixed/expanded eval datasets (manual admin adds); community proposals via GitHub.
  - **Termination**: Simulation: Fixed rounds. Prod: Clients train per selection, restart for new datasets.
  - **Error Handling**: Prod: Retry logic for network flakes; simulation: Fast-fail.
- **Reorg Rules**: New files <500 LOC (e.g., split server_app.py further); add to memory-bank/tech.md.

## High-Level Architecture Updates
- **Runtime Modes** (single zk0bot CLI with subcommands for clarity):
  - **Simulation**: Via `train-fl-simulation.sh` (renamed from `train.sh`); uses Ray for local clients.
  - **Server-Prod**: Via `zk0bot server start` (single CLI tool handling production server startup with Docker Compose).
  - **Client-Prod**: Via `zk0bot client start --dataset repo/name` (single CLI tool handling production client startup with Docker Compose).
- **Docker Setup**:
  - **Custom Image**: `Dockerfile.zk0` extends `huggingface/lerobot-gpu:latest`; installs zk0 deps, copies src. Build: `docker build -t ghcr.io/ivelin/zk0:v0.4.0 -f Dockerfile.zk0 .`. Push: `docker push ghcr.io/ivelin/zk0:v0.4.0`.
  - **Compose Files**:
    - `docker-compose.server.yml`: SuperLink + ServerApp; env for zk0 config (pyproject.toml mount).
    - `docker-compose.client.yml`: SuperNode + ClientApp; volumes for private datasets.
    - Overlays: `docker-compose.tls.yml` for prod security.
  - **Entrypoints**: Use Python modules (e.g., `python -m flwr.server.app src.server_app:app`); pass zk0 params. zk0bot CLI wraps these for user-friendly execution.
- **Dataset Handling**:
  - Prod Clients: Startup env `DATASET_URI=<local:/path|hf:user/repo>`; load via LeRobot (private HF needs token env).
  - Metrics: Clients append `{"private_dataset": f"{task_name}-{uuid.uuid4()}", ...}` to fit/evaluate returns.
  - Server Eval: Expand pyproject.toml `[tool.zk0.datasets.eval]` with community proposals (e.g., add 2 placeholder SO-101 datasets).
- **Onboarding**:
  - **Node Operators Doc**: `docs/NODE-OPERATORS.md` – Steps: Apply via GitHub issue (template: setup details, dataset uniqueness, Discord handle); get invite key; install zk0bot (`curl -fsSL https://get.zk0.bot | bash`); run `zk0 client start --dataset yourrepo/yourdsname --invite-key=xyz`.
  - **GitHub Template**: `.github/ISSUE_TEMPLATE/node-operator-application.md` – Fields: Hardware (GPU/RAM), Dataset description (unique tasks, size), Experience, Discord.
  - **Server Deployment Section**: Explain SuperLink setup, port exposure, admin monitoring with `zk0 server start`.

## Detailed Todo List
This sprint is multi-step; tracking via checklist for execution.

- [x] Research Flower Docker best practices (completed via MCP tool)
- [x] Branch Management
- [x] Create branch `feat-prod-deployment` from `main`
- [x] Bump version to 0.4.0 in pyproject.toml
- [x] Initial commit: "feat: init prod deployment sprint"
- [x] Docker Infrastructure
- [x] Create Dockerfile.zk0 (extend flwr/serverapp, add LeRobot/zk0 deps, copy src, <500 LOC)
- [x] Build/test custom image locally (docker build -t zk0:dev)
- [ ] Push to GHCR (setup GitHub actions for auto-build/push on tag)
- [x] Create docker-compose.server.yml (SuperLink + ServerApp, ports 9091-9093, zk0 entrypoint)
- [x] Create docker-compose.client.yml (SuperNode + ClientApp, dataset volume, env for URI/invite key)
- [x] Test server prod: docker compose -f server.yml up; verify APIs on localhost:9091-9093
- [x] Test client prod: docker compose -f client.yml up --env dataset-uri=local:/fake/path
- [ ] Runtime Mode Implementation
- [ ] Rename train.sh to train-fl-simulation.sh and update for simulation-only use
- [ ] Integrate server/client startup into zk0bot CLI (subcommands like zk0bot server start, zk0bot client start --dataset URI --invite-key key; wraps Docker Compose internally, no separate scripts needed)
- [ ] Refactor server_app.py: Mode guards for simulation (Ray) vs prod (external connects); <500 LOC, split if needed (e.g., server_prod.py)
- [ ] Refactor client_app.py: Prod mode loads dataset from URI (LeRobot HF/local loader); append UUID metrics; <500 LOC
- [ ] Update configs/datasets.py: Prod client ignores partitions; server expands eval list (add 2 placeholder SO-101 datasets)
- [ ] Implement invite key validation (simple env check on connect; defer advanced to v0.5)
- [ ] Update wandb_utils.py: Prefix metrics with node_id + private_dataset_uuid (anonymized)
- [ ] Node Operator CLI Tool (zk0bot)
- [ ] Research and design zk0bot CLI (inspired by RocketPool rocketpool CLI and Opencode installer: bash one-step install via curl -fsSL https://get.zk0.bot | bash, commands like zk0 server start/stop/log, zk0 client start --dataset repo/name --invite-key key, zk0 status, zk0 config)
- [ ] Implement zk0bot as bash-based CLI (single binary or script bundle, hosted on GitHub releases; installer downloads and sets up PATH; integrates Docker/Compose for server/client control)
- [ ] Create installer script (get.zk0.bot -> curl -fsSL https://get.zk0.bot | bash, detects OS, downloads binary, adds to PATH, verifies Docker/Compose)
- [ ] Add zk0bot commands: server start/stop/log/status, client start/stop/log --dataset URI --invite-key, config (env vars, dataset setup)
- [ ] Add zk0bot installation guide to docs/NODE-OPERATORS.md (one-step curl install, quick start examples)
- [ ] Test zk0bot: Verify installer on Ubuntu/Mac/Windows (WSL), commands integrate with startup scripts (e.g., zk0 server start --invite-key=xyz)
- [ ] Documentation & Onboarding
- [ ] Create docs/NODE-OPERATORS.md: Steps for apply (GitHub issue), client setup with zk0bot, server overview, Discord link
- [ ] Create .github/ISSUE_TEMPLATE/node-operator-application.md: Form fields (hardware, dataset info, etc.)
- [ ] Update README.md: Add production section with zk0bot examples, startup scripts
- [ ] Update docs/ARCHITECTURE.md: Add prod mode diagram, differences table, CLI integration
- [ ] Update docs/DEVELOPMENT.md: Docker testing guidelines, <500 LOC rule
- [ ] Code Reorganization
- [ ] Audit src/: Split large files (e.g., server_app.py → server_core.py + server_prod.py); ensure <500 LOC new files
- [ ] Move utils to src/zk0/core/ for modularity
- [ ] Update tests: Add prod mode integration tests (mock external connects); run in Docker
- [ ] Enforce <500 LOC rule: Add to memory-bank/tech.md ("New source files must be <500 lines for maintainability")
- [ ] Testing & Validation
- [ ] Run full pytest in Docker (coverage >80%, include prod mocks)
- [ ] E2E Test: Simulation (unchanged), server-prod + 2 client-prod (fake datasets)
- [ ] Security Scan: docker scan zk0:dev; verify TLS optional
- [ ] Performance: Benchmark prod vs sim (e.g., round time with external "nodes")
- [ ] Finalization
- [ ] Update memory-bank/context.md: v0.4.0 progress, prod features
- [ ] PR: Merge feat-prod-deployment to main; tag v0.4.0
- [ ] Push Docker images; update docs with GHCR links

## Next Steps for Execution
1. **Review & Approval**: Confirm plan aligns with vision (e.g., invitation model, dataset privacy). Suggest changes?
2. **Mode Switch**: Once approved, switch to code mode for implementation.
3. **Execution**: Follow todo list sequentially; update progress regularly.

**Mermaid Diagram: Mode Flow**
```mermaid
graph TD
    A[Start zk0] --> B{Mode?}
    B -->|simulation| C[Local Ray Clients<br/>Shared SO-100 Partitions]
    B -->|server-prod| D[Docker Compose Server<br/>SuperLink + ServerApp<br/>Fixed Eval Datasets]
    B -->|client-prod| E[Docker Compose Client<br/>SuperNode + ClientApp<br/>Private Dataset Mount]
    C --> F[FL Rounds: Server Assigns<br/>Partitions via pyproject.toml]
    D --> G[Accept Connections<br/>Assign node_id Dynamically<br/>Aggregate Updates]
    E --> H[Connect to Server<br/>Startup Dataset URI<br/>Report UUID-Labeled Metrics]
    F --> I[WandB: Aggregated Public Metrics]
    G --> I
    H --> I
    I --> J[End Session<br/>Restart for New Dataset]