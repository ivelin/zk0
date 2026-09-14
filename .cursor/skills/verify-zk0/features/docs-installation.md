# Installation docs

Installation is the visitor docs page that tells developers how to create the conda `zk0` env, install CUDA PyTorch / LeRobot / the project, and where training outputs land. It is a rendered Markdown page, not a live installer.

## Sub-features

- `install-open` serves `/docs/INSTALLATION.html` with H1 Installation.
- `install-from-home` is reached from home Get Started "For Developers".
- `install-from-footer` is reached from footer Navigation "Installation".
- `install-cross-links` points at Architecture, Node Operators, and Running Simulations.

## How to get to it (user POV)

- On home, choose **For Developers**.
- In the footer, choose **Installation**.
- Open `{URL}/docs/INSTALLATION.html` or `https://zk0.bot/docs/INSTALLATION.html`.
- From other docs pages, choose the "Installation Guide" / "INSTALLATION" cross-link.

## Driving it with control-zk0

Preconditions:

- `control-zk0 doctor` is green.
- Home has already been fetched once this run, or you accept opening Installation as a deep link.

- **Follow home CTA.** Run `control-zk0 assert / --contains 'href="/docs/INSTALLATION.html"' --contains "For Developers"`. Then `control-zk0 fetch /docs/INSTALLATION.html --out .cursor/skills/verify-zk0/artifacts/$ZK0_VERIFY_RUN_ID/docs-installation/page`.
- **Rendered title.** `page.status` is `200`. `page.body` contains `Installation` as the H1 text and the front-matter title string `zk0 Installation: Set Up Federated Learning for SmolVLA Robotics` (page `<title>` / SEO).
- **Developer steps.** `page.body` contains `conda create -n zk0 python=3.10`, `pip install lerobot[smolvla]==0.3.3`, and `pip install -e .`.
- **Cross-links.** `page.body` contains links toward Architecture, Node Operators, and Running (relative `ARCHITECTURE` / `NODE-OPERATORS` / `RUNNING` as Jekyll rendered them).
- **Second view.** Re-run `control-zk0 assert /docs/INSTALLATION.html --contains "Installation" --contains "conda create -n zk0"`. Still 200.
- **Optional browser view.** Click **For Developers** on `{URL}/`. The location becomes `/docs/INSTALLATION.html` and the H1 reads Installation.

## Gotchas

- Jekyll emits `/docs/INSTALLATION.html`, not `/docs/INSTALLATION`. A missing `.html` may 404 on the local webrick server.
- This page documents conda/Docker and later mentions Flower simulation scripts. Reading those sentences is not permission to start SuperLink or `./train-fl-simulation.sh`.
- `website/docs/` is a copy made at launch. If you edit root `docs/INSTALLATION.md` after launch without restarting, the served page is stale.
