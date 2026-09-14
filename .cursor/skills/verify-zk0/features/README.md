# zk0.bot verification map

This directory is the maintained source for verifying visitor-facing behavior of the zk0.bot Jekyll site (`website/` plus docs copied at serve time). Read this index before driving, then use the matching feature file as the recipe.

## Ownership

verify-zk0 owns: zk0.bot site + claimed visitor paths on that host (Jekyll under `website/`, docs pages, installer/docs links that live in this repo).

It does not own: Bootstrap invite/MCP Accept flows, pirin.ai marketing routes, or live Flower SuperLink / SuperNode sessions.

## Baseline preconditions

- Launch via `control-zk0 launch` (isolated `127.0.0.1:$PORT`, dest `/tmp/zk0-verify-$RUN_ID/site`).
- Export `ZK0_VERIFY_RUN_ID` from launch output.
- Run `control-zk0 doctor` and require `Welcome to zk0`, Community Buzz, and `/docs/INSTALLATION.html` 200.
- Never drive `http://127.0.0.1:4000` or `https://zk0.bot` unless doctor says this run owns that listen port.
- Do not start Flower, SuperLink, SuperNode, Spark, or `zk0bot.sh`.

## Driving conventions

- Start every recipe from the baseline unless its preconditions say otherwise.
- Prefer routes and visible link text from `website/index.md` and `website/_includes/footer.html` over CSS position.
- Treat every command as literal. Keep quoted strings and flags unchanged.
- Run HTTP actions through `control-zk0 fetch` / `assert` / `drive`.
- Optional browser/CDP screenshots are a second view, not a substitute for the curl action.
- Cleanup must not remove proof artifacts.

## Proof and skip reporting

- Capture the user action and the resulting state, not only the final screen.
- HTTP proof includes status, headers, and body (or a named excerpt) plus a `PROOF.md`.
- Linked asset proof includes Content-Type (image/png, text/css, application/pdf).
- Mutation proof does not apply to these static pages; re-GET the same path as the second view.
- Record the feature ID and entry point with every artifact.
- Report an unreachable path with the attempted URL and the unmet precondition.
- Do not report a skipped entry point as verified through a different path.
- If `GET /get-zk0bot.sh` is 404 because `get-zk0bot.sh` is absent from this checkout, say so. Do not invent an installer.

## Feature entry contract

Each feature file starts with an H1 title and one paragraph describing the user-visible behavior. It then uses exactly four H2 sections in this order.

1. `Sub-features` lists short IDs with one line for each behavior.
2. `How to get to it (user POV)` lists every user entry point.
3. `Driving it with control-zk0` starts with `Preconditions:` and uses labeled bullets that pair each user action with an exact command and observable result.
4. `Gotchas` lists traps that can waste or invalidate a verification run.

Keep implementation details out of the map. Name only user paths, stable handles, required state, commands, and observable proof.

## Features

- [Home and Community Buzz](./home.md) covers the hero, Discover zk0, Get Started links, concept image, white paper, and tweet embeds.
- [Installation docs](./docs-installation.md) covers `/docs/INSTALLATION.html` from home and footer.
- [Node Operators docs](./docs-node-operators.md) covers `/docs/NODE-OPERATORS.html` from home and footer.
- [get-zk0bot installer path](./get-zk0bot.md) covers the installer curl documented on the Node Operators page and optional `GET /get-zk0bot.sh`.
- [Docs footer nav](./docs-nav.md) covers RUNNING, ARCHITECTURE, CONTRIBUTING, and LICENSE from the site footer.
