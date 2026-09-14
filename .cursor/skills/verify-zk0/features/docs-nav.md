# Docs footer nav

The site footer is the persistent navigation for visitors who want Running, Architecture, Contributing, License, Installation, Node Operators, Home, or GitHub after they leave the hero.

## Sub-features

- `nav-footer` renders `footer.site-footer` with the Navigation list from `website/_includes/footer.html`.
- `nav-running` serves `/docs/RUNNING.html` (H1 "Running the Project").
- `nav-architecture` serves `/docs/ARCHITECTURE.html` (H1 "Architecture").
- `nav-contributing` serves `/CONTRIBUTING.html`.
- `nav-license` serves `/LICENSE` (copied static file).

## How to get to it (user POV)

- Scroll to the footer on home or any custom-layout page.
- Choose **Running**, **Architecture**, **Contribute**, **License**, **Installation**, **Node Operators**, **Home**, or **GitHub**.
- Open the paths directly: `{URL}/docs/RUNNING.html`, `{URL}/docs/ARCHITECTURE.html`, `{URL}/CONTRIBUTING.html`, `{URL}/LICENSE`.

## Driving it with control-zk0

Preconditions:

- `control-zk0 doctor` is green.
- Home HTML has been fetched or you start from `{URL}/` and scroll to the footer.

- **See footer links on home.** `control-zk0 assert / --contains "Navigation" --contains 'href="/docs/RUNNING.html"' --contains 'href="/docs/ARCHITECTURE.html"' --contains 'href="/CONTRIBUTING.html"' --contains 'href="/LICENSE"' --contains 'href="https://github.com/ivelin/zk0"'`.
- **Open Running.** `control-zk0 fetch /docs/RUNNING.html --out .cursor/skills/verify-zk0/artifacts/$ZK0_VERIFY_RUN_ID/docs-nav/running`. Status 200. Body contains `Running the Project` and `train-fl-simulation.sh` as documentation.
- **Open Architecture.** `control-zk0 fetch /docs/ARCHITECTURE.html --out .cursor/skills/verify-zk0/artifacts/$ZK0_VERIFY_RUN_ID/docs-nav/architecture`. Status 200. Body contains `Architecture` and `Flower`.
- **Open Contributing.** `control-zk0 fetch /CONTRIBUTING.html --out .cursor/skills/verify-zk0/artifacts/$ZK0_VERIFY_RUN_ID/docs-nav/contributing`. Status 200. Body contains `Contributing to Federated Learning for Robotics AI`.
- **Open License.** `control-zk0 fetch /LICENSE --out .cursor/skills/verify-zk0/artifacts/$ZK0_VERIFY_RUN_ID/docs-nav/license`. Status 200. Body contains the project license text (Apache header from root `LICENSE`).
- **Second view.** Re-GET `/docs/RUNNING.html` and confirm 200 + `Running the Project`.
- **Optional browser view.** From `{URL}/`, click footer **Running**, then **Architecture**. Each click changes the path and H1 as above.

## Gotchas

- Architecture is a long page with mermaid source. Mermaid.js is loaded from `cdn.jsdelivr.net` in `website/_includes/head.html`. Curl proof is the mermaid fence / text, not a rendered SVG. Empty diagrams in a no-JS screenshot are not a failed page.
- Footer Discord is `https://discord.gg/dhMnEne7RP`. That is an off-host link; do not treat Discord as in-scope.
- `/LICENSE` has no `.html` suffix. `/LICENSE.html` is the wrong path.
- RUNNING documents `flwr run` and Docker flags. Do not execute them for this feature.
