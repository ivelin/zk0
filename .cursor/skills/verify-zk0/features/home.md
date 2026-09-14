# Home and Community Buzz

The zk0.bot home page introduces the project (hero, concept diagram, Discover zk0, feature cards), points visitors at Installation and Node Operators, and shows Community Buzz as in-page Twitter embed markup.

## Sub-features

- `home-hero` renders "Welcome to zk0" and the intro about teaching robots without sharing family videos.
- `home-discover` shows Discover zk0, the white paper link, and the four feature cards (Flower, SmolVLA, SO-100, Production-Ready).
- `home-get-started` exposes "For Developers" → Installation and "For Node Operators" → Node Operators.
- `home-buzz` shows the Community Buzz heading and `twitter-tweet` blockquotes.
- `home-assets` serves the concept PNG, custom CSS, QR image, and white paper PDF the page links.

## How to get to it (user POV)

- Open `https://zk0.bot/` (production) or `{URL}/` on the verification instance.
- Choose the footer **Home** link from any docs page.
- Arrive from an external share of `https://zk0.bot`.

## Driving it with control-zk0

Preconditions:

- `control-zk0 doctor` reports this run owns `{URL}` and `GET /` contains Welcome to zk0.
- No Flower or SuperLink process is required or running for this proof.

- **Open home.** Run `control-zk0 drive home`. Exit 0. Artifacts land in `.cursor/skills/verify-zk0/artifacts/$RUN_ID/home/` with `home.body` containing `Welcome to zk0`, `Decentralized AI for the next generation of helpful robots`, `Discover zk0`, and `Community Buzz`.
- **Get Started links.** In `home.body`, `href="/docs/INSTALLATION.html"` appears with visible text `For Developers`, and `href="/docs/NODE-OPERATORS.html"` appears with `For Node Operators`.
- **White paper link.** `home.body` contains `href="/docs/zk0-white-paper-Dec-2025.pdf"`. `white-paper.status` is `200` and `white-paper.headers` includes `application/pdf`.
- **Concept image.** `home.body` contains `alt="zk0 Federated Learning Concept Diagram"`. `concept.png.status` is `200` and `concept.png.headers` is an image type.
- **Community Buzz markup.** `home.body` contains `class="social-posts"`, heading text `Community Buzz`, and `twitter-tweet` (including the HICAM / `@ivelini` photo post blockquote).
- **Second view.** `drive home` re-GETs `/` and still finds `Welcome to zk0` and `Community Buzz`.
- **Optional browser view.** Open `{URL}/` in a browser. Screenshot the hero and the Community Buzz heading. Twitter cards may stay unloaded without `platform.twitter.com`; that is a production-boundary gap, not a failed home render.

## Gotchas

- Root `index.md` is **not** the GitHub Pages home. `build-site.sh` / Actions build from `website/index.md`. Assert `Welcome to zk0`, not the root file's "Welcome to zk0: Collaborative AI for Humanoid Robots".
- `./build-site.sh --serve` listens on `:4000` and writes `website/_site`. Doctor must see `/tmp/zk0-verify-$RUN_ID/site` and this run's port.
- Curl does not execute `https://platform.twitter.com/widgets.js`. Do not fail home because embed iframes are empty; fail if the `twitter-tweet` markup or Community Buzz heading is missing.
- Headless Chrome `--screenshot` of `/` can hang after writing the PNG while `widgets.js` talks to Twitter. Kill that browser PID (not `killall chrome`); the PNG is already valid. Prefer `control-zk0 drive home` as the action proof.
- The QR image (`/assets/images/zk0-qr-code.png`) is in the Share block; `drive home` does not fetch it. Fetch it separately if you claim the Share block.
