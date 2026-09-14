---
name: verify-zk0
description: Drive the zk0.bot Jekyll visitor site (website/ plus copied docs pages) the way a user does. Use when proving a home, Community Buzz, docs, or installer-link change on zk0.bot — not Flower/SuperLink, Bootstrap invites, or pirin.ai routes.
---

# Verify zk0.bot

zk0.bot is a public Jekyll site for the [ivelin/zk0](https://github.com/ivelin/zk0) project. The visitor surface is the pages built from `website/` (home, custom layout, footer nav, assets) plus Markdown copied from root `docs/` at serve/build time. This skill owns that host and those claimed visitor paths only.

It does **not** own Bootstrap invite/MCP Accept flows, pirin.ai marketing routes, or live Flower SuperLink / SuperNode / SuperExec. `zk0bot.sh` and `./train-fl-simulation.sh` exist in this repo; do not start them from this skill.

Read `features/README.md` before driving. A proof that uses one convenient entry point is incomplete when the map lists others.

There is no Playwright or Cypress harness in this repo. The in-repo check the site authors already describe is curl against a local Jekyll listen (`localhost:4000` in `build-site.sh` / memory-bank). This skill ships `control-zk0` as that curl recipe, plus optional browser/CDP screenshots for visual proof.

## Launch

Needs Ruby 3.x and Bundler that can install `Gemfile.lock` (Jekyll `~> 4.3.2`, minima, jekyll-seo-tag, jekyll-sitemap). No site auth and no `.env` for visitor pages. Repo `.bundle/config` may point at `/home/ivelin/.gem`; the helper ignores that config and vendors gems under `.cursor/skills/verify-zk0/.bundle`.

Official documented serve (shared `:4000`, `website/` as cwd):

```bash
./build-site.sh --serve
# Ready: Jekyll verbose log + http://localhost:4000/
# Teardown: Ctrl-C the process you started. Do not killall jekyll/ruby.
```

`./build-site.sh --serve` always binds the Jekyll default (`localhost:4000`) and does not take a port flag. Two agents must not both drive that shared instance.

Verification launch (isolated port + destination):

```bash
chmod +x .cursor/skills/verify-zk0/scripts/control-zk0
.cursor/skills/verify-zk0/scripts/control-zk0 launch
export ZK0_VERIFY_RUN_ID=<printed RUN_ID>
.cursor/skills/verify-zk0/scripts/control-zk0 doctor
```

`launch` copies root `docs/`, `LICENSE`, and `CONTRIBUTING.md` into `website/` the same way `./build-site.sh --serve` does, then starts:

`bundle exec jekyll serve --source . --destination /tmp/zk0-verify-$RUN_ID/site --host 127.0.0.1 --port $PORT --disable-disk-cache --no-watch`

Default `$PORT` is `14000` (`ZK0_VERIFY_PORT` overrides). Ready when `GET $URL/` is 200 and the HTML contains `Welcome to zk0`. The Jekyll log is `.cursor/skills/verify-zk0/runs/$RUN_ID/jekyll.log`.

Teardown: `control-zk0 stop` (see Cleanup).

## Doctor

Run `control-zk0 doctor` first whenever anything looks off. It is read-only. It must report:

- the PID recorded for this `RUN_ID` is alive
- that PID (or its recorded listen child) owns `$PORT` (`lsof` / `ss`)
- destination dir is `/tmp/zk0-verify-$RUN_ID/site` (this run's build, not `website/_site` from a stray `./build-site.sh`)
- `GET $URL/` is 200 and contains `Welcome to zk0` and `Community Buzz`
- `GET $URL/docs/INSTALLATION.html` is 200

Refuse to drive `http://127.0.0.1:4000`, `https://zk0.bot`, or any other URL unless doctor says this run's PID owns that listen port. Two verification instances can run side by side if each has its own `--port` / `ZK0_VERIFY_PORT` and destination. Never attach to a shared instance.

## Drive

Harness: `control-zk0` (curl) against `{URL}` from `control-zk0 env`. Use a browser/CDP only as a second view for screenshots; do not treat a screenshot alone as the action.

Stable handles from this repo (`website/index.md`, `website/_includes/footer.html`, copied `docs/*.md`):

| Visitor action | Handle / route |
|---|---|
| Home | `GET /` — `h1` "Welcome to zk0", `.hero`, `.intro` |
| Concept figure | `img[alt="zk0 Federated Learning Concept Diagram"]` → `/assets/images/zk0-fl-concept.png` |
| White paper | `a[href="/docs/zk0-white-paper-Dec-2025.pdf"]` |
| Get Started → developers | `a[href="/docs/INSTALLATION.html"]` ("For Developers") |
| Get Started → operators | `a[href="/docs/NODE-OPERATORS.html"]` ("For Node Operators") |
| Community Buzz | `.social-posts` heading "Community Buzz", `blockquote.twitter-tweet` |
| Footer nav | `footer.site-footer` links: `/`, `/docs/INSTALLATION.html`, `/docs/RUNNING.html`, `/docs/ARCHITECTURE.html`, `/docs/NODE-OPERATORS.html`, `/CONTRIBUTING.html` |
| Docs H1s | Installation; zk0 Node Operators Guide; Running the Project; Architecture |
| Installer (docs) | NODE-OPERATORS page text `curl -fsSL https://raw.githubusercontent.com/ivelin/zk0/.../get-zk0bot.sh \| bash` |
| Installer (site file) | `GET /get-zk0bot.sh` only if `get-zk0bot.sh` exists at repo root (copied in build mode). This checkout does not contain that file. |

```bash
.cursor/skills/verify-zk0/scripts/control-zk0 fetch / --out .cursor/skills/verify-zk0/artifacts/$ZK0_VERIFY_RUN_ID/home/home
.cursor/skills/verify-zk0/scripts/control-zk0 assert / --contains "Welcome to zk0" --contains "Community Buzz"
.cursor/skills/verify-zk0/scripts/control-zk0 drive home
.cursor/skills/verify-zk0/scripts/control-zk0 assert /docs/INSTALLATION.html --contains "Installation"
.cursor/skills/verify-zk0/scripts/control-zk0 assert /docs/NODE-OPERATORS.html --contains "zk0 Node Operators Guide"
```

Follow the matching file under `features/` for the labeled action/result pairs. Do not start `zk0bot server`, SuperLink, or any FL process as a stand-in for a visitor path.

## Evidence

Write under `.cursor/skills/verify-zk0/artifacts/$RUN_ID/<feature>/`. That directory survives `stop`.

Proof standards:

- Exercise the real visitor path (browser URL or curl of the same path a user opens). Do not rewrite Markdown in place of hitting the served HTML.
- Capture the action and the resulting state, not only the final screen: save `*.status`, `*.headers`, and `*.body` (or a screenshot taken after the click).
- Side effects: this site is static. The second view is a re-GET of the same path, plus any linked asset (`custom.css`, concept PNG, white paper PDF) that the page claims.
- Mocks only at production boundaries. Twitter embed iframes load `platform.twitter.com`; curl proof is the in-repo `blockquote.twitter-tweet` markup, not a live tweet card. Do not mock Jekyll or replace docs HTML with fixtures.
- Record the feature ID and entry point in `PROOF.md` next to the captures.
- `drive home` writes `PROOF.md` plus home HTML, `custom.css`, concept PNG, and the white paper PDF.

## Cleanup

```bash
.cursor/skills/verify-zk0/scripts/control-zk0 stop
```

Kills only the PID (and its children) recorded in `.cursor/skills/verify-zk0/runs/$RUN_ID/state.env`. Deletes `/tmp/zk0-verify-$RUN_ID` and that run's state directory. Does not delete `.cursor/skills/verify-zk0/artifacts/`. Does not kill by process name `jekyll`, `ruby`, or `bundle`. After a failed attempt, run `stop` before launching again so ports and dest dirs are not stranded.

Copied `website/docs/`, `website/LICENSE`, and `website/CONTRIBUTING.md` are the same gitignored serve scaffolding `./build-site.sh --serve` leaves behind; `stop` does not remove them.

## Helpers

`.cursor/skills/verify-zk0/scripts/control-zk0` is executable. Subcommands: `launch`, `doctor`, `env`, `fetch`, `assert`, `drive home`, `stop`. It resolves the repo root from its path.

```bash
CTRL=.cursor/skills/verify-zk0/scripts/control-zk0
$CTRL launch
export ZK0_VERIFY_RUN_ID=<printed RUN_ID>
$CTRL doctor
$CTRL drive home
$CTRL stop
```
