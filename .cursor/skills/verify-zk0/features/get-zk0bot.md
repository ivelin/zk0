# get-zk0bot installer path

Visitors are told to install zk0bot with a curl-piped `get-zk0bot.sh` one-liner documented on the Node Operators page. Jekyll is also configured to publish a `get-zk0bot.sh` file if that script is present in the site source. This checkout does not contain `get-zk0bot.sh` at the repo root (and `website/get-zk0bot.sh` is gitignored), so the local site path may 404 even while the docs path is valid.

## Sub-features

- `installer-docs` shows the curl one-liners on `/docs/NODE-OPERATORS.html`.
- `installer-site-file` serves `GET /get-zk0bot.sh` when the script was copied into `website/` (`website/_config.yml` `include: get-zk0bot.sh`).
- `installer-absent` records a 404 and the missing source file instead of inventing a script.

## How to get to it (user POV)

- Open Node Operators and read **Install zk0bot CLI** / the Full Production Session Example curl blocks.
- If the site is publishing the script, request `https://zk0.bot/get-zk0bot.sh` or `{URL}/get-zk0bot.sh` (the path `build-site.sh` build mode copies and memory-bank curl-tests).
- The home page does not link `get-zk0bot.sh` directly in `website/index.md`.

## Driving it with control-zk0

Preconditions:

- `control-zk0 doctor` is green.
- You have already opened, or will open, `/docs/NODE-OPERATORS.html` as the user entry (not a raw GitHub fetch as the primary action).
- Do not pipe the installer into `bash` during verification. Do not clone onto `~/zk0`.

- **Read the documented curl.** `control-zk0 fetch /docs/NODE-OPERATORS.html --out .cursor/skills/verify-zk0/artifacts/$ZK0_VERIFY_RUN_ID/get-zk0bot/ops`. `ops.body` contains both:
  - `https://raw.githubusercontent.com/ivelin/zk0/dev/get-zk0bot.sh`
  - `https://raw.githubusercontent.com/ivelin/zk0/main/website/get-zk0bot.sh`
- **Probe the site file.** `control-zk0 fetch /get-zk0bot.sh --out .cursor/skills/verify-zk0/artifacts/$ZK0_VERIFY_RUN_ID/get-zk0bot/script`.
- **If the source file exists** at repo-root `get-zk0bot.sh` (launch copies it into `website/`): `script.status` is `200`, body starts with a shell shebang, and `PROOF.md` says the site path is published.
- **If the source file is absent** (this checkout): `script.status` is `404`. Write that status into `PROOF.md`. That is a complete proof of the visitor path as shipped: docs describe GitHub raw URLs; the Jekyll include has nothing to publish. Do not add a dummy `get-zk0bot.sh` to make the GET succeed.
- **Second view.** Re-assert the NODE-OPERATORS curl strings. Re-GET `/get-zk0bot.sh` and confirm the same status as the first probe.

## Gotchas

- `./build-site.sh` copies `../get-zk0bot.sh` only in **build** mode, not `--serve`. `control-zk0 launch` copies it only when the root file exists.
- `website/get-zk0bot.sh` is listed in root `.gitignore`. A local leftover of that path is not proof it ships on GitHub Pages.
- Hitting `raw.githubusercontent.com` is a production boundary. Optional HEAD of those URLs may 404 if the file was never committed; that does not invalidate the docs-page proof.
- Never execute the installer. Execution would clone, create conda envs, and is outside this skill.
