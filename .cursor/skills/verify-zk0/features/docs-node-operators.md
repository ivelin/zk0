# Node Operators docs

Node Operators is the visitor guide for applying to the zk0 federated network, installing zk0bot via the documented curl one-liner, and reading server/client command examples. The page is documentation. It does not start a SuperNode.

## Sub-features

- `ops-open` serves `/docs/NODE-OPERATORS.html` with H1 "zk0 Node Operators Guide".
- `ops-from-home` is reached from home Get Started "For Node Operators".
- `ops-apply` links the GitHub issue template for node-operator applications.
- `ops-installer-text` shows the `get-zk0bot.sh` curl one-liners (see also [get-zk0bot installer path](./get-zk0bot.md)).

## How to get to it (user POV)

- On home, choose **For Node Operators**.
- In the footer, choose **Node Operators**.
- Open `{URL}/docs/NODE-OPERATORS.html` or `https://zk0.bot/docs/NODE-OPERATORS.html`.
- From Installation or Architecture, choose the Node Operators cross-link.

## Driving it with control-zk0

Preconditions:

- `control-zk0 doctor` is green.
- Do not run `zk0bot server start`, `zk0bot client start`, or any SuperLink command during this proof.

- **Follow home CTA.** Run `control-zk0 assert / --contains 'href="/docs/NODE-OPERATORS.html"' --contains "For Node Operators"`. Then `control-zk0 fetch /docs/NODE-OPERATORS.html --out .cursor/skills/verify-zk0/artifacts/$ZK0_VERIFY_RUN_ID/docs-node-operators/page`.
- **Rendered title.** `page.status` is `200`. `page.body` contains `zk0 Node Operators Guide` and `Install zk0bot CLI`.
- **Application path.** `page.body` contains the GitHub issue URL with `template=node-operator-application.md` (`https://github.com/ivelin/zk0/issues/new?template=node-operator-application.md`).
- **Installer documentation.** `page.body` contains `curl -fsSL https://raw.githubusercontent.com/ivelin/zk0/dev/get-zk0bot.sh | bash` and `curl -fsSL https://raw.githubusercontent.com/ivelin/zk0/main/website/get-zk0bot.sh | bash`.
- **CLI examples (text only).** `page.body` contains `zk0bot server start` and `zk0bot client start` as documentation. Observing that text is the proof. Running those commands is out of ownership.
- **Second view.** `control-zk0 assert /docs/NODE-OPERATORS.html --contains "zk0 Node Operators Guide" --contains "get-zk0bot.sh"`.
- **Optional browser view.** Click **For Node Operators** on `{URL}/`. Location is `/docs/NODE-OPERATORS.html`.

## Gotchas

- The page describes tmux, conda, SuperLink, and SuperNode. Those are operator runbook sentences. Starting them fails the ownership contract for this skill.
- Two different raw GitHub URLs appear (`dev/get-zk0bot.sh` vs `main/website/get-zk0bot.sh`). Assert the strings the page actually has; do not "fix" them during verify.
- The issue template lives in `.github/ISSUE_TEMPLATE/node-operator-application.md`. A 200 on that GitHub URL is a production-boundary check, not required for the local docs proof.
