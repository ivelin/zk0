# home proof

- feature: home
- entry: GET http://127.0.0.1:14000/
- run: 20260914T124849-3238
- captured: 2026-09-14T12:48:59Z
- action: GET / (visitor home) via `control-zk0 drive home`
- resulting state: 200 HTML with Welcome to zk0, Discover zk0, Community Buzz, Get Started links
- side effects: none (static Jekyll page). Second GET / still 200 with the same identity strings.
- assets: custom.css=200 concept.png=200 white-paper.pdf=200
- production boundary: Twitter widget JS (platform.twitter.com) is not executed by curl; markup `twitter-tweet` + Community Buzz is the in-repo proof.
- second view: `home-browser.png` (headless Chrome). Hero, intro, and concept diagram visible. Twitter cards not required.
- cleanup: `control-zk0 stop` removed `/tmp/zk0-verify-20260914T124849-3238` and `runs/20260914T124849-3238`. This directory remained.
