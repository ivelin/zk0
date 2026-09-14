# verify-zk0 artifacts

Proof from a verification run is written here as:

`.cursor/skills/verify-zk0/artifacts/$RUN_ID/<feature>/`

`control-zk0 stop` deletes the Jekyll dest dir and `.cursor/skills/verify-zk0/runs/$RUN_ID` only. It never deletes this directory.

Typical files from `control-zk0 drive home`:

- `PROOF.md` — feature id, entry URL, action, resulting state, side-effect note
- `home.status` / `home.headers` / `home.body`
- `custom.css.*`, `concept.png.*`, `white-paper.*`
- optional `home-browser.png` (second view; Twitter cards may stay unloaded)

Large re-fetched binaries (`white-paper.body`, `concept.png.body`) are gitignored; they remain on disk after `stop`. Status/headers for those assets are committed with the proof.
