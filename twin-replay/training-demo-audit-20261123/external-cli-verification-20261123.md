# External CLI Verification Stage — Training-Demo Certification Audit (2026-11-23)

**Stage:** external final verification of the training-demo certification
audit (`training-demo-certification-audit-20261123.md`, committed
`6ad16ac`), per the founder's standing rule of 2026-11-23 14:29 CT —
important results are verified through BOTH the Grok Build CLI and the
Gemini Antigravity (agy) CLI before delivery. **Correction on the
record:** the audit's §6 reported neither CLI found on Muse's VM or
the DGX; that search was too narrow. Per the founder's correction
(2026-11-23 15:12 CT), the Grok Build CLI is installed on the DGX and
this stage was run with it. The agy leg remains outstanding (§6).

## 1. Tool location

- **Grok Build CLI — FOUND on the DGX Spark:**
  `/home/openclaw/.local/bin/grok` → `/home/openclaw/.grok/bin/grok`,
  version `grok 1.0.88 (2fe3608f7398) [stable]`, exit code 0 on
  `--version` and on a headless single-turn smoke prompt (`READY`).
  The earlier search missed it because it did not cover the openclaw
  login user's `~/.local/bin` / `~/.grok` on the DGX.
- **agy / Antigravity CLI — NOT FOUND on the DGX.** Wide net, all
  read-only: login-shell PATH of the openclaw user; `~/.local/bin`,
  `~/bin`, `~/tools`, `~/fleet`, `/home/openclaw/fleet/bin`;
  `/usr/local/bin`; `/opt`; `~/snap/bin`; linuxbrew bin
  (`/home/linuxbrew/.linuxbrew/bin`); npm globals (`~/.npm-global`);
  `find` over `/home`, `/usr/local`, `/opt` (maxdepth 5) for
  `agy` / `antigravity*` / `*antigrav*`; the other readable homes
  (zk0bot, username). Only unrelated hits: an openclaw extension
  directory `google-antigravity-auth`, a models-config test file, and
  the Python stdlib `antigravity.py` module in linuxbrew. No `agy` or
  `antigravity` executable exists on this machine under any of these
  roots.

## 2. Method

Blind, same evidence, same criteria as the audit. The Grok CLI was
given, per episode, the audit's own contact sheet (front + top cameras,
6 timepoints; the exact sheets preserved in
`training-demo-audit-20261123/frames/`, copied to the DGX workdir
`/home/openclaw/muse-work/zk0-extverify-20261123/`) and the audit's
frozen judging brief (`JUDGING_BRIEF.md` — Study 001 frozen per-task
completion criteria, 2026-11-20) with an episode list appended. The
audit's verdicts were never present in the Grok workdir or prompts.
Ten headless single-turn runs (`grok --prompt-file`, web search
disabled), 25 episodes per run, 2 per task; every run exited 0 and
returned exactly 25 verdict lines in the brief's format. Per-episode
Grok verdicts: `verdicts_grok.csv` alongside this file.

## 3. Results

Coverage: **250/250 episodes** — the audit's full result set:
(a) all 249 PASS episodes, (b) the 1 FAIL (block ep604), (c) the 1
qualified PASS (cable ep2201).

**Agreement with the audit: 249/250 = 99.6%.**

| Task | Episodes | Agree | Grok PASS | Grok FAIL | Grok INDET. |
|---|---|---|---|---|---|
| block_stack | 50 | 49 | 48 | 2 | 0 |
| eye_drops_to_basket | 50 | 50 | 50 | 0 | 0 |
| ring_insert | 50 | 50 | 50 | 0 | 0 |
| tool_insert | 50 | 50 | 50 | 0 | 0 |
| cable_clip | 50 | 50 | 50 | 0 | 0 |
| **Total** | **250** | **249** | **248** | **2** | **0** |

## 4. The one disagreement

- **ep573 (block_stack)** — audit: PASS ("green block stacked on
  purple block, standing, gripper released"). Grok: FAIL ("gripper is
  back down on the green block, stack not released"). The dispute is
  narrow and specific: whether the gripper has released the stack in
  the final frames. Neither verdict is withdrawn here; under the
  certified-data gate this episode is **contested** and goes to
  human-eyes adjudication — it is not certified by this stage.

## 5. Key confirmations

- **ep604 (block_stack), the audit's one FAIL — independently
  confirmed FAIL by Grok** ("orange cube sits alone on the mat, not
  stacked on the purple block"), matching the audit's evidence clause
  almost word for word.
- **ep2201 (cable_clip), the audit's qualified PASS — Grok: PASS,
  unqualified** ("cable stays in the holder with the gripper
  retracted"). Grok concurs with the audit's primary verdict; the
  internal verifier's visibility caveat stands in the audit record.

## 6. What this stage does not close

- **The agy (Gemini) leg is still owed** — no agy/antigravity
  executable exists on the DGX (§1). The external stage is therefore
  half-executed: Grok done, Gemini outstanding, reported openly per
  the rule (never silently skipped).
- **Human-eyes approval is still owed** for the contested episode
  (ep573) and, per the founder's certified-data gate (2026-11-23
  15:34 CT), for the set as a whole before any training use.
- Net certified-by-this-stage count: 248 episodes carry audit + Grok
  agreement on PASS; ep604 is confirmed FAIL by both; ep573 is
  contested (1).
