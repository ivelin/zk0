# Instruction Audit — zk0 lane, 2026-11-23 (auditor, third role)

Founder order 2026-11-23 12:28 CT: audit ALL ongoing and planned zk0-lane work against his instructions; remove all inconsistencies. This register is the instrument — the auditor finds and rules; the parent executes removals. Sources read raw: MEMORY.md; decisions SSOT mirror (Decisions 62, 63 incl.); twin-suite v2 pre-registration + Amendments 1–3; FL search queue incl. re-sort #1; AGENTS.md. Live state verified at source 2026-11-23 ~12:30–12:40 CT: cron.list, hooks.list, DGX processes/records/wrappers, remote branch tip (study branch `adbead8` = the Amendment 3 commit), subagent states.

**Headline: no work was found actively violating the replica closure or the promotion gate at audit time.** The two replica-directed workstreams had already been stood down / re-briefed at 12:27–12:28 CT (verified at source below). Findings are therefore STALE-BRIEF / RE-BRIEF / NOTE — no live STOP-WORK.

## Inconsistencies (severity · corrective action)

| # | Item | Ruling | Evidence at source | Corrective action |
|---|---|---|---|---|
| I-1 | Matrix precision production (checkpoint evals in episode-replica scenes) | STALE-BRIEF · RE-BRIEF (correction executed 12:27–12:28, verified) | Briefed 2026-11-23 09:45, ~12 h after the 2026-11-22 21:44 ruling; precision wrapper terminated ~12:28 CT (PID 234086 gone, verified two ways by stop-work ops); stand-down logged in matrix protocol.json corrections (2026-11-23 12:27 CT entry); zero precision rollouts ever produced | Precision checkpoint evals re-brief onto spec-only scenes when the spec-only suite validates; do not relaunch the precision wrapper against replica scenes |
| I-2 | Ring executor-revision workstream (replica-scene validation as its task) | STALE-BRIEF · RE-BRIEF (correction executed 12:27, verified) | Dispatched 2026-11-23 09:42 with replica-scene validation as the deliverable, post-ruling; interrupted and re-briefed 12:27 (agent running under the STOP brief at audit time); all rev3 DGX launch attempts (11:41/11:56/12:11 CT) were PREFLIGHT_FAIL — no replica compute actually ran | Controller-only work continues; its validation moves to spec-only ring scenes per Decision 63(iii); no new replica-scene rounds |
| I-3 | Monthly realism re-audit cron — owed, never created | INCONSISTENT (omission) · RE-BRIEF | Directive 2026-11-22 17:47 CT (MEMORY.md): monthly realism re-audit, cron to be created when suite v1 lands; cable_clip v1.1 certified 2026-11-22 20:01 CT; cron.list at audit time contains no such job | Create the cron per the design in the twin-replay v1 completion report |
| I-4 | Matrix completion legs (per-task verification, frontier judging, per-task reports, Phase 6 ingestion) — ownerless | INCONSISTENT (ownership gap) · RE-BRIEF | Successor agent b2a0abe8 completed its session after the aggregator fix; the cable wrapper runs on-box unattended, but no agent is tasked to run the verification/judging/reporting legs when cable production completes | Parent re-tasks a matrix shepherd before cable production completes |
| I-5 | Parent todo: "solvability extension still owed" (56 un-attempted replica gate scenes) | INCONSISTENT (stale tracking) · NOTE | Decision 63 / Amendment 3 A3.2 cancels replica-scene solvability extension; solvability validation now applies to spec-only scenes under the spec-only program | Rewrite the v2 follow-ons todo item to match Decision 63 |
| I-6 | Circuit-breaker third healthy trace (spec_exp run) — never collected | INCONSISTENT (omission) · NOTE | Founder decision 2026-11-22 19:17 CT: collect the specialists' run as the third trace, then re-judge T1 arming; spec_exp completed 2026-11-23 01:50 CT, telemetry on DGX; no owner, no schedule, no cron. Safe state holds (T1 stays log-only) | Task the trace extraction + arming re-judgment card |
| I-7 | Instruction-source ambiguity, 21:44 → 12:24 (root cause of I-1/I-2) | NOTE | Decision 62 / Amendment 2 §A2.2 kept Stage 1 episode-suite work alive in writing ("reference controllers are the operative solvability route there (a Stage 1 build item)"); MEMORY.md's 21:44 entry records the spec-only path but no replica stop; the unambiguous written stop is Decision 63 (2026-11-23 12:24 CT). The founder's stated intent was closure at 21:44 | None technical — Decision 63 governs unambiguously from 12:24; future rulings should carry an explicit stop-list so the text and the intent cannot diverge |
| I-8 | Cable evals — live tension between the written record and the founder's 12:29 CT question | NOTE | Register rules Phase 6 + matrix-cable CONSISTENT on the written record (founder GO 2026-11-23 07:12 CT; Decision 63(iii): "Phase 6 and the closed-loop matrix in already-built scenes are unaffected and continue"). The founder's 12:29 CT question ("What cable evals did I sanction???") suggests he may not intend the matrix-cable continuation as registered | Parent reconciles with the founder; if he narrows it, the matrix cable wrapper is the next stand-down. Not stop-work: two founder-tapped written decisions authorize it |
| I-9 | Blinded-set sealed key + twin-gallery v2 images exist only on this VM | NOTE | Backup audit 2026-11-23: key deliberately kept out of the repo (blinding integrity) → no remote backup, against the spirit of the 11:24 CT backup directive; already flagged to the founder | Founder decision: accept the SPOF or authorize a private sealed backup |
| I-10 | Q-A gate PASS used the logged STUDY_GATE_ALLOW_OVERRIDES path | NOTE | Disclosed by the builder: overrides selected only which saved-checkpoint artifact the checker read (the miniature's own template, not the historical defective one); no recipe value overridden; launcher accepted via its deliberate logged path | None — recorded for completeness; white-paper record should carry the disclosure (it does, in the Q-A record) |
| I-11 | Phase 6 coordinator quiet ~50 min while its on-box driver produces | NOTE | Agent cc9d63dc last activity ~49 min before audit; records grew 315 → 330 in the window; driver + workers + policy servers up | Parent nudges the coordinator; no instruction violated |
| I-12 | zk0-stack-freshness-check cron body cites the pre-v2 stack description | NOTE | Cron body (monthly, next run Dec 16) references the 2026-11-15 Stack A description; report-only job, no stack change authority | Refresh its references at the next edit |

## Consistent items (one line each)

| Item | Matches instruction |
|---|---|
| Q-A training run (running, round ~5, guard telemetry live) | FL queue re-sort #1 GPU order (Q-A first, AMBIGUOUS branch default); founder output-minimized route approval 2026-11-23 07:53 CT; §7 prep package (carry commit, gate manifest, full-path smoke + kill-resume drill) |
| Q-A completion cron `study001-qa-completion-20261123` (enabled, 30-min) | Body matches its brief: package → HF backup to NEW repo zk0bot/study001-fl-qa-round30 → judged eval → independent verification → commit → report → self-remove; bars launching any other queue item; never writes ivelin/zk0-smolvla-fl |
| Phase 6 cable closed-loop evals (330 records, driver running) | Founder GO 2026-11-23 07:12 CT; Decision 63(iii) continuation; Stage 0 = certified twin-suite v1.1 |
| Matrix cable wrapper (alive, PID 234084) | Decision 63(iii) — matrix in already-built scenes continues (see I-8) |
| Matrix aggregator v2 fix (sources separated, timestamps) | Verify-before-asserting rule; independent reconciliation reproduced its counts exactly |
| Spec-only program + registration | Decision 63 / Amendment 3 — committed and remote-verified (study branch tip adbead8 is the Amendment 3 commit; Decision 63 appended to the SSOT mirror) |
| Spec-only Pop!_OS move (agent running under the 12:26 brief) | Founder authorization 2026-11-23 12:26 CT, same play-nice rules (≤70% whole-box, guard discipline, no boot services) written into the brief |
| Blinded rollout set (8/12 delivered, key sealed) | Founder request 2026-11-23 09:43 CT; blinding discipline verified (metadata stripped, mid-frames inspected); shortfall stated, not padded |
| Daily branch-sync cron (ran 2026-11-23 09:57 CT) | Report-only by design; maintainer's daily sync visibility (2026-11-16) |
| Guard weekly selftest cron (enabled) | AGENTS.md guard rule — never report the guard as working without a passing self-test |
| Hooks: slack-bothub-watch, drive-mirror-watch (both enabled) | Standing comms/sync monitoring; no instruction conflict found |
| Demo-laptop price-watch crons (AM/PM, goal-owned) | Demo-machine watch spec (HW4-class, new only); wrap-up after 2026-12-01 still in window |
| Q-B (fewer/longer rounds) — planned, not launched | Re-sort #1 order Q-A → Q-B → Q-D; launches only after Q-A readout |
| Q-D (FedAvgM β=0.9) — planned, not launched | Re-sort #1 order; after Q-B |
| Q-WA (weighting-off ablation) — planned, not launched | Queue §6: after the structural three, post Q-DW audit |
| Q-HPO — planned, not opened | Dependency-gated on Q-A/Q-B/Q-D readouts (queue §5/§6) |
| Q-STEPS — not launched | Trigger evaluated at re-sort #1: NOT FIRED (r20→r30 +4 pp, inside the noise floor); demoted — launching it would have been the inconsistency |
| Q-LR — not launched | Conditional behind the first structural readout; never combined with Q-A carry (queue §6) |
| Spec-only Stage 2 gate runs — not started | Pre-registration §4/§7: Stage 2 requires per-task Stage 1 certification + frozen Bounds Amendment; neither exists for the spec-only suite yet — starting now would violate the pre-registration |
| §4 judging layer (gate) — not executed | Same preconditions as Stage 2; the matrix's separate judging leg is covered by I-4 |
| Bounds Amendment — drafted, deliberately NOT registered | Pre-registration §7 procedure preserved; freezing bounds for a non-operating gate would misrepresent its status |
| Piece 12 honesty pipeline — not drafted | Founder instruction 2026-11-22: Cos consultation → verify → humanise → panel → founder approval; Cos replies still outstanding; nothing published — promotion gate intact |
| Promotion gate — no violation found | No social/website promotion in the audited set; v0.1.0 GitHub release was founder-approved and GitHub-only (2026-11-23 08:07 CT) |
| White-paper + backup directives (2026-11-23 11:23/11:24 CT) | Same-day backup audit executed; gap-closing push 1d4fd1b remote-verified; in-flight records DGX-only by design until completion (see I-9) |

## Counts

- Audited: 36 items (12 with findings, 24 consistent).
- Inconsistent by severity: STOP-WORK (active) 0 · STALE-BRIEF/RE-BRIEF 4 (I-1, I-2, I-3, I-4) · NOTE 8 (I-5 … I-12).
- I-1 and I-2 corrections were executed at 12:27–12:28 CT, before this audit, and are verified at source above; the remaining open corrective actions are I-3 (create the re-audit cron), I-4 (re-task a matrix shepherd), I-5 (rewrite the stale todo), I-6 (task the third-trace collection), I-8 (founder reconciliation on cable scope).
