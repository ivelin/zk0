# Fix-verification soak — outcome (2026-11-21)

Verdict: **SOAK PASS.** The serialized Flower substrate at study scale
(5 supernodes, 5-of-5 per round, local-epochs=2 as this code executes
it — a step count — batch 16, FedProx mu=0.01, bf16) ran clean past
the fix-verification gate:

- Segment 1 (runH): 3 rounds, 15/15 fits, 3 checkpoints, 11.2 min, rc=0.
- Segment 2 (runI): 20 rounds, 100/100 fits, 20 checkpoints, 70.3 min
  continuous (gate: >=52 min = 2x the longest fatal survival), rc=0.
- Profile throughout: load-following power, RAM releasing at every
  client handover (40 releases >2 GB in segment 2), temp max 73 C,
  zero guard STOP/CONT, Hub push skipped every round (hf_repo_id
  commented out for the soak).

Full record: reproduction protocol + postmortem "Root cause —
certified" section, research notes zk0-dgx-outage2-20261121 (local
mirror). Study 001 FL is cleared to re-execute on this substrate.
