# twin-suite v1.1 — STATUS: cable_clip CERTIFIED; other four tasks NOT CERTIFIED

Definitive validation: RUN3 (`validation/validation_results_v11.json`),
param_hash `a4eb3f8c3611894d`, seed 20261122, denominator **96** suite-wide
(four low-confidence tool_insert episodes excluded under the pre-registered
rule: ep630, ep779, ep871, ep876). Certification ruling: independent
verification V-E2E, `docs/verification-v11-e2e.md` — an independent re-run
reproduced the recorded results exactly (160/160 jobs).

| Task | Ruling | Reproduced |
|---|---|---|
| cable_clip | **CERTIFIED** | 20/20 (certified variant set: all 100 variants) |
| block_stack | NOT CERTIFIED | 0/20 — pose-σ structural limit: pose accuracy (~2.5–3.5 cm) exceeds the 1–2 cm task tolerance |
| eye_drops_to_basket | NOT CERTIFIED | 3/20 — pose-σ structural limit (as above) |
| ring_insert | NOT CERTIFIED | 0/20 — pose-σ structural limit (as above) |
| tool_insert | NOT CERTIFIED | 0/16 — pose-σ structural limit (as above) |

Publication: ONLY the certified cable_clip set is published, to Hugging
Face dataset repo `zk0bot/twin-suite` (public, CC-BY-4.0), pinned revision
`579ad96b960d1c6ad756b3e08518aa0b04eb60e7` —
https://huggingface.co/datasets/zk0bot/twin-suite
See `HF-REFERENCE.md` in this folder for the pinned revision, the sha256
manifest aggregate, and the D-02 external-input pins. The full
non-certified research record for all five tasks is in this folder
(`twin-replay/`). Nothing from v1 was or is certified; v1's record remains
below in git history.
