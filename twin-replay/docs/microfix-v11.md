# Twin-Suite v1.1 — MICRO-FIX Report (final freeze)

Workdir: /home/openclaw/muse-work/zk0-study001-twin-20261122/
Applied: 2026-11-22, DGX clock — changelog MF-1 stamped 2026-11-22 19:34 CST,
MF-2 and MF-3 stamped 2026-11-22 19:35 CST, each from a fresh date read taken
immediately before its append. CPU-only, nice -n 10, DGX training undisturbed.
Three items only; no full validation was run.

## Item 1 (MF-1, D-01) — fixtures.json hashed

Diff (code/build_scenes.py, HASH_DATA_FILES only):

    HASH_DATA_FILES = ["scenes/calibration_fk_affine.json",
                       "scenes/calibration_v2.json", "scenes/detections_px.json",
    +                  "scenes/fixtures.json"]

Nothing else about the hash changes. Per the delta verifier, fixtures are
currently shadowed by all 100 pose JSONs, so this changes the hash VALUE
but no scene and no score.

## Item 2 (MF-2, D-03) — ep784 provenance disambiguated

Diff (scenes/tool_insert__ep784.json, objects.tool.v11_repair only):

    - "chosen_source": "fk_anchor_only"
    + "chosen_source": "fk_anchor_only — SUPERSEDED by fixpass_fp_g visual_grounding"
    + "operative_source": "fixpass_fp_g visual_grounding"

Both mechanisms were applied (supersession string + explicit operative
pointer). History preserved: v11_repair.rule_v2.chosen_source still reads
"fk_anchor_only"; v11_repair.fixpass_fp_g still records the operative
grounding (chosen_source visual_grounding_fixpass, chosen_px [363, 306],
superseded FK pose). Pose VALUES unchanged — verified in-script:
tool pose remains [0.2094, -0.0365, 0.005, null].

## Item 3 (MF-3, coordinator ruling) — ep779 OUT of the denominator

Ruling: tool_insert ep779 is OUT of the v1.1 validation denominator.
Basis: FP-H evidence-based relabel (FK-anchor-only, visually
unconfirmable/indeterminate, confidence medium -> low) triggers the
preregistration registered low-confidence exclusion — the same rule that
excludes ep630/871/876. RUN2 final validation therefore ran at denominator
96 (tool_insert 16). ep779 interim agreement was 0.0, so no headline number
changes. Bookkeeping application of the registered rule, NOT a criteria change.

Written as:
(a) changelog entry MF-3 in validation/param-changelog.md;
(b) docs/twin-replay-amendment3-v11-ep779.md;
(c) additive top-level marker in scenes/tool_insert__ep779.json:
    v11_status: "OUT — low-confidence FK-only pose (amendment3)"
    (objects block verified byte-equivalent pre/post; no value changes).

## NEW param_table_hash

    38e5a9e7dd1d0330  ->  a4eb3f8c3611894d

Recomputed live by importing build_scenes.param_table_hash() after all edits.

## Determinism evidence (under the new hash)

block ep411, cable ep1939 and tool ep784 regenerated (all 5 variants,
make_draws + build, in-memory) in two separate processes; per-variant XML
SHA-256 (first 16 hex) identical across both runs for every variant:

    block_stack ep411:  64ede552f447e582 8a1585cffe6584b1 a69348fae21f4727 8f88413a2c75639c f2891221485f2ec5
    cable_clip ep1939:  e971bd04f6b70760 2cad478ed1db77ba 1db20a35ac9e9770 f05703d829ed3fd5 1a54d6272d01ff7c
    tool_insert ep784:  769a6434a20e9240 ace965234b19030e 9d78111f8797c3d1 77521d9d6beb6210 8402316a9b99e938

Cross-run comparison: BYTE-IDENTICAL (True).

## ep779 enumeration check (no full validation run)

Replicated run_validation.py enumeration for tool_insert only
(load_episode_poses + CONF_OK={medium, high} confidence test, no scene
builds, no replays):

    tool_insert INCLUDED 16 [622, 629, 683, 684, 697, 784, 824, 830, 852, 859, 883, 885, 888, 892, 902, 903]
    tool_insert EXCLUDED 4
      ep630: initial-pose confidence=low (per-object: {tool: low})
      ep779: initial-pose confidence=low (per-object: {tool: low})
      ep871: initial-pose confidence=low (per-object: {tool: low})
      ep876: initial-pose confidence=low (per-object: {tool: low})
    ep779 v11_status marker: OUT — low-confidence FK-only pose (amendment3)

ep779 is excluded by the documented low-confidence rule path, named in the
exclusion output, consistent with the MF-3 / Amendment 3 ruling.
