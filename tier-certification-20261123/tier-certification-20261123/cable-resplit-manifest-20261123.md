# cable_clip re-split manifest — founder decision 2026-11-23 22:11 CT

Decision: build a NEW held-out eval set for cable_clip from the certified training pool, because the original held-out cable set is 20/20 FAIL (founder-adjudicated) and is removed from the active eval set. Scope: cable_clip only; the other four tasks keep their existing held-out sets. Rationale (founder, on record): training will be redone on certified data anyway, so episodes moving from the old training pool into held-out do not contaminate the redo — the redo trains only on the remaining 40.

## Eligibility (applied to the 50 training cable episodes, all primary PASS)
Unanimous PASS across every leg that read the episode (primary, verifier/re-judge where present, Grok, Gemini) AND not flagged under the gripper-release standard AND not contested AND not ep2201 (the qualified PASS, founder-named exclusion; its verifier read was INDETERMINATE).

Eligible pool: 49 episodes: 2159, 2160, 2161, 2162, 2163, 2164, 2165, 2166, 2167, 2168, 2169, 2170, 2171, 2172, 2173, 2174, 2175, 2176, 2177, 2178, 2179, 2180, 2181, 2182, 2183, 2184, 2185, 2186, 2187, 2188, 2189, 2190, 2191, 2192, 2193, 2194, 2195, 2196, 2197, 2198, 2199, 2200, 2202, 2203, 2204, 2205, 2206, 2207, 2208

Excluded: ep2201 only (qualified PASS; verifier INDETERMINATE).

## Draw
Method: random.Random(20261123).sample(eligible pool sorted by episode index, 10). Seed 20261123.

Drawn 10 — NEW held-out eval set for cable_clip (cleanliness grade recorded per the decision):

- ep2164 — cleanliness CLEAN
- ep2169 — cleanliness CLEAN
- ep2170 — cleanliness CLEAN
- ep2183 — cleanliness CLEAN
- ep2190 — cleanliness CLEAN
- ep2193 — cleanliness CLEAN
- ep2194 — cleanliness CLEAN
- ep2195 — cleanliness CLEAN
- ep2200 — cleanliness CLEAN
- ep2208 — cleanliness CLEAN

## Remaining 40 — training set
2159, 2160, 2161, 2162, 2163, 2165, 2166, 2167, 2168, 2171, 2172, 2173, 2174, 2175, 2176, 2177, 2178, 2179, 2180, 2181, 2182, 2184, 2185, 2186, 2187, 2188, 2189, 2191, 2192, 2196, 2197, 2198, 2199, 2201, 2202, 2203, 2204, 2205, 2206, 2207

## Removed from the active eval set
The original 20 held-out cable episodes (all FAIL; five founder-adjudicated FAIL 2026-11-23: 2020, 2042, 2046, 2092, 2101) are marked eval_set_status = REMOVED in verdicts_master.csv per the founder's "Remove failed ones". Their FAIL records stay in the audit history (heldout-episode-certification-audit-20261123.md); they are no longer part of the active eval set.
