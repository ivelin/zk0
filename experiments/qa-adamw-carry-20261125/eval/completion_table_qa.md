# Q-A completion table — JUDGED completion (offline replay), proxy metric

Final = two-judge unanimity (CERTIFIED). SINGLE-LEG counts are one leg's
verdicts, labeled, never final. DISPUTE packets are excluded from completed
counts and listed in the index. Denominator: 90 held-out episodes
(20/task x4 + cable 10, founder re-split 2026-11-23).

| checkpoint | scope | certified | single-leg completed | disputes | n |
|---|---|---|---|---|---|
| centralized_qa (all 90) | | 0 | 71 | 0 | 90 |
| centralized_qa | block_stack | 0 | 16 | 0 | 20 |
| centralized_qa | eye_drops_to_basket | 0 | 15 | 0 | 20 |
| centralized_qa | ring_insert | 0 | 19 | 0 | 20 |
| centralized_qa | tool_insert | 0 | 12 | 0 | 20 |
| centralized_qa | cable_clip | 0 | 9 | 0 | 10 |
| fl_qa_r30 (all 90) | | 0 | 27 | 0 | 90 |
| fl_qa_r30 | block_stack | 0 | 3 | 0 | 20 |
| fl_qa_r30 | eye_drops_to_basket | 0 | 7 | 0 | 20 |
| fl_qa_r30 | ring_insert | 0 | 10 | 0 | 20 |
| fl_qa_r30 | tool_insert | 0 | 7 | 0 | 20 |
| fl_qa_r30 | cable_clip | 0 | 0 | 0 | 10 |
| fl_qa_r20 (all 90) | | 0 | 15 | 0 | 90 |
| fl_qa_r20 | block_stack | 0 | 2 | 0 | 20 |
| fl_qa_r20 | eye_drops_to_basket | 0 | 3 | 0 | 20 |
| fl_qa_r20 | ring_insert | 0 | 5 | 0 | 20 |
| fl_qa_r20 | tool_insert | 0 | 3 | 0 | 20 |
| fl_qa_r20 | cable_clip | 0 | 2 | 0 | 10 |
| fl_qa_r10 (all 90) | | 0 | 6 | 0 | 90 |
| fl_qa_r10 | block_stack | 0 | 0 | 0 | 20 |
| fl_qa_r10 | eye_drops_to_basket | 0 | 3 | 0 | 20 |
| fl_qa_r10 | ring_insert | 0 | 1 | 0 | 20 |
| fl_qa_r10 | tool_insert | 0 | 2 | 0 | 20 |
| fl_qa_r10 | cable_clip | 0 | 0 | 0 | 10 |
| specialists (diagonal pooled) | | 0 | 71 | 0 | 90 |
| spec_block_stack | diagonal (block_stack) | 0 | 13 | 0 | 20 |
| spec_eye_drops_to_basket | diagonal (eye_drops_to_basket) | 0 | 18 | 0 | 20 |
| spec_ring_insert | diagonal (ring_insert) | 0 | 18 | 0 | 20 |
| spec_tool_insert | diagonal (tool_insert) | 0 | 18 | 0 | 20 |
| spec_cable_clip | diagonal (cable_clip) | 0 | 4 | 0 | 10 |

**Reading the table:** counts are the Muse judge leg only (SINGLE-LEG, labeled, never final) — the Gemini second leg is pending and certification (two-judge unanimity) completes when it lands. For fl_qa_r10/r20 only the fixed 40-packet canary subset was judged (r10: 6/40, r20: 15/40, r30 canary: 12/40); their remaining 50 packets per arm are UNJUDGED in the index. Headline arms (centralized_qa, fl_qa_r30) and the specialist diagonal were judged on the full 90.
