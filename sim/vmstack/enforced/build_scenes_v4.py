"""Build scenes-v4/: the active scenes re-pointed at the v4 arms
(corrections lane, 2026-11-27). Text-level include swap ONLY — every
other byte of each scene is identical to its admitted v3/v3cam source,
so spawn poses, objects and the fitted camera rigs carry over exactly.

ep91000 is included because the B3 canary re-run needs the C1 scene on
the v4 arm. The folded-home keyframe is added to the three ACTIVE
scenes afterwards by apply_folded.py.
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "scenes-v4"
OUT.mkdir(exist_ok=True)
ARM = str(HERE)  # absolute include prefix used by the v3 scenes

SWAPS = [
    ("scenes-v3/block_stack__ep453_v0__v3cam.xml",
     "block_stack__ep453_v0__v4cam.xml",
     f'include file="{ARM}/arm_padded_v3cam.xml"',
     f'include file="{ARM}/arm_padded_v4cam.xml"'),
    ("scenes-v3/block_stack__ep411_v3__v3cam.xml",
     "block_stack__ep411_v3__v4cam.xml",
     f'include file="{ARM}/arm_padded_v3cam.xml"',
     f'include file="{ARM}/arm_padded_v4cam.xml"'),
    ("scenes-v3/ring_insert__ep48_v1__v3cam.xml",
     "ring_insert__ep48_v1__v4cam.xml",
     f'include file="{ARM}/arm_padded_v3cam.xml"',
     f'include file="{ARM}/arm_padded_v4cam.xml"'),
    ("scenes-v3/ring_insert__ep91000_v0__v3.xml",
     "ring_insert__ep91000_v0__v4.xml",
     f'include file="{ARM}/arm_padded_v3.xml"',
     f'include file="{ARM}/arm_padded_v4.xml"'),
]


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


record = {}
for src, dst, old, new in SWAPS:
    text = (HERE / src).read_text()
    assert text.count(old) == 1, (src, text.count(old))
    (OUT / dst).write_text(text.replace(old, new))
    record[dst] = {"source": src, "source_sha256": sha256(HERE / src),
                   "sha256": sha256(OUT / dst), "change": "arm include only"}
    print(dst, record[dst]["sha256"][:16])

(HERE / "corrections-20261127" / "scenes_v4_build_record.json").write_text(
    json.dumps(record, indent=1))
print("record written")
