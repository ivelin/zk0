"""Criteria-provenance regression test (ring band-edge postmortem, 2026-11-25).

Founder ruling (2026-11-25): scoring criteria must not carry hand-made
numeric rules that fight the physics engine. Every numeric threshold in a
criterion needs a recorded derivation — a physical measurement or a founder
decision. An underived number is a defect: it may exist only as a FLAGGED
manifest entry with a pending disposition, never silently.

This test enforces that against the frozen twin-suite criteria in
twin-replay/code/ (replay.py, twin_params.py, run_validation.py) using the
manifest twin-replay/code/criteria-provenance.json:

1. Every numeric constant in twin_params.py is classified in the manifest,
   with the value the manifest reviewed.
2. Every manifest source pattern still matches the criteria code exactly
   once (any edit to a threshold's code line forces a manifest review).
3. Every float literal used as a comparison operand in replay.evaluate()
   and in run_validation.py is covered by a manifest entry.
4. Every numeric literal anywhere in evaluate() is covered (complete
   census — a new invented number in the scoring function fails here).
5. Every SCORING/DEVICE entry either carries a derivation + source
   (status DERIVED or FOUNDER-DECIDED) or is explicitly FLAGGED with a
   pending disposition (status RATIONALE-ONLY or UNDERIVED).

Strengthened 2026-11-25 (founder correction, 12:00 CT — classification
alone is not a gate; the gate is):

6. The FLAGGED queue is CLOSED. It holds exactly the underived entries
   the 2026-11-25 postmortem recorded (FROZEN_FLAGGED_QUEUE below, 15
   entries after the ring band's resolution) and may only SHRINK — when
   an entry gains a derivation, is converted to a physics-state read, or
   is removed by a founder ruling. A NEW underived threshold can no
   longer be admitted by flagging it: any flagged entry outside the
   frozen set fails the suite.
7. Every SCORING entry declares what quantity it gates (manifest field
   "quantity"): "engine-resolved" (a height, position, gap, angle,
   settle or contact outcome the physics engine already resolves),
   "input-geometry" (a dimension of a built/measured object — allowed
   only with a measurement citation), "agreement-statistic" (a fraction
   over verdicts, not a physics quantity), or "physics-state" (a boolean
   read of engine state, no numeric band). A threshold on an
   engine-resolved quantity with no derivation is a HARD FAILURE unless
   it is one of the frozen postmortem flags awaiting its own founder
   ruling. Undeclared quantity = unclassified = failure.

Together with checks 3–4 (a new numeric literal in a criterion fails the
census until classified), this means a fresh invented band in ANY task's
criterion goes red three ways: unclassified literal, undeclared
quantity, or admission to a closed queue.

Scope notes (deliberate, recorded): integer literals in comparisons are
indices/counters, not thresholds, and are excluded from check 3; scene
geometry/material constants are classified (kind SCENE) under the
physics-parameters.md tagging regime and are not required to carry
derivations — the derivation rule binds thresholds that gate scoring and
the replay device. Runs in CI via .github/workflows/ci.yml
(pytest tests/unit) and locally with the repo's normal test command.
Stdlib only.
"""

import ast
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = REPO_ROOT / "twin-replay" / "code"
MANIFEST_PATH = CODE_DIR / "criteria-provenance.json"

DERIVED_STATUSES = {"DERIVED", "FOUNDER-DECIDED"}
FLAGGABLE_STATUSES = {"RATIONALE-ONLY", "UNDERIVED"}
ALL_STATUSES = DERIVED_STATUSES | FLAGGABLE_STATUSES | {"LABELED"}
GATED_KINDS = {"SCORING", "DEVICE"}

# What a SCORING entry may gate (manifest field "quantity"). Only
# "engine-resolved" bands are the defect class the founder's ruling
# targets: a hand-made number applied to a quantity the physics engine
# already resolves. "input-geometry" entries are dimensions of built or
# measured objects and must cite the measurement (check 5 enforces the
# citation for derived entries). Bookkeeping constants (counts, indices)
# never gate a quantity and stay out of this classification.
QUANTITY_CLASSES = {
    "engine-resolved",
    "input-geometry",
    "agreement-statistic",
    "physics-state",
}

# The flagged queue as recorded by the ring band-edge postmortem
# (2026-11-25), after the ring band itself was resolved by the founder's
# ruling the same day. CLOSED set: it may only shrink (derivation,
# conversion to a physics-state read, or founder-ruling removal). Any
# flagged entry outside this set is a new underived threshold trying to
# enter by being flagged — a hard failure (founder correction,
# 2026-11-25 12:00 CT).
FROZEN_FLAGGED_QUEUE = {
    "code.bar_80",
    "code.block_inset",
    "code.reproduced_frac",
    "code.tool_angle",
    "param.ATTACH_CLOSEDNESS",
    "param.CABLE_CHAN_Z_HALF_M",
    "param.FRICTION_MULT_RANGE",
    "param.GRASP_DIST_M",
    "param.GRIP_MIN_EXCURSION_DEG",
    "param.JITTER",
    "param.RELEASE_CLOSEDNESS",
    "param.RING_THICK",
    "param.SECONDARY_THRESH_M",
    "param.SETTLE_S",
    "param.SLOT_SIZE",
}


def _load_manifest():
    assert MANIFEST_PATH.exists(), (
        f"criteria-provenance manifest missing: {MANIFEST_PATH}. "
        "Every frozen-criterion threshold must be classified there "
        "(see twin-replay/code/CRITERIA-PROVENANCE.md)."
    )
    return json.loads(MANIFEST_PATH.read_text())


def _norm(v):
    return json.loads(json.dumps(v))


def _param_literals():
    """Module-level ALL-CAPS constants in twin_params.py with literal values."""
    tree = ast.parse((CODE_DIR / "twin_params.py").read_text())
    out = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            try:
                val = ast.literal_eval(node.value)
            except Exception:
                continue
            if isinstance(val, (int, float, tuple, dict)) and not isinstance(val, bool):
                out[node.targets[0].id] = val
    return out


def _compare_float_literals(path, func_name=None):
    """Float literals used as direct comparison operands (thresholds).

    Integer operands are indices/counters and are out of scope by design.
    """
    tree = ast.parse(path.read_text())
    lits = set()

    def visit(node):
        for n in ast.walk(node):
            if isinstance(n, ast.Compare):
                for side in [n.left, *n.comparators]:
                    if isinstance(side, ast.Constant) and isinstance(side.value, float):
                        lits.add(side.value)
                    elif (
                        isinstance(side, ast.UnaryOp)
                        and isinstance(side.operand, ast.Constant)
                        and isinstance(side.operand.value, float)
                    ):
                        lits.add(side.operand.value)

    if func_name is None:
        visit(tree)
    else:
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                visit(node)
    return lits


def _all_numeric_literals_in_function(path, func_name):
    tree = ast.parse(path.read_text())
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            for n in ast.walk(node):
                if (
                    isinstance(n, ast.Constant)
                    and isinstance(n.value, (int, float))
                    and not isinstance(n.value, bool)
                ):
                    out.add(n.value)
    return out


def _entries(manifest, **match):
    return [
        e for e in manifest["entries"] if all(e.get(k) == v for k, v in match.items())
    ]


def test_every_twin_params_constant_is_classified():
    manifest = _load_manifest()
    by_name = {e["name"]: e for e in manifest["entries"]}
    missing = [n for n in _param_literals() if n not in by_name]
    assert not missing, (
        "twin_params.py constants with NO criteria-provenance manifest entry: "
        f"{sorted(missing)}. A numeric constant may not enter the frozen "
        "criteria silently — classify it in "
        "twin-replay/code/criteria-provenance.json (derivation or flag)."
    )


def test_param_values_match_manifest():
    manifest = _load_manifest()
    by_name = {e["name"]: e for e in manifest["entries"]}
    stale = []
    for name, val in _param_literals().items():
        entry = by_name.get(name)
        if (
            entry is not None
            and "value" in entry
            and _norm(entry["value"]) != _norm(val)
        ):
            stale.append(name)
    assert not stale, (
        f"manifest values no longer match twin_params.py for: {sorted(stale)}. "
        "A threshold changed without a provenance review."
    )


def test_manifest_patterns_match_exactly_once():
    manifest = _load_manifest()
    bad = []
    for e in manifest["entries"]:
        if "pattern" in e:
            text = (CODE_DIR / e["file"]).read_text()
            if text.count(e["pattern"]) != 1:
                bad.append(e["id"])
    assert not bad, (
        f"manifest patterns that no longer match exactly once in the criteria "
        f"source: {bad}. The criteria code changed under a recorded threshold — "
        "review and update the manifest entry."
    )


def test_compare_literals_are_covered():
    manifest = _load_manifest()
    problems = []
    for fname, func in [("replay.py", "evaluate"), ("run_validation.py", None)]:
        covered = set()
        for e in _entries(manifest, file=fname):
            covered.update(e.get("covers_literals", []))
        uncovered = _compare_float_literals(CODE_DIR / fname, func) - covered
        if uncovered:
            problems.append(f"{fname}: {sorted(uncovered)}")
    assert not problems, (
        "comparison threshold literals with no provenance entry — "
        + "; ".join(problems)
        + ". A new numeric threshold entered a criterion without a recorded "
        "derivation. Classify it in criteria-provenance.json (derive it, or "
        "flag it UNDERIVED with a pending disposition)."
    )


def test_evaluate_literal_census_is_covered():
    manifest = _load_manifest()
    covered = set()
    for e in _entries(manifest, file="replay.py"):
        covered.update(e.get("covers_literals", []))
    uncovered = (
        _all_numeric_literals_in_function(CODE_DIR / "replay.py", "evaluate") - covered
    )
    assert not uncovered, (
        f"numeric literals inside replay.evaluate() not covered by any "
        f"manifest entry: {sorted(uncovered)}. Every number in the scoring "
        "function must be classified (threshold with derivation/flag, or "
        "recorded as an incidental in code.evaluate_incidentals)."
    )


def test_scoring_thresholds_have_derivation_or_flag():
    manifest = _load_manifest()
    bad = []
    for e in manifest["entries"]:
        assert e["status"] in ALL_STATUSES, f"{e['id']}: unknown status {e['status']}"
        if e["kind"] not in GATED_KINDS:
            continue
        if e["status"] in DERIVED_STATUSES:
            if not e.get("derivation") or not e.get("source"):
                bad.append(
                    f"{e['id']} (claims {e['status']} but derivation/source is empty)"
                )
        elif e["status"] in FLAGGABLE_STATUSES:
            if not e.get("flagged") or not e.get("disposition"):
                bad.append(
                    f"{e['id']} ({e['status']} but not flagged with a disposition)"
                )
    assert not bad, "gated thresholds without derivation or flag: " + "; ".join(bad)


def test_flagged_entries_point_at_the_ruling_record():
    """The flagged queue is the honest pending list from the 2026-11-25
    postmortem; every flag must carry its disposition pointer so the queue
    can never silently rot."""
    manifest = _load_manifest()
    bad = [
        e["id"]
        for e in manifest["entries"]
        if e.get("flagged")
        and "ring-band-postmortem-20261125" not in e.get("disposition", "")
    ]
    assert (
        not bad
    ), f"flagged entries missing their postmortem disposition pointer: {bad}"


def test_flagged_queue_is_closed():
    """No NEW underived threshold may enter by being flagged.

    The postmortem's flagged queue is a closed, founder-visible list
    awaiting individual rulings — not an admission lane. An entry may
    leave the queue (derivation, physics-state conversion, founder
    ruling); nothing may join it.
    """
    manifest = _load_manifest()
    flagged = {e["id"] for e in manifest["entries"] if e.get("flagged")}
    newcomers = flagged - FROZEN_FLAGGED_QUEUE
    assert not newcomers, (
        f"flagged entries outside the frozen 2026-11-25 postmortem queue: "
        f"{sorted(newcomers)}. The flagged queue is CLOSED — a new numeric "
        "threshold needs a recorded derivation (measurement or founder "
        "decision), not a flag. If the founder has ruled on a genuinely "
        "new pending item, the ruling and the queue amendment are recorded "
        "in the postmortem record first."
    )


def test_scoring_quantities_declared_and_engine_resolved_bands_derived():
    """The upstream gate (founder correction, 2026-11-25 12:00 CT).

    Every SCORING entry must declare the quantity it gates. A numeric
    band on a quantity the physics engine already resolves (heights,
    positions, gaps, angles, settle/contact outcomes) is a hard failure
    unless the number carries a recorded derivation — the only
    exceptions are the frozen postmortem flags, each awaiting its own
    founder ruling (test_flagged_queue_is_closed keeps that set from
    growing).
    """
    manifest = _load_manifest()
    problems = []
    for e in manifest["entries"]:
        if e["kind"] != "SCORING":
            continue
        q = e.get("quantity")
        if q not in QUANTITY_CLASSES:
            problems.append(
                f"{e['id']}: quantity {q!r} is not one of "
                f"{sorted(QUANTITY_CLASSES)} — declare what this threshold "
                "gates (engine-resolved / input-geometry / "
                "agreement-statistic / physics-state)"
            )
            continue
        if q == "engine-resolved" and e["status"] in FLAGGABLE_STATUSES:
            if e["id"] not in FROZEN_FLAGGED_QUEUE:
                problems.append(
                    f"{e['id']}: underived band on an engine-resolved "
                    f"quantity ({e['name']}) with status {e['status']} — "
                    "a criterion may not band a quantity the physics "
                    "engine resolves without a recorded derivation "
                    "(measurement or founder decision); read the engine's "
                    "resolved state instead"
                )
        if q == "input-geometry" and e["status"] in DERIVED_STATUSES:
            if not e.get("derivation") or not e.get("source"):
                problems.append(
                    f"{e['id']}: input-geometry entry claims {e['status']} "
                    "but cites no measurement (derivation/source empty)"
                )
    assert not problems, "engine-resolved gate failures:\n" + "\n".join(problems)


def test_flagged_queue_summary(capsys):
    """Visibility: print the currently flagged (underived) thresholds.

    This test passes by design — the frozen flagged entries are the
    recorded, honest state pending individual founder dispositions, and
    test_flagged_queue_is_closed guarantees no entry has joined the
    queue since the postmortem. The printed list keeps the queue
    visible in every CI log.
    """
    manifest = _load_manifest()
    flagged = [e for e in manifest["entries"] if e.get("flagged")]
    with capsys.disabled():
        print(
            f"\n[criteria-provenance] {len(flagged)} FLAGGED underived thresholds pending founder disposition:"
        )
        for e in flagged:
            print(f"  - {e['id']}: {e['name']} = {e.get('value')} [{e['status']}]")
    assert all(e.get("disposition") for e in flagged)
