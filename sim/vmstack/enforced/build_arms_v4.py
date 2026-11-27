"""Build the v4 arm XMLs (corrections lane, 2026-11-27; founder order
2026-11-27 05:27 CT: "Double fact check and apply corrections to the sim
model").

Change vs v3 (ONLY this): every sts3215 forcerange — the class default and
all six per-actuator overrides — is set to the stall torque of the
STS3215-C001, the servo the standard SO-101 follower kit ships:

  DERIVATION (double-fact-checked, corrections-20261127/fact_check_record.md):
  - Variant: TheRobotStudio SO-ARM100 README (the bill of materials the
    LeRobot SO-101 documentation links to), "Parts for One Follower Arm":
    6x "STS3215 Servo 7.4V, 1/345 gear (C001)". The 12 V C018 is an
    optional upgrade, not the standard kit.
  - Stall torque: Feetech STS3215 product specification, edition A/0
    (2020-04-10), electrical specification 5-4: 19.5 kg.cm at 7.4 V
    (16.5 kg.cm at 6 V). Corroborated by Seeed Studio and OpenELAB C001
    listings (19.5 kg.cm stall / 5 kg.cm rated at 7.4 V).
  - Conversion: 19.5 kg.cm x 0.0980665 N.m/(kg.cm) = 1.9123 N.m.
  The v3 values were: class default +/-2.94 N.m (= C018 stall, the wrong
  variant for the standard kit) and per-actuator overrides +/-3.35 N.m
  (= 34.2 kg.cm, exceeding every published STS3215 variant; underived).

Everything else is byte-identical to v3 (geometry, joints, gains, sites,
cameras). This changes the seal tuple (arm element) for everything
downstream: every (scene, arm, ...) tuple citing a v3 arm is superseded.
refetch/arm.xml is an upstream fetched artifact and is NOT modified; the
correction lives here and in the v4 XMLs.

ADDENDUM (2026-11-27, actuation-corrections lane): the builder also
inserts <contact><exclude body1="base" body2="shoulder"/></contact>
(with its derivation comment) into the two PADDED outputs. The bolted
base x shoulder interface interpenetrates -27.87 mm by construction and,
left live, carried 12 solver contacts (~966 kN at the Run-9 witness)
injecting phantom gripper-site accelerations of 63-408 g
(neon-defect-verification-20261127/VERDICT.md, Defect 1 CONFIRMED;
contact-set diff in corrections-actuation-20261127/contact_set_diff.json
proves the exclude removes exactly those contacts and nothing else).
arm_tool_v4.xml is NOT given the exclude here: no scenes-v4 scene
includes it and it is outside that lane's tasked scope; its base x
shoulder pair shows the same construction (flagged in the lane report).

Inputs : arm_padded_v3.xml, arm_tool_v3.xml, arm_padded_v3cam.xml
Outputs: arm_padded_v4.xml, arm_tool_v4.xml, arm_padded_v4cam.xml
         corrections-20261127/arm_build_record_v4.json
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).parent
REC = HERE / "corrections-20261127"
OLD_ACT = 'forcerange="-3.35 3.35"'
OLD_CLS = 'forcerange="-2.94 2.94"'
NEW = 'forcerange="-1.9123 1.9123"'
HEADER = ("<!-- arm v4 (2026-11-27, corrections lane): sts3215 forcerange "
          "corrected to the STS3215-C001 stall torque 1.9123 N.m "
          "(19.5 kg.cm @ 7.4 V, Feetech spec A/0 2020-04-10; C001 = the "
          "standard SO-101 follower-kit servo per the SO-ARM100 BOM). "
          "v3 carried +/-3.35 N.m per-actuator overrides (underived; above "
          "every published variant) on a +/-2.94 N.m class default (C018 "
          "stall). Geometry and all other elements byte-identical to v3. "
          "Derivation: corrections-20261127/fact_check_record.md -->\n")

PAIRS = [("arm_padded_v3.xml", "arm_padded_v4.xml"),
         ("arm_tool_v3.xml", "arm_tool_v4.xml"),
         ("arm_padded_v3cam.xml", "arm_padded_v4cam.xml")]

EXCLUDE_DSTS = {"arm_padded_v4.xml", "arm_padded_v4cam.xml"}
CONTACT_BLOCK = ("  <!-- actuation-corrections lane (2026-11-27): the bolted base x shoulder\n"
                 "       interface (pan motor inside the base housing) interpenetrates\n"
                 "       -27.87 mm by construction and, left live in the collision set,\n"
                 "       carried 12 solver contacts (~966 kN total normal force at the\n"
                 "       Run-9 witness) injecting phantom gripper-site accelerations of\n"
                 "       63-408 g into every step (neon-defect-verification-20261127/\n"
                 "       VERDICT.md, Defect 1 CONFIRMED). The pair is mechanism-internal\n"
                 "       (enforcement_scan already exempts it as such); the exclude below\n"
                 "       removes it from the collision set. Geometry unchanged. -->\n"
                 "  <contact>\n"
                 "    <exclude body1=\"base\" body2=\"shoulder\"/>\n"
                 "  </contact>\n")


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


record = {"change": "forcerange -> +/-1.9123 N.m (C001 stall) on class "
                    "default + all per-actuator overrides; header comment added; "
                    "actuation-corrections addendum (2026-11-27): base x shoulder "
                    "contact exclude inserted into the two padded outputs",
          "derivation": "19.5 kg.cm x 0.0980665 = 1.9123 N.m; variant C001 "
                        "per SO-ARM100 BOM (follower: 6x STS3215 7.4V 1/345 "
                        "C001); stall per Feetech STS3215 spec A/0 2020-04-10",
          "files": {}}
for src, dst in PAIRS:
    text = (HERE / src).read_text()
    n_act = text.count(OLD_ACT)
    n_cls = text.count(OLD_CLS)
    assert n_act == 6, (src, "actuator overrides", n_act)
    assert n_cls == 1, (src, "class default", n_cls)
    out = text.replace(OLD_ACT, NEW).replace(OLD_CLS, NEW)
    # insert header comment after the <mujoco ...> opening line
    lines = out.split("\n")
    for i, ln in enumerate(lines):
        if ln.strip().startswith("<mujoco"):
            lines.insert(i + 1, HEADER.rstrip("\n"))
            break
    out = "\n".join(lines)
    if dst in EXCLUDE_DSTS:
        assert "<contact>" not in out, (dst, "contact block already present")
        assert "  <worldbody>" in out, (dst, "worldbody anchor missing")
        out = out.replace("  <worldbody>", CONTACT_BLOCK + "  <worldbody>", 1)
    (HERE / dst).write_text(out)
    record["files"][dst] = {"source": src, "source_sha256": sha256(HERE / src),
                            "sha256": sha256(HERE / dst),
                            "actuator_overrides_changed": n_act,
                            "class_defaults_changed": n_cls}
    print(dst, record["files"][dst]["sha256"][:16], "act", n_act, "cls", n_cls)

(REC / "arm_build_record_v4.json").write_text(json.dumps(record, indent=1))
print("record written")
