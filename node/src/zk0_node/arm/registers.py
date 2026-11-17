"""The telemetry contract's channel table (SSOT decision 55).

The bridge may read ONLY these registers from the arm — the channels
the real Feetech STS3215 reports through LeRobot (contract
`zk0-telemetry-contract-20261116/contract.md` section 2, verified
there from LeRobot's feetech tables + the rustypot STS3215 definition
LeRobot 0.7.0 loads). Ground truth that has no register — contact
force, object pose, sim state — has no entry here and therefore no
way through the bridge: the read set is this table, enforced by
`ArmBridge.read_contract_state` and pinned by a unit test.

Register addresses/sizes/scales are manufacturer/LeRobot source
facts as recorded in the contract; the two conversions the contract
marks UNVERIFIED (Present_Velocity physical units, Present_Current
sign) are carried as raw counts here too — the bridge reports them,
it does not invent units for them.
"""

# name -> (address, size_bytes, scale/unit note)
CONTRACT_CHANNELS = {
    "Present_Position": (56, 2, "encoder counts, 4096/rev; LeRobot "
                                "normalizes to degrees in the "
                                "calibrated frame"),
    "Present_Velocity": (58, 2, "raw counts, sign-magnitude bit 15; "
                                "physical-unit conversion UNVERIFIED "
                                "(contract section 2) — reported raw"),
    "Present_Load": (60, 2, "firmware control-output duty: magnitude "
                            "0-1000 counts = 0-100% of the torque-limit "
                            "scale (0.1%/count), direction bit 10; "
                            "NOT a measured torque (contract section 4: "
                            "certified as a contact detector, not a "
                            "force regulator)"),
    "Present_Voltage": (62, 1, "0.1 V per count"),
    "Present_Temperature": (63, 1, "1 degC per count"),
    "Status": (65, 1, "error bitmask: bit 0 voltage, 1 sensor, "
                      "2 temperature, 3 current, 5 overload"),
    "Moving": (66, 1, "boolean: 0 stationary, 1 moving"),
    "Present_Current": (69, 2, "6.5 mA per count; sign convention "
                               "UNVERIFIED (contract section 2) — "
                               "reported as read"),
    "Goal_Position": (42, 2, "the command channel: LeRobot "
                             "send_action sync-writes it; controller-"
                             "side state, read back for the record"),
}

#: The feedback channels read every control tick (Goal_Position is
#: the node's own last command, tracked by the bridge itself).
FEEDBACK_CHANNELS = tuple(k for k in CONTRACT_CHANNELS
                          if k != "Goal_Position")

#: Status bit names (contract section 2, register reference).
STATUS_BITS = {0: "voltage", 1: "sensor", 2: "temperature",
               3: "current", 5: "overload"}


def decode_status(value: int) -> list:
    """Names of the set fault bits in a Status register value."""
    return [name for bit, name in sorted(STATUS_BITS.items())
            if int(value) & (1 << bit)]
