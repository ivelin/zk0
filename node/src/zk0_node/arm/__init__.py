"""Arm bridge package (build slice 3): the SO-101 hardware path.

bridge.ArmBridge is the node's one path to the arm; bus.py holds
the adapter seam (real LeRobot follower / MOCK), registers.py the
telemetry contract's channel table (the only registers the bridge
may read), safety_layer.py the spec section 4 hard safety layer
every action passes before it reaches the bus.
"""
