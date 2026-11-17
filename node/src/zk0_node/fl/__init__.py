"""Federated rounds for the zk0 Node (spec sections 2.3, 5.3).

Two scopes share this code: the operator-run LOCAL network (section
2.3 — the aggregator runs on the operator's own machine/LAN, no
hosted connector, no internet) and, later, the hosted staging/prod
rounds. v0.1 slice 2 ships the local network end-to-end.
"""
