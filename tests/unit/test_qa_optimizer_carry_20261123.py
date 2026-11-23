"""Q-A AdamW carry: optimizer state must persist across a round boundary.

Functional: save via the production helper after 10 steps, load into a
fresh optimizer built with the next round's scheduled LR; the AdamW step
counter must read 10 and the scheduled LR must win over the saved one.
Wiring: the three touched modules must route optimizer_state_path.
"""
import pathlib

import torch

from src.training.training_setup import load_optimizer_state, save_optimizer_state

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _make_opt(lr):
    torch.manual_seed(1234)
    model = torch.nn.Linear(4, 2)
    opt = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-10
    )
    return model, opt


def _steps(opt):
    return [float(s["step"]) for s in opt.state_dict()["state"].values()]


def test_optimizer_state_persists_across_round_boundary(tmp_path):
    model, opt = _make_opt(lr=1e-4)
    x = torch.randn(3, 4)
    y = torch.randn(3, 2)
    for _ in range(10):
        opt.zero_grad()
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        opt.step()
    assert _steps(opt) and all(s == 10 for s in _steps(opt))

    state_path = tmp_path / "client_0" / "optimizer_state.pt"
    save_optimizer_state(opt, state_path)
    assert state_path.exists()
    assert not pathlib.Path(str(state_path) + ".tmp").exists()

    # Round 2: fresh optimizer, different scheduled LR.
    _model2, opt2 = _make_opt(lr=5e-5)
    assert load_optimizer_state(opt2, state_path) is True
    assert all(s == 10 for s in _steps(opt2))
    # Scheduled LR re-applied after load (saved 1e-4 must not ride in).
    assert all(g["lr"] == 5e-5 for g in opt2.param_groups)

    # Missing file is a clean no-op (round 1 of a fresh run).
    _m3, opt3 = _make_opt(lr=1e-4)
    assert load_optimizer_state(opt3, tmp_path / "nope.pt") is False


def test_carry_wiring_present():
    ts = (ROOT / "src/training/training_setup.py").read_text()
    tr = (ROOT / "src/training/train.py").read_text()
    cc = (ROOT / "src/client/client_core.py").read_text()
    assert "optimizer_state_path" in ts
    assert "optimizer_state_path" in tr
    assert "optimizer_state_path" in cc
    assert "self.client_id" in cc  # per-client state path derivation
