"""Soak smoke: real client fit path on stanl1y/record_60_simple (new hub-v3.0
client), mirroring smoke_a3.py exactly (epochs=2 -> 2 steps, eval_batches=1),
plus a dtype assertion proving the Stage B fp32 revert (policy must be bf16)."""
import sys
import torch
from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO")
sys.path.insert(0, "/home/openclaw/muse-work/zk0-repro-stageA-20261121")
from src.common.utils import load_lerobot_dataset
from src.training.model_utils import get_model
from src.training.train import train
from src.training.evaluation import test

device = torch.device("cuda:0")
dataset = load_lerobot_dataset("stanl1y/record_60_simple")
loader = torch.utils.data.DataLoader(dataset, num_workers=0, batch_size=16, shuffle=True, pin_memory=False, drop_last=True)
policy = get_model(dataset.meta)
dtypes = {p.dtype for p in policy.parameters()}
print("POLICY_DTYPES:", dtypes)
assert dtypes == {torch.bfloat16, torch.float32}, f"unexpected dtype set after revert: {dtypes}"
policy.to(device)
metrics = train(net=policy, trainloader=loader, epochs=2, batch_size=16, device=device,
                global_params=None, fedprox_mu=0.01, initial_lr=1e-3, partition_id=2, round_num=1)
print("TRAIN_METRICS:", metrics)
loss, num_examples, eval_metrics = test(policy, device, batch_size=16, eval_batches=1, dataset=dataset)
print("EVAL_RESULT:", loss, num_examples, eval_metrics)
assert eval_metrics["successful_batches"] >= 1, "eval: no successful batch"
print("SMOKE_OK")
