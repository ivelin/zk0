#!/bin/bash
cd lerobot
python lerobot/scripts/train.py \
  resume=true wandb.enable=true wandb.project=zk0 training.offline_steps=100000 \
  hydra.run.dir=outputs/train/2024-11-05/11-43-17_pusht_diffusion_default/ 
