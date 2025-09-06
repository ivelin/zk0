# SmolVLA Model Specifications

**Created**: 2025-09-06
**Last Updated**: 2025-09-06
**Version**: 1.0.0
**Author**: Kilo Code

## Model Architecture
- **Model Size**: 450M parameters total
- **Vision-Language Model (VLM)**: SmolVLM2 backbone
  - Vision Encoder: SigLIP
  - Language Decoder: SmolLM2
- **Action Expert**: Flow matching transformer (~100M parameters)
- **Design Choices for Efficiency and Robustness**:
  - Visual token reduction (64 tokens per frame using PixelShuffle)
  - Layer skipping (half of VLM layers for faster inference)
  - Interleaved cross and self-attention blocks
  - Flow matching objective for continuous action prediction
  - Reduced hidden size (75% of VLM's) for lightweight deployment

## Training and Inference
- **Training Data**: Community-shared datasets under `lerobot` tag
- **Supported Hardware**: Consumer GPUs, CPUs, even MacBooks
- **Asynchronous Inference**: 30% faster response, 2× task throughput
- **Real-world Performance**: SO-100 and SO-101 compatibility

## Installation Requirements
```bash
# Install LeRobot with SmolVLA dependencies from PyPI
pip install 'lerobot[smolvla]'
```

## Usage Examples

### Finetune Pretrained Model
```bash
python lerobot/scripts/train.py \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=lerobot/svla_so100_stacking \
  --batch_size=64 \
  --steps=20000 \
  --output_dir=outputs/train/my_smolvla \
  --job_name=my_smolvla_training \
  --policy.device=cuda \
  --wandb.enable=true
```

### Train from Scratch
```bash
python lerobot/scripts/train.py \
  --dataset.repo_id=lerobot/svla_so100_stacking \
  --batch_size=64 \
  --steps=200000 \
  --output_dir=outputs/train/my_smolvla \
  --job_name=my_smolvla_training \
  --policy.device=cuda \
  --wandb.enable=true