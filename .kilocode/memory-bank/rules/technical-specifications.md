# Technical Specifications and Official Documentation

## SmolVLA Model Specifications

### Model Architecture
- **Model Size**: 450M parameters total
- **Vision-Language Model (VLM)**: SmolVLM2 backbone
  - Vision Encoder: SigLIP
  - Language Decoder: SmolLM2
- **Action Expert**: Flow matching transformer (~100M parameters)
- **Design Choices**:
  - Visual token reduction (64 tokens per frame)
  - Layer skipping (half of VLM layers for faster inference)
  - Interleaved cross and self-attention blocks

### Training and Inference
- **Training Data**: Community-shared datasets under `lerobot` tag
- **Supported Hardware**: Consumer GPUs, CPUs, even MacBooks
- **Asynchronous Inference**: 30% faster response, 2× task throughput
- **Real-world Performance**: SO-100 and SO-101 compatibility

### Installation Requirements
```bash
# Install LeRobot with SmolVLA dependencies
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e ".[smolvla]"
```

### Usage Examples

#### Finetune Pretrained Model
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

#### Train from Scratch
```bash
python lerobot/scripts/train.py \
  --dataset.repo_id=lerobot/svla_so100_stacking \
  --batch_size=64 \
  --steps=200000 \
  --output_dir=outputs/train/my_smolvla \
  --job_name=my_smolvla_training \
  --policy.device=cuda \
  --wandb.enable=true
```

## Flower Framework Specifications

### Framework Version
- **Current Version**: Flower 1.20.0
- **Framework Type**: Federated Learning Framework
- **Architecture**: Client-Server with Deployment Engine

### Supported ML Frameworks
- PyTorch
- TensorFlow
- JAX
- MLX
- 🤗 Transformers
- PyTorch Lightning
- scikit-learn
- XGBoost
- fastai
- Pandas

### Key Components
- **ClientApp**: Defines client-side federated learning logic
- **ServerApp**: Defines server-side aggregation logic
- **Strategies**: Federated learning algorithms (FedAvg, FedProx, etc.)
- **Mods**: Built-in modifications for differential privacy, compression, etc.

### Execution Modes
- **Simulation Mode**: Local testing and development
- **Deployment Mode**: Production distributed execution
- **GPU Support**: Configurable GPU resource allocation

### Installation and Setup
```bash
# Install Flower
pip install flwr[simulation]

# For latest features
pip install flwr[simulation] --pre
```

### Basic Usage
```bash
# Run simulation
flwr run .

# Run with GPU federation
flwr run . local-simulation-gpu

# Override configuration
flwr run . local-simulation-gpu --run-config "num-server-rounds=5 fraction-fit=0.1"
```

## Integration Requirements

### SmolVLA + Flower Integration
- **Framework Compatibility**: Flower 1.20.0 with Ray 2.31.0
- **Dataset Format**: Flower Datasets for partitioning
- **Model Loading**: Direct integration with LeRobot SmolVLA models
- **Federated Dataset**: FederatedLeRobotDataset for distributed data

### Key Integration Points
1. **Client Implementation**: Extend NumPyClient for SmolVLA
2. **Server Strategy**: Use FedAvg or custom strategies
3. **Data Partitioning**: LeRobotDatasetPartitioner for episode-based splitting
4. **Model Aggregation**: Flower's parameter aggregation mechanisms

## Performance Benchmarks

### SmolVLA Performance
- **SO-100 Success Rate**: 78.3% (with community pretraining)
- **SO-101 Generalization**: Strong transfer capabilities
- **Simulation Benchmarks**: Matches/exceeds larger VLAs on LIBERO, Meta-World
- **Real-world Tasks**: Pick-place, stacking, sorting, tool manipulation

### Flower Performance
- **Scalability**: Supports 10+ clients in simulation
- **Communication Efficiency**: Optimized parameter transmission
- **GPU Utilization**: Efficient resource allocation
- **Memory Management**: Streaming and batch processing

## Official Documentation References

### SmolVLA Resources
- **Blog Post**: https://huggingface.co/blog/smolvla
- **Model Hub**: https://huggingface.co/lerobot/smolvla_base
- **Paper**: https://huggingface.co/papers/2506.01844
- **Code**: https://github.com/huggingface/lerobot

### Flower Resources
- **Documentation**: https://flower.ai/docs/framework/
- **Framework Hub**: https://flower.ai
- **GitHub**: https://github.com/adap/flower
- **Examples**: https://github.com/adap/flower/tree/main/examples

## Compliance Requirements

### Technical Standards
- **Python Version**: 3.8+ (recommended 3.10)
- **PyTorch Version**: Compatible with SmolVLA requirements
- **CUDA Version**: 11.0+ for GPU support
- **Memory Requirements**: 8GB+ RAM, 4GB+ VRAM for GPU

### Dataset Standards
- **Format**: LeRobot format with lerobot tag
- **Annotation**: Clear task descriptions (max 30 characters)
- **Camera Views**: Standardized naming (OBS_IMAGE_1, OBS_IMAGE_2, etc.)
- **Frame Rate**: 30 FPS for SO-100/SO-101

### Quality Standards
- **Test Coverage**: 80% minimum for new code
- **Documentation**: Complete API documentation
- **Reproducibility**: Seeds for all experiments
- **Performance**: Regular benchmarking against baselines

## Implementation Guidelines

### Development Workflow
1. **Environment Setup**: Use conda environment "zk0"
2. **Dependency Installation**: Follow official installation guides
3. **Code Structure**: Follow LeRobot and Flower patterns
4. **Testing**: Comprehensive unit and integration tests
5. **Documentation**: Inline documentation and README updates

### Deployment Considerations
1. **Hardware Requirements**: Match official recommendations
2. **Network Configuration**: Proper TLS and authentication setup
3. **Monitoring**: Logging and performance monitoring
4. **Scalability**: Plan for multiple clients and rounds

### Maintenance Requirements
1. **Version Updates**: Stay current with SmolVLA and Flower releases
2. **Security Patches**: Apply security updates promptly
3. **Performance Tuning**: Regular optimization based on benchmarks
4. **Community Engagement**: Follow updates and contribute back

---

**Last Updated**: 2025-09-03
**Based on Official Documentation**: SmolVLA blog, model hub, and Flower framework docs
**Version**: 1.0.0