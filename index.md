---
title: zk0 - Decentralized Robotics AI
description: Open source federated learning for SmolVLA on SO-100 datasets with ZK proofs and blockchain incentives.
permalink: /
og_image: /docs/images/lerobot_flower_splash.png
---

# Welcome to zk0: Collaborative AI for Humanoid Robots

zk0 [zee-ˈkō] is like a big family where robot builders from around the world teach their robots new tricks together, without sharing private family photos or videos. Imagine your home robot learning to fold clothes or pick up toys by watching what other robots do in their homes—safely and privately!

[![Discover zk0: LeRobot + Flower Integration](docs/images/lerobot_flower_splash.png)](https://flower.ai/docs/examples/quickstart-lerobot.html)

## 🚀 Latest Milestone: SmolVLA Federated Model Released

Train your robot with our community-trained model on Hugging Face:

- **Model**: [ivelin/zk0-smolvla-fl](https://huggingface.co/ivelin/zk0-smolvla-fl) (250 rounds, final loss 0.495)
- **Key Features**: FedProx aggregation, dynamic scheduling, SO-100 tasks
- **Load It**: 
  ```python
  from transformers import AutoModel
  model = AutoModel.from_pretrained("ivelin/zk0-smolvla-fl")
  ```

[Explore the full release & analysis](README.html#latest-model-release) | [WandB Dashboard](https://wandb.ai/ivelin-eth/zk0/runs/zk0-sim-fl-run-2025-10-20_23-44-35)

## Join the Movement

### Upcoming: DevConnect Meetup
Register for our Robotics AI session in Buenos Aires, Nov 18, 2025: [Federated Learning with ZK Proofs](https://lu.ma/embed/event/evt-udINVLo325xhKsG/simple).

[Watch Past Talks](https://www.youtube.com/embed/fwAtTOZttWo?si=3d50oQtSvMvGxNg6) | [Join Discord](https://discord.gg/dhMnEne7RP)

### Why Contribute?
Decentralize humanoid AI: Train on your data, earn incentives via ZK proofs & blockchain. From $100T market potential to open alternatives—be part of it.

[Read the Vision](README.html#why) | [Roadmap Teaser](README.html#roadmap)

## Get Started in Minutes

1. Clone: `git clone https://github.com/ivelin/zk0`
2. Setup: `conda create -n zk0 python=3.10 && pip install -e .`
3. Run Simulation: `./train-fl-simulation.sh`

[Full Installation Guide](/docs/INSTALLATION.html) | [Node Operator Setup](/docs/NODE-OPERATORS.html)

## Explore zk0

- [Project Overview](/README.html)
- [Architecture](/docs/ARCHITECTURE.html)
- [Contribute](/CONTRIBUTING.html)
- [License](/LICENSE)

**Repository**: [GitHub/ivelin/zk0](https://github.com/ivelin/zk0)

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "SoftwareApplication",
  "name": "zk0",
  "description": "Open source federated learning for decentralized robotics AI",
  "applicationCategory": "DeveloperApplication",
  "offers": {"@type": "Offer", "price": "0"},
  "author": {"@type": "Person", "name": "ivelin.eth"},
  "url": "https://zk0.bot",
  "sameAs": ["https://github.com/ivelin/zk0"]
}
</script>

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "FAQPage",
  "mainEntity": [{
    "@type": "Question",
    "name": "What is zk0?",
    "acceptedAnswer": {
      "@type": "Answer",
      "text": "zk0 enables collaborative training of SmolVLA models via federated learning on SO-100 datasets, with ZK proofs for privacy and blockchain for incentives."
    }
  }]
}
</script>