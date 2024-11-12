# zk0 [zee-ˈkō]


zk0 is an Open Source humanoid built collaboratively by a community of contributors of code, compute and data. 

<img src="https://github.com/user-attachments/assets/9dd876a0-6668-4b9f-ad0d-94a540353418" width=100>


# Why

Why do we need yet another humanoid project? 
As of 2024 AI technology has [advanced enough to speculate](https://x.com/elonmusk/status/1786367513137233933) that within a decade most people will have their own humanoid buddy. By some estimates humanoids will become $100 Trillion market (10B humanoids * $10,000 per unit).

[Today's leading closed source humanoid](https://x.com/Tesla_Optimus/status/1846294753144361371) is trained on [100,000 GPU farm](https://nvidianews.nvidia.com/news/spectrum-x-ethernet-networking-xai-colossus) with real world data collected from millions of cars labeled by able human drivers. 
This is an enormous scale of compute and data that is hard to compete with as a centrazlied entity. 
However it would be interesting to see if a decentralized approach might produce useful results over time.
On the chance that proprietary humanoids ever go rogue, it would be nice to have open source alternatives.

# How

zk0 is composed of several major building blocks:
- Generative AI: 
  * [HuggingFace LeRobot](https://huggingface.co/lerobot) for the Open Source 3D printed robot parts and end-to-end vision language action models.
- Federated Learning: 
  * [Flower](https://flower.ai/) for collaborative training of AI models
- Zero Knowledge Proofs: 
  * [EZKL](https://ezkl.xyz/) for verification of contributed model checkpoints trained on local data.


# Directory Structure

```shell
zk0
│
├── lerobot             # clone of remote lerobot repo: 
│                       #    https://github.com/huggingface/lerobot.git
│
├── federate            # federated learning layer
│   │
│   └── federatelerobot_example/
│                       # Federated Learning Example with Flower and LeRobot Diffusion PushT task
│
└── README.md           # This README file
```

