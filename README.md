# zk0 [zee-ˈkō]

An Open Source humanoid trained collaboratively by a community of builders.

<img src="https://github.com/user-attachments/assets/9dd876a0-6668-4b9f-ad0d-94a540353418" width=300>

# Why

AI technology has [advanced enough to speculate](https://x.com/elonmusk/status/1786367513137233933) that within a decade most people will have their own humanoid buddy. By some estimates humanoids will become $100 Trillion market (5B humanoids * $20,000 per unit).

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

# Quickstart Example

[Here](https://github.com/ivelin/zk0/tree/federate-pusht-gym/federate) is a complete example demonstrating federated learning with the LeRobot PushT dataset. Shows client-server architecture, data partitioning, and model update aggregation. 

# Contribute

Following is the high level directory structure of the [code repository](https://github.com/ivelin/zk0/tree/main). Jump in, try the example and explore. Contributors are welcome!

```shell
zk0
│
├── lerobot             # clone of remote lerobot repo: 
│                       #    https://github.com/huggingface/lerobot.git
│
├── federate            # federated learning layer
│   │
│   └── lerobot_example/
│                       # Federated Learning Example with Flower and LeRobot Diffusion PushT task
│
└── README.md           # This README file
```

# Social Media


<blockquote class="twitter-tweet"><p lang="en" dir="ltr">It&#39;s time for a complete open-source stack for autonomy/robotics plus distributed learning. The first step is here: <a href="https://twitter.com/LeRobotHF?ref_src=twsrc%5Etfw">@LeRobotHF</a> + <a href="https://twitter.com/flwrlabs?ref_src=twsrc%5Etfw">@flwrlabs</a> LFG 🚀<a href="https://twitter.com/comma_ai?ref_src=twsrc%5Etfw">@comma_ai</a> <a href="https://twitter.com/wayve_ai?ref_src=twsrc%5Etfw">@wayve_ai</a> <a href="https://twitter.com/Figure_robot?ref_src=twsrc%5Etfw">@Figure_robot</a> <a href="https://twitter.com/Tesla?ref_src=twsrc%5Etfw">@Tesla</a> <a href="https://t.co/8O8cSD3SbO">https://t.co/8O8cSD3SbO</a> <a href="https://t.co/oVUOLTvwzm">https://t.co/oVUOLTvwzm</a></p>&mdash; nic lane (@niclane7) <a href="https://twitter.com/niclane7/status/1879597539676266726?ref_src=twsrc%5Etfw">January 15, 2025</a></blockquote> 


<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Open-source robots just got a boost. Frameworks like Flower FL enable faster learning, efficient scaling, and continuous knowledge sharing using real-world data. <a href="https://t.co/j8VSGiWF0W">https://t.co/j8VSGiWF0W</a></p>&mdash; 𝚐𝔪𝟾𝚡𝚡𝟾 (@gm8xx8) <a href="https://twitter.com/gm8xx8/status/1879633368427761785?ref_src=twsrc%5Etfw">January 15, 2025</a></blockquote> 


<blockquote class="twitter-tweet"><p lang="en" dir="ltr">We are not so far from a future where robots will be constantly learning by interacting with humans and their environments.<br><br>Frameworks like <a href="https://twitter.com/flwrlabs?ref_src=twsrc%5Etfw">@flwrlabs</a> will enable these robots to learn much faster by continuously sharing their learnings.<br><br>We really live in a sci-fi movie 😅 <a href="https://t.co/kAz3xZ2qvB">https://t.co/kAz3xZ2qvB</a></p>&mdash; Remi Cadene (@RemiCadene) <a href="https://twitter.com/RemiCadene/status/1879592068865282227?ref_src=twsrc%5Etfw">January 15, 2025</a></blockquote>


<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Federated Learning Meets Robotics: 🤖 LeRobot + 🌼 Flower<br><br>This demo demonstrates how robots in remote environments can collaboratively train an AI model using their local data, which is then aggregated into a shared model. <br><br>In this quickstart, you will train a Diffusion policy… <a href="https://t.co/i32MkbxoPW">pic.twitter.com/i32MkbxoPW</a></p>&mdash; Flower (@flwrlabs) <a href="https://twitter.com/flwrlabs/status/1879571258532036739?ref_src=twsrc%5Etfw">January 15, 2025</a></blockquote> 

<iframe width="560" height="315" src="https://www.youtube.com/embed/fwAtTOZttWo?si=3d50oQtSvMvGxNg6" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>


# Share 

![image](https://github.com/user-attachments/assets/e03913ec-62a0-4b05-a286-6fc18dfd433f)


