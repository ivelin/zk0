---
layout: default
title: zk0 - Decentralized Robotics AI
description: Open source federated learning for SmolVLA on SO-100 datasets with ZK proofs and blockchain incentives.
permalink: /
---

<div class="hero">
  <div class="hero-content">
    <h1>Welcome to zk0</h1>
    <p>Decentralized AI for the next generation of helpful robots</p>
  </div>
</div>

<div class="intro">
  <p>Imagine teaching robots to help around the house—like picking up toys or sorting laundry—but without sharing family videos from different homes. zk0 makes this possible by letting robots learn skills together safely and privately, just like neighbors sharing tips without showing their own photos. It's open-source technology that brings smarter, more helpful robots to everyone while keeping your data secure.</p>
</div>

<div class="concept-diagram" style="text-align: center;">
  <img src="/assets/images/zk0-fl-concept.png" alt="zk0 Federated Learning Concept Diagram" style="max-width: 80%; height: auto;" />
</div>

<h1>Discover zk0</h1>

<p>zk0 is an open-source federated learning platform for training advanced vision-language-action models on real-world robotics datasets. Currently focused on SmolVLA with SO-100 manipulation tasks, zk0 aspires to evolve with the latest open-source ML models and advanced humanoid embodiments as they become available, enabling broader decentralized robotics AI development.</p>

<div class="features">
  <div class="feature-card">
    <h3>🌸 Federated Learning with Flower</h3>
    <ul>
      <li>Secure, scalable training without sharing raw data</li>
      <li>Privacy-preserving model updates across distributed clients</li>
      <li>Handles heterogeneous data and devices</li>
    </ul>
  </div>
  
  <div class="feature-card">
    <h3>🤖 SmolVLA Integration</h3>
    <ul>
      <li>Efficient 450M parameter models for robotics AI</li>
      <li>Vision-language-action capabilities for manipulation tasks</li>
      <li>Optimized for consumer GPUs and real-world deployment</li>
    </ul>
  </div>
  
  <div class="feature-card">
    <h3>📦 SO-100 Datasets</h3>
    <ul>
      <li>Real-world manipulation tasks (pick-place, stacking, tool use)</li>
      <li>Diverse robotics scenarios for robust skill learning</li>
      <li>Community-driven dataset expansion</li>
    </ul>
  </div>
  
  <div class="feature-card">
    <h3>🚀 Production-Ready</h3>
    <ul>
      <li>Docker-based deployment for easy scaling</li>
      <li>zk0bot CLI for node operators and monitoring</li>
      <li>Advanced scheduling for optimal convergence</li>
    </ul>
  </div>
</div>

<div class="get-started">
  <h2>Get Started Today</h2>
  <p>Join the decentralized robotics revolution</p>
  <a href="/docs/INSTALLATION.html">For Developers</a>
  <a href="/docs/NODE-OPERATORS.html">For Node Operators</a>
  <a href="https://github.com/ivelin/zk0">View on GitHub</a>
</div>

<h2>Quick Start</h2>

```bash
# Clone and install
git clone https://github.com/ivelin/zk0.git
cd zk0
conda create -n zk0 python=3.10
conda activate zk0
pip install -e .

# Quick check with tiny simulation
./train-fl-simulation.sh --tiny

# Or test a single dataset first
./train-lerobot-standalone.sh -d lerobot/pusht -s 10
```

<div class="social-posts">
  <h2>Community Buzz</h2>
  <blockquote class="twitter-tweet"><p lang="en" dir="ltr">It's time for a complete open-source stack for autonomy/robotics plus distributed learning. The first step is here: <a href="https://twitter.com/LeRobotHF?ref_src=twsrc%5Etfw">@LeRobotHF</a> + <a href="https://twitter.com/flwrlabs?ref_src=twsrc%5Etfw">@flwrlabs</a> LFG 🚀<a href="https://twitter.com/comma_ai?ref_src=twsrc%5Etfw">@comma_ai</a> <a href="https://twitter.com/wayve_ai?ref_src=twsrc%5Etfw">@wayve_ai</a> <a href="https://twitter.com/Figure_robot?ref_src=twsrc%5Etfw">@Figure_robot</a> <a href="https://twitter.com/Tesla?ref_src=twsrc%5Etfw">@Tesla</a> <a href="https://t.co/8O8cSD3SbO">https://t.co/8O8cSD3SbO</a> <a href="https://t.co/oVUOLTvwzm">https://t.co/oVUOLTvwzm</a></p>&mdash; nic lane (@niclane7) <a href="https://twitter.com/niclane7/status/1879597539676266726?ref_src=twsrc%5Etfw">January 15, 2025</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

  <blockquote class="twitter-tweet"><p lang="en" dir="ltr">Open-source robots just got a boost. Frameworks like Flower FL enable faster learning, efficient scaling, and continuous knowledge sharing using real-world data. <a href="https://t.co/j8VSGiWF0W">https://t.co/j8VSGiWF0W</a></p>&mdash; 𝚐𝔪𝟾𝚡𝚡𝟾 (@gm8xx8) <a href="https://twitter.com/gm8xx8/status/1879633368427761785?ref_src=twsrc%5Etfw">January 15, 2025</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

  <blockquote class="twitter-tweet"><p lang="en" dir="ltr">We are not so far from a future where robots will be constantly learning by interacting with humans and their environments.<br><br>Frameworks like <a href="https://twitter.com/flwrlabs?ref_src=twsrc%5Etfw">@flwrlabs</a> will enable these robots to learn much faster by continuously sharing their learnings.<br><br>We really live in a sci-fi movie 😅 <a href="https://t.co/kAz3xZ2qvB">https://t.co/kAz3xZ2qvB</a></p>&mdash; Remi Cadene (@RemiCadene) <a href="https://twitter.com/RemiCadene/status/1879592068865282227?ref_src=twsrc%5Etfw">January 15, 2025</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

  <blockquote class="twitter-tweet"><p lang="en" dir="ltr">Federated Learning Meets Robotics: 🤖 LeRobot + 🌼 Flower<br><br>This demo demonstrates how robots in remote environments can collaboratively train an AI model using their local data, which is then aggregated into a shared model. <br><br>In this quickstart, you will train a Diffusion policy… <a href="https://t.co/i32MkbxoPW">pic.twitter.com/i32MkbxoPW</a></p>&mdash; Flower (@flwrlabs) <a href="https://twitter.com/flwrlabs/status/1879571258532036739?ref_src=twsrc%5Etfw">January 15, 2025</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
</div>

<div class="share">
  <h2>Share</h2>
  <img src="https://github.com/user-attachments/assets/e03913ec-62a0-4b05-a286-6fc18dfd433f" alt="zk0 QR Code" style="max-width: 200px; height: auto;" />
</div>
