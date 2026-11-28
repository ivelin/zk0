---
title: "zk0 Founding Design Partners: Run a Node on Your Own Arm and Data"
description: "Founding design partners are the first external labs and operators to run zk0 nodes on their own robot arms and private datasets — and to shape the node design while it is still being formed. Your data stays on your machines; only model updates move."
---

# zk0 Founding Design Partners

zk0 pools compute and data from independent nodes into one shared model. The network of nodes is the asset — and the founding design partners are the labs and operators who stand up the first external nodes and shape how the node works while it is still being formed.

## Who This Is For

- You have a physical robot arm — an SO-100, SO-101, or another LeRobot-compatible arm — and your own data, or data you are about to record.
- You can commit compute to run a zk0 node on your data during federated training rounds.
- You want a say in the node design: installation, data handling, round scheduling, and validation — the parts that decide whether running a node fits a real lab.

## How It Works

- **Your data stays on your machines.** Training happens locally on your node; only model updates move. Raw data never leaves your environment.
- **One shared model.** Every node trains the same shared SmolVLA model — there are no private per-node variants, so what your node learns improves the model for everyone, and what the network learns comes back to your node.
- **Founding partners shape the design.** Partners work directly with the zk0 team on the node experience. What the founding cohort needs in order to run nodes in real labs is what gets built first.

## What We Ask

- Run a zk0 node on your own data as part of the founding cohort.
- Give structured feedback on the node experience — what works, what breaks, what is missing.
- Meet the same data and licensing terms as every zk0 contributor (see the application checklist): you own your data, and the model weights from local training are licensed to the community under Apache License 2.0 when aggregated into the shared model.

## How to Apply

1. **Review Requirements**: Read the [Node Operators Guide](NODE-OPERATORS) — requirements, setup, and how federation works. Ensure you have:
   - A physical robot arm (SO-100/SO-101 compatible) and a private robotics dataset
   - A GPU-enabled machine (recommended for training)
   - A stable internet connection
   - Basic familiarity with Conda and tmux

2. **Submit Application**: Create a new issue using our [Founding Design Partner Application Template](https://github.com/ivelin/zk0/issues/new?template=founding-design-partner-application.md)

3. **Wait for Approval**: Our team will review your application and contact you via Discord

## Other Ways to Contribute

Not joining the founding cohort? You can still contribute as a [data or compute contributor](DATA-BEATS-PARAMETERS) through the standard [node operator application](https://github.com/ivelin/zk0/issues/new?template=node-operator-application.md).

## Community and Support

### Discord
Join our Discord community for support and updates: [zk0 Discord](https://discord.gg/dhMnEne7RP)

### Contact
- Email: [operators@zk0.bot](mailto:operators@zk0.bot?subject=Founding+Design+Partner)
- Discord: [zk0-team](https://discord.gg/dhMnEne7RP)

---

*Founding design partners join under the same open-source terms as the rest of zk0: Apache License 2.0.*
