---
title: "Data Beats Parameters: Why zk0 Trains a 450M Robot Model — and Who We Are Recruiting to Train It"
description: "The published evidence says robot-arm policies gain more from diverse data than from bigger models. zk0 is recruiting SO-100/SO-101 data contributors and 10GB+ VRAM compute contributors to train a shared SmolVLA — data never leaves your disk."
---

# Data Beats Parameters: Why zk0 Trains a 450M Robot Model — and Who We Are Recruiting to Train It

**zk0 is actively recruiting contributors.** We are building a shared, open robot manipulation model — SmolVLA, 450 million parameters — trained federatively by a community of robot owners and GPU owners. Your robot data never leaves your machine; only model updates travel. This article lays out the published evidence behind our two design bets, because we think they are the opposite of what the headlines suggest:

1. **We do not need a bigger model.** Bigger vision-language-action (VLA) models exist — 3B, 7B, even 14B. There is no controlled evidence that a robot arm needs them. In every matched comparison we can find, data and training recipe beat parameter count.
2. **We need more — and more diverse — data.** That is why zk0 recruits people, not GPUs alone: SO-101 arm owners with private demonstrations, and compute contributors with ordinary consumer GPUs.

If you only read one section, read the evidence. If it convinces you, [join us](#join-zk0-two-ways-to-contribute).

## Evidence 1: SmolVLA's own scaling curve is nearly flat — its data curve is not

SmolVLA's authors (Shukor et al., Hugging Face, 2025) trained the same architecture at three sizes and evaluated on LIBERO, the standard manipulation benchmark [1]:

| Model size | LIBERO avg. success |
|---|---|
| 0.24B | 82.75% |
| **0.45B (zk0's model)** | **87.3%** |
| 2.25B (5× the 0.45B model) | 88.75% |

Five times the parameters bought **+1.45 points** on LIBERO. (On the harder Meta-World suite the same scale-up bought more — 57.3% → 68.24% — so capacity is not worthless; it is simply not the binding constraint at our scale.)

Now compare the data lever, from the same paper. On real SO-100 hardware, the 0.45B model trained multi-task scored **51.7%** average success. After pretraining on a community corpus of just **481 datasets / 22,900 episodes** contributed by LeRobot users, the identical model scored **78.3%** (Pick-Place 75%, Stacking 90%, Sorting 70%) [1]. **+26.6 points from community data; +1.45 points from 5× parameters.** That asymmetry is zk0's founding observation.

## Evidence 2: Across the field, recipe and data keep beating size

- **Same weights, better recipe:** OpenVLA is a 7B model that scores 76.5% on LIBERO. OpenVLA-OFT kept the *same 7B weights* and changed the fine-tuning recipe, reaching 97.1%\* [2]. The capability was in the training, not in a bigger network.
- **Small can win outright:** X-VLA, at **0.9B** parameters, reports 98.1% on LIBERO [3] — above every 7B-class score published for that benchmark.
- **Diversity is the scaling variable:** Lin et al. (ICLR 2025), in a study with 40,000+ demonstrations, found that "the generalization performance of the policy follows a roughly power-law relationship with the number of environments and objects," while "once the number of demonstrations per environment or object reaches a certain threshold, additional demonstrations have minimal effect" [4]. Generalization scales with *variety* of data — precisely what a distributed community of robot owners produces and a single lab cannot.
- **A honesty check on the benchmark itself:** LIBERO-PRO (2025) showed that models scoring over 90% on standard LIBERO "collapse to 0.0%" under its generalized setting of combined perturbations [5]. Much of the remaining gap at the top of the leaderboard measures layout memorization, not capability. Chasing it with bigger models optimizes the wrong thing.

\* *Precision note:* OFT's 97.1% also adds wrist-camera and proprioceptive inputs. With the original single-image inputs, the recipe alone reaches 95.3% — still +18.8 points over the 76.5% baseline, with zero added parameters [2].

## Evidence 3: Federation works — and its bottleneck is pretraining data, which is recruitable

The most decision-relevant study for zk0 is a September 2026 empirical testbed of federated fine-tuning for VLAs (Duan et al., UCL and Cisco Research) — we flag it clearly: it is an **arXiv preprint from a single paper (two institutions), not yet peer-reviewed or replicated** [6]. Its findings:

- Federated π0.5 (3.2B) **matched** centralized training: 97.50% vs 97.00% on LIBERO.
- Federated SmolVLA **lagged**: 75.25% vs 90.00% centralized.
- The authors' explanation matters more than the scores: weight averaging works while clients stay close to a strongly pretrained base. π0.5 was pretrained on **>10,000 hours** of robot data; SmolVLA's community pretraining is roughly **100 hours** (22.9k episodes). SmolVLA's federated gap is a *pretraining-depth* gap, not a size gap.

Read that again: the measured fix for federated small-model training is **more community pretraining data** — the exact resource zk0 exists to gather, and the exact resource this article is recruiting. The same study found aggregation choice matters (FedAvg 97.50% vs FedProx 94.25% in their testbed), and we are benchmarking both in our stack rather than assuming [6].

A second September 2026 preprint, RoboFL, shows the upgrade path if a bigger base is ever warranted: clients federated a 3B model by training only a rank-16 LoRA adapter — 40.85M trainable parameters, 311.63 MiB exchanged per round — and reached 83.12% on RoboTwin 2.0 [7]. Bigger models remain reachable later, cheaply, *if* evidence ever demands them. Nothing needs to be decided today, and nothing about zk0's small-model start forecloses it.

## The Tesla parallel: data first, parameters later — at the largest scale in robotics

The best-resourced robot-learning program in history ran the same experiment, and its sequence is instructive. We cite it as an analogy, with each claim attributed to its source — and we correct two popular overclaims along the way, because an evidence-based project should.

- **The hardware was modest.** Tesla's HW3 computer (2019) carries two redundant Samsung 14nm chips; each chip's two neural accelerators peak at 36.86 TOPS, i.e. 73.7 TOPS per chip, 144 TOPS counting both chips on the board [8][9]. Production FSD ran on HW3 from 2019, and Tesla was still shipping HW3-specific FSD builds in 2025 [10]. A very hard perception-and-driving problem was carried for years by an inference chip far weaker — on paper — than a gaming PC.
- **The data was not modest.** Tesla's investor decks chart cumulative FSD miles passing 1 billion by April 2024 [11] and roughly 6 billion (Supervised) miles by Q3 2025 [12]; Tesla's own dashboard currently reports **over 10.5 billion km driven with FSD (Supervised) active** [13]. Its disclosed *training* corpus was described in clips, not kilometres: ~1 million 10-second clips with 6 billion labeled objects (2021) [14]. On the Q2 2023 earnings call, Musk described scaling training examples: "at 1 million training examples, it barely works… you get like 10 million training examples, it's like — it becomes incredible" [15].
- **Correction 1:** the circulating claim of "10 billion km of driver-labeled footage" is **not** something Tesla has reported. The 10-billion-km figure traces to Musk's 2016 *Master Plan, Part Deux*, where 6 billion miles (10 billion km) was his estimate of the fleet learning *required* for regulatory approval [16] — a target, not a labeled dataset. The honest version is stronger anyway: billions of real-world miles, with training data scaled in millions of curated examples.
- **Correction 2:** "HW4 runs models with 10× the parameters of HW3" is **not** a Tesla claim we can find. Musk has described HW4 as roughly **3–5× faster in inference** than HW3 (June 2024) [17]; the "10×" numbers belong to **AI5 vs HW4** (same meeting) [17] and to FSD v14's parameter count *versus its predecessor release* (Musk on X, August 2025; rollout coverage later put it at 6× vs v13) [18][19]. Even in the HW4 era, Tesla's own FSD v13 release notes (December 2024) listed "3× model size scaling" as an *upcoming* improvement [20] — the bigger model came *after* years of data-driven gains on the smaller hardware generation.

Our interpretation, labeled as such: at Tesla's scale, data scaled first and parameters followed to polish the long tail. A community project with a fraction of Tesla's resources should copy the sequence, not skip it.

- **The hardware comparison, fairly made.** An RTX 3090 (2020) carries 24 GB of GDDR6X and peaks at 284 INT8 TOPS (dense) per NVIDIA's architecture whitepaper [21][22] — above HW3's board total on that metric — and unlike the car's inference ASIC, it also *trains*. Hugging Face's LeRobot hardware guide lists SmolVLA fine-tuning at roughly **10–16 GB peak VRAM** (batch 8, AdamW; the guide calls its figures indicative) [23]. That is why our compute floor is the **10 GB+ VRAM class** — RTX 3060 12 GB, 3080, 3090, 4070 and up — not a data-center card.

## What zk0 is, concretely

zk0 is an open-source (Apache-2.0) federated learning platform built on Flower and LeRobot. Node operators run a SuperNode on their own machine: training happens locally on private SO-100/SO-101 datasets, and only model updates are sent to the aggregator — raw episodes never leave the operator's disk. The current shared model is SmolVLA (450M), chosen deliberately: small enough for consumer GPUs, strong enough that the published evidence says data — not size — is its frontier.

We are also forming a **hands-on workshop series** for new contributors — bring an arm, leave with a live node and your first dataset in a federated round. No dates yet; register interest through the same routes below and you will hear first.

## Join zk0: two ways to contribute

**Path 1 — Data contributor (you have the arm).**
You own a LeRobot SO-100 or SO-101 and have private demonstrations — the application asks about datasets of 100+ episodes — that you cannot or will not upload to a public hub. That privacy constraint is the point of zk0: your data trains the shared model *without leaving your machine*.

**Path 2 — Compute contributor (you have the GPU).**
You have a consumer GPU in the 10 GB+ VRAM class (RTX 3060 12 GB / 3080 / 3090 / 4070 or better) and want your idle cycles training a community robot model as a zk0 node.

**Both paths start at the same door:**

1. Read the [Node Operators Guide](https://zk0.bot/docs/NODE-OPERATORS.html) — requirements, setup, and how federation works.
2. Apply via the [node operator application](https://github.com/ivelin/zk0/issues/new?template=node-operator-application.md) on GitHub (reviews typically within 1–3 days).
3. Say hello in the [zk0 Discord](https://discord.gg/dhMnEne7RP), or write to [operators@zk0.bot](mailto:operators@zk0.bot?subject=Join+Network).

The model gets better when the community's data does. Bring yours.

## References

1. Shukor, M. et al. "SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics." arXiv:2506.01844, 2025. https://arxiv.org/abs/2506.01844
2. Kim, M. J. et al. "Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success" (OpenVLA-OFT). arXiv:2502.19645, 2025. https://arxiv.org/abs/2502.19645
3. "X-VLA: Soft-Prompted Transformer as a Scalable Cross-Embodiment Vision-Language-Action Model." arXiv:2510.10274, 2025. https://arxiv.org/abs/2510.10274
4. Lin, F. et al. "Data Scaling Laws in Imitation Learning for Robotic Manipulation." ICLR 2025. arXiv:2410.18647. https://arxiv.org/abs/2410.18647
5. "LIBERO-PRO: Towards Robust and Fair Evaluation of Vision-Language-Action Models." arXiv:2510.03827, 2025. https://arxiv.org/abs/2510.03827
6. Duan, J. et al. "An Empirical Study and Open Testbed for Federated Fine-Tuning of Vision-Language-Action Models" (decent-vla). arXiv:2609.22973, Sept 2026. *Preprint; not peer-reviewed.* https://arxiv.org/abs/2609.22973
7. "RoboFL: Federated Learning for Vision-Language-Action Models via LoRA Expert Assembly." arXiv:2609.34968, Sept 2026. *Preprint; not peer-reviewed.* https://arxiv.org/abs/2609.34968
8. WikiChip. "FSD Chip — Tesla" (documenting Tesla Autonomy Day, April 22, 2019). https://en.wikichip.org/wiki/tesla_(car_company)/fsd_chip
9. The Register. "Tesla claims its self-driving car chip can do 144 trillion operations per second," April 22, 2019. https://www.theregister.com/2019/04/22/tesla_self_driving_cars/
10. Not a Tesla App. Software update 2024.45.32.10 release notes (FSD v12.6.3 for HW3), Feb 2025. https://www.notateslaapp.com/software-updates/version/2024.45.32.10/release-notes
11. Tesla, Inc. Q1 2024 Update Deck (cumulative FSD Beta miles chart; Tesla AI reported >1 billion FSD miles, April 5, 2024). https://digitalassets.tesla.com/tesla-contents/image/upload/IR/TSLA-Q1-2024-Update.pdf
12. Tesla, Inc. Q3 2025 Update Deck (cumulative miles driven with FSD (Supervised)). https://assets-ir.tesla.com/tesla-contents/IR/TSLA-Q3-2025-Update.pdf
13. Tesla. "FSD (Supervised) Evidence Dashboard" (accessed Nov 13, 2026). https://www.tesla.com/fsd-evidence-dashboard
14. NVIDIA / CleanTechnica. "Tesla's AI Training Supercomputers Powered by NVIDIA" (reporting AI Day 2021: ~1M clips, 1.5 PB, 6B labeled objects), Aug 19, 2021. https://cleantechnica.com/2021/08/19/nvidia-teslas-ai-training-supercomputers-powered-by-us/
15. Tesla Q2 2023 earnings call transcript (Musk on training-example scaling), July 19, 2023. https://www.roic.ai/quote/TSLA/transcripts/2023-year/2-quarter
16. Musk, E. "Master Plan, Part Deux." Tesla, July 20, 2016. https://www.tesla.com/master-plan-part-deux
17. Not a Tesla App. "Tesla Officially Announces FSD Hardware 5.0 and How It Compares to Hardware 4.0" (reporting the June 13, 2024 shareholder meeting: HW4 ~3–5× HW3 inference; AI5 ~10× HW4), June 17, 2024. https://www.notateslaapp.com/news/2081/tesla-officially-announces-fsd-hardware-50-and-how-it-compares-to-hardware-40
18. Electrek. "Elon Musk promising Tesla unsupervised self-driving… with 10X parameter count" (Musk on X, Aug 2025), Aug 12, 2025. https://electrek.co/2025/08/12/elon-musk-promising-tesla-unsupervised-self-driving-to-less-nag/
19. Notebookcheck. "Tesla FSD 14 released for HW4 vehicles…" (6× parameter count vs FSD 13), Oct 7, 2025. https://www.notebookcheck.net/Tesla-FSD-14-released-for-HW4-vehicles-with-Sloth-driving-mode-and-parking-spot-selection.1133360.0.html
20. Drive Tesla Canada. "Tesla starts limited rollout of FSD v13.2" (v13.2 release notes: upcoming "3x model size scaling"), Dec 2, 2024. https://driveteslacanada.ca/news/tesla-starts-limited-rollout-of-fsd-v13-2-and-early-tests-show-impressive-new-capabilities/
21. NVIDIA. "GeForce RTX 3090 / 3090 Ti" product specifications (24 GB GDDR6X). https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3090-3090ti/
22. NVIDIA. "NVIDIA Ampere GA102 GPU Architecture Whitepaper v2.1" (RTX 3090: 284 INT8 TOPS dense; 71 TFLOPS FP16 tensor with FP32 accumulate). https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.1.pdf
23. Hugging Face. LeRobot "Compute Hardware Guide" (SmolVLA / small VLA: ~10–16 GB peak VRAM, batch 8, AdamW; figures indicative). https://huggingface.co/docs/lerobot/en/hardware_guide
