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

The best-resourced robot-learning program in history ran the same experiment, and its sequence is instructive. We cite it as an analogy, with each claim attributed to its source — and we pin the two most-repeated Tesla claims to their primary sources along the way, because an evidence-based project should.

- **The hardware was modest.** Tesla's HW3 computer (2019) carries two redundant Samsung 14nm chips; each chip's two neural accelerators peak at 36.86 TOPS, i.e. 73.7 TOPS per chip, 144 TOPS counting both chips on the board [8][9]. Production FSD ran on HW3 from 2019, and Tesla was still shipping HW3-specific FSD builds in 2025 [10]. A very hard perception-and-driving problem was carried for years by an inference chip far weaker — on paper — than a gaming PC.
- **The data was not modest.** Tesla's investor decks chart cumulative FSD miles passing 1 billion by April 2024 [11] and roughly 6 billion (Supervised) miles by Q3 2025 [12]; the cumulative total crossed **10 billion miles** in May 2026 [23], and Tesla's FSD safety-page counter — miles driven with FSD (Supervised) *engaged*, a measure of exposure rather than of labeled training data — read **17,500,515,880 miles** when captured on November 13, 2026 [13]. Elluswamy put the usage figure in words on the Q3 2025 earnings call: customers "have used Full Self-Driving supervised for a total of 6 billion [miles] as of yesterday" [32]. Its disclosed *training* corpus was described in clips, not kilometres: ~1 million 10-second clips with 6 billion labeled objects (2021) [14]. On the Q2 2023 earnings call, Musk described scaling training examples: "at 1 million training examples, it barely works… you get like 10 million training examples, it's like — it becomes incredible" [15]. That curation funnel, not raw mileage, is the asset.
- **Correction 1:** the circulating claim of "10 billion km of driver-labeled footage" is **not** something Tesla has reported. The 10-billion figure traces to Musk's 2016 *Master Plan, Part Deux*, where 6 billion miles (10 billion km) was his estimate of the fleet learning *required* for regulatory approval [16] — a target, not a labeled dataset. Musk restated the same threshold a decade later, in the same posture: "roughly 10 billion miles of training data is needed to achieve safe unsupervised self-driving" (X, January 8, 2026) [24] — again a requirement, stated in miles. The honest version is stronger anyway: billions of real-world miles, with training data scaled in millions of curated examples.
- **The "10× parameters" story, from the primary record.** The popular claim that "HW4 runs models with 10× the parameters of HW3" misattaches a real number. Here is that number's actual paper trail, in Tesla's own words:
  - In August 2025, Musk announced the model itself — twice on X: the coming FSD release would bring "a dramatic gain with a 10X higher parameter count" [18], and Tesla was training "a new FSD model with ~10X params and a big improvement to video compression loss. Probably ready for public release end of next month if testing goes well" [25]. Tesla skipped an interim 4× model to go straight to the 10× [25].
  - FSD v14 shipped that October on HW4 — but the 10× did not ship with it. The v14.1 and v14.3 release notes contain no parameter or model-size language at all [34][35]. Musk confirmed it on the Q3 2025 earnings call (October 22, 2025): "We're going to be increasing the parameter count by an order of magnitude. That's not in 14.1." [32] The pattern is older than v14: Tesla's v13 release notes claimed, as shipped, "4.2x data scaling" and "5x training compute scaling," while listing "3x model size scaling" only under *Upcoming Improvements* [38][19].
  - Then, in April 2026, Musk deferred the large model explicitly: **"Our rate of advancement with the small model has been so fast that the large model has not yet caught up. V15 will be the large model."** [26] So the compact model is what consumer builds from v12 through v14 actually run. Secondary outlets put that model at roughly 1 billion parameters and v15 at roughly 10 billion [27] — Tesla has never published an absolute parameter count; only the ratio is Tesla's own. And the large model is a *software* generation, not new silicon: Musk has said v15 "will run on AI4" [33]. Elluswamy told the Q2 2026 earnings call that the Robotaxi fleet was "already running early versions of the V15 FSD software," with 40% of v15's improvement tracks merged [37] — he did not say those early builds contain the full large model.
  - HW3's side of the divide was stated just as plainly. Elluswamy acknowledged in August 2024 that HW3 runs a "relatively smaller model" than AI4 [28]. On the Q4 2024 earnings call, Musk conceded that HW3 computers would have to be upgraded for customers who bought FSD: "The honest answer is that we're going to have to upgrade people's Hardware 3 computer for those that have bought Full Self Driving… that's going to be painful and difficult, but we'll get it done" [29]. By the Q1 2026 call, the verdict was unhedged: "Hardware 3 simply does not have the capability to achieve unsupervised FSD" — with the binding constraint named: HW3 "has only 1/8 of the memory bandwidth of Hardware 4" [33]. What HW3 received instead, in June 2026, was a distilled build: Tesla's own v14 Lite release notes say it "Distilled the intelligence from HW4 V14 into HW3" [36], and Elluswamy described it as a build that "distills the driving behavior from AI4's v14 series into both the camera and compute config of AI3" [30]. The binding constraint, per Musk, is not TOPS but bandwidth: "the AI3 computer only has ~15% of the effective memory bandwidth of AI4" [30]. Firmware teardowns show the mechanism: the end-to-end driving network alone tripled from 2.3 GB in v12 to 7.5 GB in v13 [31].
  - The other "10×" in circulation is a *hardware* figure and a different claim: at the June 2024 shareholder meeting, HW4 was described as roughly 3–5× HW3 in inference, and AI5 as ~10× HW4 [17]. Tesla has never published a TOPS figure for HW4; its stated numbers are HW3's.

  So the documented 10× is a *model-generation* step — the pending v15 large model versus the compact model that carried FSD from v12 through v14 — not an HW4-versus-HW3 figure, and not a line in any set of release notes (we checked the v14.1 and v14.3 note texts; they contain no parameter language at all). And the sequencing is no longer our inference alone: by Musk's own account, the large model waited because the small one kept getting better on data.

Our interpretation, labeled as such: at Tesla's scale, data scaled first — that half is Tesla's own documented record, in its release notes and earnings calls. The second half remains ahead of the evidence: the order-of-magnitude model is a pending v15 software plan on the same AI4 silicon, and its long-tail benefit is stated intent, not a demonstrated result. A community project with a fraction of Tesla's resources should copy the documented sequence — data first — not skip to the step Tesla itself has not yet shipped.

- **The hardware comparison, fairly made.** An RTX 3090 (2020) carries 24 GB of GDDR6X and peaks at 284 INT8 TOPS (dense) per NVIDIA's architecture whitepaper [20][21] — above HW3's board total on that metric — and unlike the car's inference ASIC, it also *trains*. Hugging Face's LeRobot hardware guide lists SmolVLA fine-tuning at roughly **10–16 GB peak VRAM** (batch 8, AdamW; the guide calls its figures indicative) [22]. That is why our compute floor is the **10 GB+ VRAM class** — RTX 3060 12 GB, 3080, 3090, 4070 and up — not a data-center card.

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
13. Tesla. "FSD (Supervised) Vehicle Safety Report" — cumulative miles driven with FSD (Supervised) engaged; the counter read 17,500,515,880 miles at capture on Nov 13, 2026. https://www.tesla.com/fsd/safety
14. NVIDIA / CleanTechnica. "Tesla's AI Training Supercomputers Powered by NVIDIA" (reporting AI Day 2021: ~1M clips, 1.5 PB, 6B labeled objects), Aug 19, 2021. https://cleantechnica.com/2021/08/19/nvidia-teslas-ai-training-supercomputers-powered-by-us/
15. Tesla Q2 2023 earnings call transcript (Musk on training-example scaling), July 19, 2023. https://www.roic.ai/quote/TSLA/transcripts/2023-year/2-quarter
16. Musk, E. "Master Plan, Part Deux." Tesla, July 20, 2016. https://www.tesla.com/master-plan-part-deux
17. Not a Tesla App. "Tesla Officially Announces FSD Hardware 5.0 and How It Compares to Hardware 4.0" (reporting the June 13, 2024 shareholder meeting: HW4 ~3–5× HW3 inference; AI5 ~10× HW4), June 17, 2024. https://www.notateslaapp.com/news/2081/tesla-officially-announces-fsd-hardware-50-and-how-it-compares-to-hardware-40
18. Electrek. "Elon Musk promising Tesla unsupervised self-driving… with 10X parameter count" (Musk on X, Aug 2025), Aug 12, 2025. https://electrek.co/2025/08/12/elon-musk-promising-tesla-unsupervised-self-driving-to-less-nag/
19. Drive Tesla Canada. "Tesla starts limited rollout of FSD v13.2" (v13.2 release notes: upcoming "3x model size scaling"), Dec 2, 2024. https://driveteslacanada.ca/news/tesla-starts-limited-rollout-of-fsd-v13-2-and-early-tests-show-impressive-new-capabilities/
20. NVIDIA. "GeForce RTX 3090 / 3090 Ti" product specifications (24 GB GDDR6X). https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3090-3090ti/
21. NVIDIA. "NVIDIA Ampere GA102 GPU Architecture Whitepaper v2.1" (RTX 3090: 284 INT8 TOPS dense; 71 TFLOPS FP16 tensor with FP32 accumulate). https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.1.pdf
22. Hugging Face. LeRobot "Compute Hardware Guide" (SmolVLA / small VLA: ~10–16 GB peak VRAM, batch 8, AdamW; figures indicative). https://huggingface.co/docs/lerobot/en/hardware_guide
23. The Verge. "Tesla crosses 10 billion FSD miles" (cumulative FSD miles passed 10 billion, May 2026). https://www.theverge.com/transportation/922900/tesla-10-billion-miles-unsupervised-fsd-robotaxi-elon-musk
24. Tesorb. "$425 and $1.75 trillion: what FSD V15 must deliver for" (reporting Musk's Jan 8, 2026 X post: "roughly 10 billion miles of training data is needed to achieve safe unsupervised self-driving"; Morgan Stanley note of Apr 10, 2026). https://tesorb.com/fsd-v15-unsupervised-driving-spacex-ipo/
25. Not a Tesla App. "Tesla FSD V14: 10x Parameter Update, New Features, Releases in September" (quoting Musk on X, Aug 6, 2025: "Tesla is training a new FSD model with ~10X params…"), Aug 7, 2025. https://www.notateslaapp.com/news/2999/tesla-fsds-10x-parameter-update-many-new-features-release-in-september
26. Drive Tesla Canada. "Tesla confirms FSD 10× model coming in v15" (Musk on X, Apr 9, 2026: "Our rate of advancement with the small model has been so fast that the large model has not yet caught up. V15 will be the large model."), Apr 2026. https://driveteslacanada.ca/news/tesla-confirms-fsd-10x-model-coming-in-v15
27. Not a Tesla App. "Tesla AI Chief Talks About What's Coming in FSD V15" (current builds ~1 billion parameters; v15 = 10× to ~10 billion, HW4-only; Elluswamy on v15), Sept 11, 2026. https://www.notateslaapp.com/news/4679/tesla-ai-chief-talks-about-whats-coming-in-fsd-v15
28. Electrek. "Tesla FSD Europe launch backlash: HW3 owners launch claim site" (recording Elluswamy's X post of Aug 21, 2024: HW3 runs a "relatively smaller model" than AI4, with workarounds), Apr 14, 2026. https://electrek.co/2026/04/14/tesla-fsd-europe-hw3-owners-dutch-claim/
29. Drive Tesla Canada. "Tesla to upgrade HW3 computers for FSD owners — 'it will be painful and difficult'" (Musk, Q4 2024 earnings call, Jan 29, 2025), Jan 30, 2025. https://driveteslacanada.ca/news/tesla-to-upgrade-hw3-computers-for-fsd-owners-it-will-be-painful-and-difficult/
30. TeslaHubs. "Everything You Need to Know About Tesla FSD V14 Lite" (quoting Elluswamy on X, June 29, 2026: the build "distills the driving behavior from AI4's v14 series into both the camera and compute config of AI3"; Musk, same day: AI3 has "~15% of the effective memory bandwidth of AI4"; Tesla AI account on distillation). https://teslahubs.com/blogs/tips/everything-you-need-to-know-about-tesla-fsd-v14-lite
31. Not a Tesla App. "Tesla's FSD V13 Pushes HW4 Hardware Capabilities; End of Line for HW3?" (Greentheonly's firmware analysis: end-to-end driving network 2.3 GB in v12 → 7.5 GB in v13), Apr 18, 2025. https://www.notateslaapp.com/news/2655/teslas-fsd-v13-pushes-hw4-hardware-capabilities-end-of-line-for-hw3
32. Tesla Q3 2025 earnings call transcript (Oct 22, 2025) — Musk: "We're going to be increasing the parameter count by an order of magnitude. That's not in 14.1."; Elluswamy: customers "have used Full Self-Driving supervised for a total of 6 billion [miles] as of yesterday." https://www.marketbeat.com/earnings/reports/2025-10-23-tesla-inc-stock/
33. Tesla Q1 2026 earnings call transcript (Apr 22, 2026) — Musk: "Hardware 3 simply does not have the capability to achieve unsupervised FSD"; HW3 "has only 1/8 of the memory bandwidth of Hardware 4"; v15 "will run on AI4." https://stockanalysis.com/stocks/tsla/transcripts/547944-q1-2026/
34. Not a Tesla App. FSD v14.1 release notes (firmware 2025.32.8.5; HW4; no parameter or model-size language anywhere in the notes), Oct 2025. https://www.notateslaapp.com/software-updates/version/2025.32.8.5/release-notes
35. Not a Tesla App. FSD v14.3 release notes (firmware 2026.2.9.6; HW4; no parameter or model-size language anywhere in the notes), Apr 2026. https://www.notateslaapp.com/software-updates/version/2026.2.9.6/release-notes
36. Not a Tesla App. FSD v14 Lite release notes (firmware 2026.20.6.10): "Distilled the intelligence from HW4 V14 into HW3," June 2026. https://www.notateslaapp.com/software-updates/version/2026.20.6.10/release-notes
37. Tesla Q2 2026 earnings call transcript (Jul 22, 2026) — Elluswamy: the Robotaxi fleet is "already running early versions of the V15 FSD software"; of v15's seven improvement tracks, "already 40% of those tracks merged together." https://stockanalysis.com/stocks/tsla/transcripts/653184-q2-2026/
38. Not a Tesla App. FSD v13.2.6 release notes (firmware 2024.45.32.10): shipped bullets include "4.2x data scaling" and "5x training compute scaling"; "3x model size scaling" appears only under Upcoming Improvements. https://www.notateslaapp.com/software-updates/version/2024.45.32.10/release-notes
