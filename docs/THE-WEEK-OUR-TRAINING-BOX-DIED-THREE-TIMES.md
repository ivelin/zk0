---
title: "The Week Our Training Box Died Three Times"
description: "Three times in four days, our training machine froze dead. This is the full record of what killed it, how we found out, and what fixed it."
permalink: /the-week-our-training-box-died-three-times/
---

# The Week Our Training Box Died Three Times

Three times in four days, our training machine froze dead. No warning, no error message. Someone had to walk over and restart it by hand.

This is the full record of what killed it, how we found out, and what fixed it. It is a process lesson: it says nothing about how well our federated model performs. That evaluation is separate, and it has not run yet.

## What happened?

Our federated training runs share one machine — a single training box that every learner (each arm's training job) takes turns on. Mid-run, all three times: not a crash with a log line we could read — a freeze. The only fix was a person, physically restarting it.

## Why not just blame the hardware?

That was the easy move, and we nearly made it. Instead we rebuilt the failure on purpose — small and safe — and changed one thing at a time, writing down what we expected to see before each run.

The suspect was our own training program: the version that loaded every learner into one process and held them all in memory at once.

## What actually killed it?

Us. The pattern, confirmed from the raw machine logs:

- Every learner loaded into a single process at once, and memory never let go — each stage held what it had grabbed.
- With memory pinned, the chip ran hotter and hotter across the run.
- The machine cut out. No warning, no error message — it just stopped.

An independent reviewer — someone who didn't build any of it — re-checked the raw machine logs (the temperature and memory readings) and confirmed the pattern. Root cause: certified, 2026-11-21.

## What fixed it?

The fix was already sitting in our own history: the way this project originally ran federated training. One learner at a time. Each finishes, lets go of all the memory it was holding, and only then does the next one start.

We ran that way for 70 straight minutes at the same size as the runs that kept dying — longer than any run that had ever killed the box — without a single problem.

## Who did this work?

Most of the experimentation pipeline itself — running the reproduction, watching the readings, keeping the records — was managed by the AI agent (Muse) that runs our experiments day to day, with guidance and judgment from a person at the decision points, and a person able to stop it at any moment. A year ago, frontier models were not good enough to help with robotics experimentation at this level. It's new enough that we're still learning what it changes — this series is partly about that.

## What this does not prove

Nothing here says the federated model is good. This piece is about a training machine and how we run it. Whether federated learning works for robot arms — measured against a centralized model, same tasks, same bar — is a separate evaluation. It has not run yet, and we will publish it either way.

## The receipts

- The public execution record, on the zk0 repo: [study001/fl-substrate-20261121](https://github.com/ivelin/zk0/tree/study001/fl-substrate-20261121?utm_source=blog&utm_medium=signal&utm_campaign=piece1-box-died&utm_content=technical)
- The postmortem and reproduction notes behind this piece: [research_notes/zk0-dgx-outage2-20261121/](https://github.com/ivelin/zk0/tree/study001/fl-substrate-20261121/research_notes/zk0-dgx-outage2-20261121?utm_source=blog&utm_medium=signal&utm_campaign=piece1-box-died&utm_content=technical)
- Root cause certified by an independent verifier from raw telemetry, 2026-11-21.

## Raise your hand

We're building zk0 — many robot arms, one shared brain, trained by people whose data never leaves their desk. Contributors join with compute, with data, or with both; the ongoing human work is the judgment calls and the data only you can supply — the agent manages the pipeline.

Enrollment is not open. If you build, research, or run robots, register your interest at [zk0.bot](https://zk0.bot/?utm_source=blog&utm_medium=signal&utm_campaign=piece1-box-died&utm_content=technical) — when enrollment opens, the people on that list hear first.
