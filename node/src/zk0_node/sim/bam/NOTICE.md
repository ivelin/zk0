# NOTICE — BAM servo profile provenance and licences

This profile (`servo/bam/`) combines three sources. The repo's own
licence (Apache-2.0, `LICENSE` at the repo root) covers zk0's
contribution; the components below keep their own licences.

## 1. Rhoban/bam — fitted servo model + parameter set

- Source: https://github.com/Rhoban/bam
- Pin: commit `e9a619d56da5236206f4de6ceec2c1ee1b497b5c`
  (package reports version 1.0.2)
- Licence: **Apache-2.0**
- What is used: the BAM actuator model library (runtime dependency,
  not vendored) and its fitted STS3215 parameter set
  `bam/params/feetech_sts3215_7_4V/m6.json`, vendored verbatim at
  `params/feetech_sts3215_7_4V/m6.json` in this profile
  (md5 `3209fcb05bfbfddf68ae779fb87df02d`, byte-identical to the
  pinned source).

## 2. punkfab/robot-actuators — SO-101 conversion / reference code

- Source: https://github.com/punkfab/robot-actuators
- Pin: commit `fb59c959045b73d7e49b203d3a98ce53472511f3`
- Licence: **MIT** — full text in `LICENSE.punkfab-mit`
  (Copyright (c) 2026 Dan Newcome)
- What is used: `sts3215/arm.py`, vendored verbatim at `arm.py` in
  this profile (md5 `63bfd2fc72fd0222b9c613436647e546`): the `bamify`
  MJCF conversion and the `StsServos` controller class, used as-is.

## 3. zk0 — integration

- `bam_pilot.py`: zk0's servo-layer integration of the two components
  above into the zk0 test harness, reproduced verbatim from the
  certified BAM tier build (`zk0-bam-tier-20261116`, md5
  `1bfaabf2f57f93a26a8109bd058214e3`). Apache-2.0, per the repo
  `LICENSE`.
