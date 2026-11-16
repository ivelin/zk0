## What changed and why

<!-- One or two sentences: the change, and the reason for it. -->

## How it was verified

<!-- Commands run and their results — not intentions. If checks were run
     differently from CI (e.g. pytest directly instead of the Docker CI
     image), say so. -->

## Independent verification

<!-- Who or what verified this independently of the builder (a separate
     reviewer/agent, or the maintainer), and what they checked.
     If not applicable, say why. -->

## Claims and content checks

- [ ] Every public-facing claim in this PR traces to a certified record (link it)
- [ ] Sim results are not presented as real-world evidence; provenance marks (sim-trained / sim-validated / real-validated) are exact
- [ ] Content split respected: no docs/reports/analysis that belong in the maintainer's private docs, no objects/models that belong on Hugging Face
- [ ] No secrets, tokens, or credentials anywhere in the diff
