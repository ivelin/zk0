# Study 001 Twin-Replay — Amendment 2: Publication (2026-11-22, 17:57 CT)

Founder addition, folded in by the coordinator before validation results
exist.

1. The certified twin suite is published to the zk0bot Hugging Face org as
   a dataset repo (working name `zk0bot/twin-suite-v1`), holding: scene
   files, variants, parameter table, validation record, verification
   records. The pinned revision + per-file content hashes go into a public
   reference file in the GitHub repo on `study001/fl-substrate-20261121`
   (survivable-copy rule).
2. ONLY scenes/variants that passed demo-replay validation AND independent
   certification are published. Failed/excluded variants stay in the DGX
   workdir record, marked non-certified, never in the collection.
3. Known constraint (HF connector doc, verified capability): the current
   fine-grained token creates repos under zk0bot but CANNOT create HF
   Collections (403, repos-only scopes). Publication therefore lands as a
   dataset repo; a formal HF Collection wrapper requires the founder to
   create it in the web UI or a classic write token. Flagged in the
   completion report; does not block the dataset-repo publication.
4. Publication happens after Phase 5 certification, from the certified
   list exactly; the publisher does not re-decide membership.
