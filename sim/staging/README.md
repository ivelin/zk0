# staging/

The harness's working-data root inside this snapshot. The scene
builder reads its calibration inputs from `staging/scenes/`
(`calibration_v2.json`, `calibration_fk_affine.json` — shipped).
`staging/records/` is where replay records land when the harness runs
with its default path patch. If you place the public base corpus
(`zk0bot/armnetbench_v01_lerobot_so101`) at
`staging/armnetbench_v01_lerobot_so101/`, the parameter module reads
the frame rate from its `meta/info.json` live; without it, the
verified fallback (20 fps) applies.
