# twin-suite v1.1 — Hugging Face reference + external-input pins (D-02)

## HF publication

- Repo: `zk0bot/twin-suite` (dataset, public, CC-BY-4.0)
- URL: https://huggingface.co/datasets/zk0bot/twin-suite
- **Pinned revision (HEAD commitOid): `579ad96b960d1c6ad756b3e08518aa0b04eb60e7`**
- Contents: 168 files, 19,081,155 bytes — the CERTIFIED cable_clip set only
  (20 episode JSONs, 20 base + 100 variant scene XMLs, 13 arm-mesh STLs),
  suite docs, and `validation/validation_results_v11.json`. The other four
  tasks are NOT certified and are NOT in the HF repo.
- param_hash: `a4eb3f8c3611894d`
- sha256 manifest: full per-file manifest (path + sha256 + size, 168 files)
  is committed beside this file as `hf-manifest-v11.json`.
  **Manifest aggregate SHA-256: `1cb900db08c8a4b3d0ff6f4e87b7ff5c728a002b1c3fb9f8c77cc8979152b773`**
  (aggregate = sha256 of the canonical lines `"<sha256>  <path>\n"` sorted
  by path, exactly as listed in hf-manifest-v11.json).

## D-02 external-input pins (inputs the suite was built/validated against)

Vendored SO-101 arm model (referenced by every scene XML via
`<include file="/home/openclaw/muse-work/zk0-sim-v01-20261115/sim/model/so101_new_calib.xml"/>`):

- `so101_new_calib.xml` sha256: `d75253eb568e8a7214db9c631ab7bed4217f608a26f7276ebe9a7636cac82580`
- Mesh files (`sim/model/assets/`, byte-identical to the copies in
  `scenes/assets/` and in the HF package):
  - `base_motor_holder_so101_v1.stl` `8cd2f241037ea377af1191fffe0dd9d9006beea6dcc48543660ed41647072424`
  - `base_so101_v2.stl` `bb12b7026575e1f70ccc7240051f9d943553bf34e5128537de6cd86fae33924d`
  - `motor_holder_so101_base_v1.stl` `31242ae6fb59d8b15c66617b88ad8e9bded62d57c35d11c0c43a70d2f4caa95b`
  - `motor_holder_so101_wrist_v1.stl` `887f92e6013cb64ea3a1ab8675e92da1e0beacfd5e001f972523540545e08011`
  - `moving_jaw_so101_v1.stl` `785a9dded2f474bc1d869e0d3dae398a3dcd9c0c345640040472210d2861fa9d`
  - `rotation_pitch_so101_v1.stl` `9be900cc2a2bf718102841ef82ef8d2873842427648092c8ed2ca1e2ef4ffa34`
  - `sts3215_03a_no_horn_v1.stl` `75ef3781b752e4065891aea855e34dc161a38a549549cd0970cedd07eae6f887`
  - `sts3215_03a_v1.stl` `a37c871fb502483ab96c256baf457d36f2e97afc9205313d9c5ab275ef941cd0`
  - `under_arm_so101_v1.stl` `d01d1f2de365651dcad9d6669e94ff87ff7652b5bb2d10752a66a456a86dbc71`
  - `upper_arm_so101_v1.stl` `475056e03a17e71919b82fd88ab9a0b898ab50164f2a7943652a6b2941bb2d4f`
  - `waveshare_mounting_plate_so101_v2.stl` `e197e24005a07d01bbc06a8c42311664eaeda415bf859f68fa247884d0f1a6e9`
  - `wrist_roll_follower_so101_v1.stl` `4b17b410a12d64ec39554abc3e8054d8a97384b2dc4a8d95a5ecb2a93670f5f4`
  - `wrist_roll_pitch_so101_v2.stl` `6c7ec5525b4d8b9e397a30ab4bb0037156a5d5f38a4adf2c7d943d6c56eda5ae`

Eval split (Study 001 held-out episodes):

- `split.json` (`zk0-study001-20261119/run/eval-traj-20261122/split/split.json`)
  sha256: `a8ca1528774ba8800c5a6ae83629b24ab9aa404c5018c1a1003edeb828f35922`

Dataset metadata (ArmNetBench LeRobot SO-101; also the fps source, 20 fps):

- `meta/info.json` (`zk0-onarm-20261117/data/armnetbench_v01_lerobot_so101/meta/info.json`)
  sha256: `2bba6714747f578d6d5afe0376e6dd001e59dcdd600f2d70d8cbdd98a74a8cab`

Eval trajectory records:

- `records/*.npz` (`zk0-study001-20261119/run/eval-traj-20261122/records/`):
  800 files. Manifest hash (sha256 of the `sha256sum` output over the 800
  files in sorted-name order, i.e. lines `"<sha256>  <name>\n"`):
  `bc4d351d74600046715b97dcfee2fc34bb40cbe6d32e9dfe4ab4b73a520afd08`
