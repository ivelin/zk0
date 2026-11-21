import os
os.environ.setdefault("HF_HUB_OFFLINE", "0")
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata, LeRobotDataset

IDS = [
    "ethanCSL/direction_test",
    "giovipeg/record-test",
    "stanl1y/record_60_simple",
    "qm30631122/so101_grab_pen_20sec_50ep_SCB_102725_2",
    "olingoudey/so101_stationary_mug445",
]
for rid in IDS:
    try:
        meta = LeRobotDatasetMetadata(rid)
        print(f"OK   {rid}: episodes={meta.total_episodes} frames={meta.total_frames} version={meta.info.codebase_version}")
    except Exception as e:
        print(f"FAIL {rid}: {type(e).__name__}: {str(e)[:200]}")

# one real item (video decode) from the two not-yet-proven paths
for rid in ["stanl1y/record_60_simple", "qm30631122/so101_grab_pen_20sec_50ep_SCB_102725_2"]:
    try:
        ds = LeRobotDataset(rid, episodes=[0])
        item = ds[0]
        print(f"ITEM OK {rid}: keys={len(item)} action_shape={tuple(item[action].shape)}")
    except Exception as e:
        print(f"ITEM FAIL {rid}: {type(e).__name__}: {str(e)[:200]}")
