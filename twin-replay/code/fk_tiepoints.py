import numpy as np, mujoco, glob, os, json
MODEL="/home/openclaw/muse-work/zk0-sim-v01-20261115/sim/model/so101_new_calib.xml"
REC="/home/openclaw/muse-work/zk0-study001-20261119/run/eval-traj-20261122/records"
m=mujoco.MjModel.from_xml_path(MODEL); d=mujoco.MjData(m)
jn=["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"]
qadr=[m.joint(n).qposadr[0] for n in jn]
gid=m.body("gripper").id; wid=m.body("wrist").id
out={}
for f in sorted(glob.glob(REC+"/base__*.npz")):
    ep=os.path.basename(f)[6:-4]
    gt=np.load(f)["gt"]
    row={}
    for tag,idx in (("first",0),("final",-1)):
        d.qpos[:]=0
        for a,v in zip(qadr,gt[idx]): d.qpos[a]=np.deg2rad(v)
        mujoco.mj_forward(m,d)
        row[tag]={"gripper_xyz":d.xpos[gid].tolist(),"wrist_xyz":d.xpos[wid].tolist(),"q_deg":gt[idx].tolist()}
    out[ep]=row
json.dump(out,open("/home/openclaw/muse-work/zk0-study001-twin-20261122/scenes/fk_tiepoints.json","w"),indent=1)
print("episodes",len(out))
for k in list(out)[:3]: print(k,out[k]["first"]["gripper_xyz"],out[k]["final"]["gripper_xyz"])
