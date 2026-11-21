import re
p = "pyproject.toml"
lines = open(p).read().splitlines(keepends=True)
out = []
clients_done = False
for ln in lines:
    s = ln.rstrip("\n")
    if s.startswith("options.num-supernodes = 2"):
        out.append("options.num-supernodes = 5 # SOAK SCALE: 5 supernodes (baseline pool subset; baseline file had 12, Stages A-C miniature had 2), number of super nodes / clients must match exactly the number of train datasets in order to guarantee correct mapping\n")
    elif s.startswith("num-server-rounds = 20"):
        out.append("num-server-rounds = 3 # SOAK 2026-11-21: 3 rounds (file baseline 20) - fix-verification soak, see reproduction protocol\n")
    elif s.startswith("hf_repo_id = "):
        out.append("# SOAK 2026-11-21: hf_repo_id commented out so the soak cannot push to the published repo; local checkpoints still save every round (checkpoint_interval=1). Original: " + s + "\n")
    elif s.startswith("    {name = \"giovipeg/record-test\"") or s.startswith("    {name = \"ethanCSL/direction_test\""):
        if not clients_done:
            out.append("    {name = \"ethanCSL/direction_test\", description = \"turn to the right side - VALIDATED CLEAN\", client_id = 0},\n")
            out.append("    {name = \"giovipeg/record-test\", description = \"Pick and place pink knob. - VALIDATED CLEAN\", client_id = 1},\n")
            out.append("    {name = \"stanl1y/record_60_simple\", description = \"Grab the red pen and place it into the bin. - VALIDATED CLEAN\", client_id = 2},\n")
            out.append("    {name = \"qm30631122/so101_grab_pen_20sec_50ep_SCB_102725_2\", description = \"Grab pen and put in to cylinder. - VALIDATED CLEAN\", client_id = 3},\n")
            out.append("    {name = \"olingoudey/so101_stationary_mug445\", description = \"Put the stuffed animal in the mug. - VALIDATED CLEAN\", client_id = 4},\n")
            clients_done = True
        # skip original entry lines
    else:
        out.append(ln)
open(p, "w").write("".join(out))
print("clients_done:", clients_done)
