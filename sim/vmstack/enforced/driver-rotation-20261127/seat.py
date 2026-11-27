"""Driver-seat client: send ONE pilot command, print the pilot's JSON reply.

    python3 seat.py "init"
    python3 seat.py "move 0.01 0 0 20"
    python3 seat.py state

Exit code 0 = reply received (read the JSON: a reply containing
"fault_rewound" means the increment was REFUSED and rolled back).
Exit code 2 = seat server unreachable (start it with start_seat.sh).
"""
import os, socket, sys

SEAT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOCK = os.path.join(SEAT, "seat.sock")

cmd = " ".join(sys.argv[1:]).strip()
if not cmd:
    print("usage: seat.py \"<pilot command>\"")
    sys.exit(64)
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
try:
    s.connect(SOCK)
except OSError as e:
    print(f"seat server unreachable: {e}")
    sys.exit(2)
s.sendall((cmd + "\n").encode())
out = b""
while True:
    chunk = s.recv(65536)
    if not chunk:
        break
    out += chunk
print(out.decode().rstrip())
