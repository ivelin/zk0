"""Driver-seat server for the zk0 sim teleop pilot (DGX seat, 2026-11-27).

Holds ONE persistent pilot453_v4.py subprocess (mjSTATE lives in it) and
exposes it over a Unix socket so any CLI agent (agy, grok, ...) can drive
with ordinary shell calls:

    python3 seat.py "move 0 0 0.05 20"     -> prints the pilot's JSON reply

One command at a time (serialized with a lock). The pilot's reply is the
JSON summary line it prints before its '>>>' prompt; the server returns
everything the pilot printed for that command, verbatim.

Paths are derived from this file's location:
  <seat>/seatbin/seat_server.py
  <seat>/validator-sets-20261123/coverage-topup-work/vmstack/enforced/
Socket: <seat>/seat.sock   Log: <seat>/seat_server.log   Pilot stderr: <seat>/pilot_stderr.log
The Python interpreter is taken from SEAT_PYTHON env or sys.executable —
start_seat.sh passes the seat venv python explicitly.
"""
import os, socket, subprocess, sys, threading

SEAT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENF = os.path.join(SEAT, "validator-sets-20261123", "coverage-topup-work",
                   "vmstack", "enforced")
SOCK = os.path.join(SEAT, "seat.sock")
PY = os.environ.get("SEAT_PYTHON", sys.executable)

pilot = subprocess.Popen(
    [PY, os.path.join(ENF, "pilot453_v4.py")],
    cwd=ENF, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
    stderr=open(os.path.join(SEAT, "pilot_stderr.log"), "ab"),
    text=True, bufsize=1)

def read_reply():
    lines = []
    while True:
        line = pilot.stdout.readline()
        if not line:
            lines.append("<pilot stdout closed>")
            break
        if line.strip() == ">>>":
            break
        lines.append(line.rstrip("\n"))
    return "\n".join(lines)

# consume the pilot's startup prompt
read_reply()

lock = threading.Lock()

def handle(conn):
    with conn:
        buf = b""
        while b"\n" not in buf:
            chunk = conn.recv(4096)
            if not chunk:
                return
            buf += chunk
        cmd = buf.decode().strip()
        with lock:
            if pilot.poll() is not None:
                conn.sendall(b"<pilot process has exited>\n")
                return
            pilot.stdin.write(cmd + "\n")
            pilot.stdin.flush()
            if cmd == "quit":
                conn.sendall(b"<pilot quit>\n")
                return
            conn.sendall((read_reply() + "\n").encode())

if os.path.exists(SOCK):
    os.unlink(SOCK)
srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
srv.bind(SOCK)
srv.listen(4)
print("seat server listening on", SOCK, "pilot pid", pilot.pid, flush=True)
while True:
    conn, _ = srv.accept()
    threading.Thread(target=handle, args=(conn,), daemon=True).start()
