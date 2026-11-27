#!/bin/bash
# Start the driver-seat server (idempotent). Rendering: EGL first (DGX GPU),
# the pilot itself logs render failures to live-watch/render_error.txt.
set -u
SEAT="$(cd "$(dirname "$0")/.." && pwd)"
if [ -S "$SEAT/seat.sock" ] && "$SEAT/venv/bin/python" "$SEAT/seatbin/seat.py" state >/dev/null 2>&1; then
  echo "seat already running"
  exit 0
fi
rm -f "$SEAT/seat.sock"
cd "$SEAT"
MUJOCO_GL="${MUJOCO_GL:-egl}" SEAT_PYTHON="$SEAT/venv/bin/python" \
  nohup "$SEAT/venv/bin/python" "$SEAT/seatbin/seat_server.py" \
  > "$SEAT/seat_server.log" 2>&1 &
for i in $(seq 1 30); do
  [ -S "$SEAT/seat.sock" ] && sleep 1 && \
    "$SEAT/venv/bin/python" "$SEAT/seatbin/seat.py" state >/dev/null 2>&1 && {
      echo "seat up"; exit 0; }
  sleep 1
done
echo "seat failed to start; see seat_server.log and pilot_stderr.log"
exit 1
