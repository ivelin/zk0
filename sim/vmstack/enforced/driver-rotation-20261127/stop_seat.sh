#!/bin/bash
# Stop the pilot (clean quit) and the seat server.
set -u
SEAT="$(cd "$(dirname "$0")/.." && pwd)"
"$SEAT/venv/bin/python" "$SEAT/seatbin/seat.py" quit >/dev/null 2>&1 || true
sleep 1
pkill -f "seatbin/seat_server.py" 2>/dev/null || true
rm -f "$SEAT/seat.sock"
echo "seat stopped"
