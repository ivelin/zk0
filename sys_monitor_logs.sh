#!/bin/bash

# monitor_logs.sh - Script to monitor GPU and system metrics during zk0 training sessions
# Logs nvidia-smi output to gpu_monitor.log and system sensors/temps to system_temps.log
# Run this before starting a tiny training session to capture data for post-restart analysis
# Files are appended to in the current directory (/home/ivelin/zk0) and survive restarts

set -e  # Exit on any error

LOG_DIR="$(pwd)/tmp"
GPU_LOG="${LOG_DIR}/gpu_monitor.log"
TEMP_LOG="${LOG_DIR}/system_temps.log"
TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')

# Function to log with timestamp
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${LOG_DIR}/monitor_setup.log"
}

log_message "Starting monitoring script at ${TIMESTAMP}. Logs will append to ${GPU_LOG} and ${TEMP_LOG}."

# Check for nvidia-smi
if command -v nvidia-smi &> /dev/null; then
    log_message "Starting GPU monitoring with nvidia-smi -l 1 (polling every 1s)."
    # Run nvidia-smi in background, appending timestamped output
    (
        while true; do
            echo "=== GPU Snapshot at $(date '+%Y-%m-%d %H:%M:%S') ===" >> "${GPU_LOG}"
            nvidia-smi >> "${GPU_LOG}"
            sleep 1
        done
    ) &
    GPU_PID=$!
    log_message "GPU monitor PID: ${GPU_PID}"
    echo ${GPU_PID} > "${LOG_DIR}/gpu_monitor.pid"
else
    log_message "WARNING: nvidia-smi not found. Install NVIDIA drivers for GPU monitoring."
fi

# Check for sensors (lm-sensors)
if command -v sensors &> /dev/null; then
    log_message "Starting system temperature monitoring with sensors (polling every 1s)."
    # Run sensors in background, appending timestamped output
    (
        while true; do
            echo "=== System Temps at $(date '+%Y-%m-%d %H:%M:%S') ===" >> "${TEMP_LOG}"
            sensors >> "${TEMP_LOG}"
            sleep 1
        done
    ) &
    TEMP_PID=$!
    log_message "Sensors monitor PID: ${TEMP_PID}"
    echo ${TEMP_PID} > "${LOG_DIR}/sensors_monitor.pid"
else
    log_message "WARNING: sensors not found. Install lm-sensors (sudo apt install lm-sensors) for full temp monitoring."
    log_message "Falling back to basic CPU thermal zones via /sys/class/thermal."
    # Fallback: Poll CPU thermal zones
    (
        while true; do
            echo "=== CPU Thermals at $(date '+%Y-%m-%d %H:%M:%S') ===" >> "${TEMP_LOG}"
            if [ -d /sys/class/thermal ]; then
                for zone in /sys/class/thermal/thermal_zone*; do
                    if [ -f "${zone}/type" ] && [ -f "${zone}/temp" ]; then
                        temp=$(cat "${zone}/temp")
                        temp_c=$((temp / 1000))  # Convert millidegrees C to C
                        type=$(cat "${zone}/type")
                        echo "${type}: ${temp_c}°C" >> "${TEMP_LOG}"
                    fi
                done
            else
                echo "No thermal zones found." >> "${TEMP_LOG}"
            fi
            # Basic CPU usage from /proc/stat (simplified load average)
            echo "CPU Load (1/5/15 min): $(cat /proc/loadavg | cut -d' ' -f1-3)" >> "${TEMP_LOG}"
            sleep 1
        done
    ) &
    TEMP_PID=$!
    log_message "Fallback temp monitor PID: ${TEMP_PID}"
    echo ${TEMP_PID} > "${LOG_DIR}/fallback_monitor.pid"
fi

# Optional: CPU usage monitoring (always run, simple top-like)
log_message "Starting basic CPU usage monitoring (polling every 5s for less noise)."
(
    while true; do
        echo "=== CPU Usage at $(date '+%Y-%m-%d %H:%M:%S') ===" >> "${TEMP_LOG}"
        # Simple CPU percentage from /proc/stat (requires two samples, but approximate here)
        top -bn1 | grep "Cpu(s)" | awk '{print $2 " % user, " $4 " % sys, " $8 " % idle"}' >> "${TEMP_LOG}"
        # Memory usage
        free -h | grep Mem: >> "${TEMP_LOG}"
        sleep 5
    done
) &
CPU_PID=$!
log_message "CPU monitor PID: ${CPU_PID}"
echo ${CPU_PID} > "${LOG_DIR}/cpu_monitor.pid"

log_message "All monitors started. To stop: kill $(cat ${LOG_DIR}/*.pid 2>/dev/null || echo 'PIDs in .pid files') or killall bash (carefully)."
log_message "Run 'tail -f gpu_monitor.log' or 'tail -f system_temps.log' to watch live during training."
log_message "After restart, inspect logs for power/temp spikes around crash time."

# Keep script running in foreground until killed (user can Ctrl+C or kill PID)
wait