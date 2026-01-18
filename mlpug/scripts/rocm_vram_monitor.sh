#!/bin/bash
# ROCm VRAM Monitor
# Logs GPU memory usage at regular intervals for all GPUs in compact format.
#
# Usage:
#   ./rocm_vram_monitor.sh [interval_seconds] [output_file]
#
# Examples:
#   ./rocm_vram_monitor.sh                          # 5s interval, stdout
#   ./rocm_vram_monitor.sh 5 vram.log              # 5s interval, file
#   ./rocm_vram_monitor.sh 2                        # 2s interval, stdout
#
# Output format (one line per observation):
#   HH:MM:SS | GPU0 GPU1 GPU2 GPU3 GPU4 GPU5 %
#   14:28:14 | 91 91 91 91 91 91 %
#
# Stop with Ctrl+C or: pkill -f rocm_vram_monitor

INTERVAL=${1:-5}
OUTPUT=${2:-/dev/stdout}

echo "ROCm VRAM Monitor started (interval: ${INTERVAL}s, output: ${OUTPUT})"
echo "Press Ctrl+C to stop"

while true; do
    VRAM=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | tail -n +2 | head -6 | \
        awk -F, '{used=$3/$2*100; printf "%d ", used}')
    echo "$(date +%H:%M:%S) | ${VRAM}%"
    sleep "$INTERVAL"
done >> "$OUTPUT"
