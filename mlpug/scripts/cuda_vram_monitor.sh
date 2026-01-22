#!/bin/bash
# CUDA VRAM Monitor
# Logs GPU memory usage at regular intervals for all GPUs in compact format.
#
# Usage:
#   ./cuda_vram_monitor.sh [interval_seconds] [output_file]
#
# Examples:
#   ./cuda_vram_monitor.sh                          # 5s interval, stdout
#   ./cuda_vram_monitor.sh 5 vram.log              # 5s interval, file
#   ./cuda_vram_monitor.sh 2                        # 2s interval, stdout
#
# Output format (one line per observation):
#   HH:MM:SS | GPU0 GPU1 ... %
#   14:28:14 | 91 91 %
#
# Stop with Ctrl+C or: pkill -f cuda_vram_monitor

INTERVAL=${1:-5}
OUTPUT=${2:-/dev/stdout}

echo "CUDA VRAM Monitor started (interval: ${INTERVAL}s, output: ${OUTPUT})"
echo "Press Ctrl+C to stop"

while true; do
    VRAM=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | \
        awk -F, '{printf "%d ", ($1/$2)*100}')
    echo "$(date +%H:%M:%S) | ${VRAM}%"
    sleep "$INTERVAL"
done >> "$OUTPUT"
