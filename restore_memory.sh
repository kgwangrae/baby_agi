#!/bin/sh
set -eu

if [ -z "${1:-}" ]; then
    echo "Usage: ./restore_memory.sh <backup_file.zip>"
    exit 1
fi

if [ ! -f "$1" ]; then
    echo "[System] Backup file not found: $1"
    exit 1
fi

echo "This will overwrite current memory_db, emotion_db, facts.json, and runtime_state.json."
printf "Type YES to continue: "
read answer

if [ "$answer" != "YES" ]; then
    echo "[System] Cancelled. Please type YES exactly to try again."
    exit 0
fi

rm -rf memory_db emotion_db facts.json runtime_state.json debug_dumps
unzip -o "$1"
echo "[System] Memory restored from $1"
