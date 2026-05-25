#!/bin/sh
set -eu

backup_file="backup_memory_$(date +'%Y_%m_%d_%H%M').zip"
items=""

for item in memory_db emotion_db facts.json runtime_state.json recent_context.json debug_dumps; do
    if [ -e "$item" ]; then
        items="$items $item"
    fi
done

if [ -z "$items" ]; then
    echo "[System] Nothing to back up."
    exit 0
fi

# shellcheck disable=SC2086
zip -r "$backup_file" $items -x "*.DS_Store" -x "__MACOSX"
ls -al "$backup_file"
