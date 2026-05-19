#!/bin/sh
set -eu

for item in memory_db emotion_db facts.json runtime_state.json; do
    if [ -e "$item" ]; then
        ls -ahl "$item"
    fi
done

echo "This will delete memory_db, emotion_db, facts.json, and runtime_state.json."
printf "Type YES to continue: "
read answer

if [ "$answer" != "YES" ]; then
    echo "[System] Cancelled."
    exit 0
fi

rm -rf memory_db emotion_db facts.json runtime_state.json
echo "[System] Memory cleared."
