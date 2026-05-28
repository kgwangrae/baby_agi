from __future__ import annotations

import json
import os
import re
import select
import sys
import termios
import time
import tty
from collections import Counter
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import chromadb

from body import BodyState


class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


RUNTIME_STATE_PATH = Path("runtime_state.json")
DUMP_DIR = Path("debug_dumps")
CHROMA_RETRY_COUNT = 3
CHROMA_RETRY_BASE_DELAY = 0.5
CJK_IDEOGRAPH_RE = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]")
CLEANUP_SAMPLE_LIMIT = 12
_NO_CHROMA_FALLBACK = object()


class NonBlockingKeyboard:
    def __enter__(self) -> "NonBlockingKeyboard":
        self.file_descriptor = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.file_descriptor)
        tty.setcbreak(self.file_descriptor)
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        termios.tcsetattr(self.file_descriptor, termios.TCSADRAIN, self.old_settings)

    @staticmethod
    def get_key() -> str:
        readable, _, _ = select.select([sys.stdin], [], [], 0)
        if readable:
            return sys.stdin.read(1).lower()
        return ""


def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def get_chroma_client(db_path: str) -> Any | None:
    return retry_chroma(lambda: chromadb.PersistentClient(path=db_path), fallback=None)


def get_collection(client: Any, collection_name: str) -> Any | None:
    return retry_chroma(lambda: client.get_or_create_collection(name=collection_name), fallback=None)


def retry_chroma(operation: Callable[[], Any], fallback: Any = _NO_CHROMA_FALLBACK) -> Any:
    last_error: Exception | None = None
    for attempt in range(CHROMA_RETRY_COUNT):
        try:
            return operation()
        except Exception as error:
            last_error = error
            if attempt < CHROMA_RETRY_COUNT - 1:
                time.sleep(CHROMA_RETRY_BASE_DELAY * (2 ** attempt))

    if fallback is not _NO_CHROMA_FALLBACK:
        return fallback
    if last_error:
        raise last_error
    return fallback


def view_overview(poll_interval: int = 3) -> None:
    with NonBlockingKeyboard() as keyboard:
        while True:
            clear_screen()
            print(f"{Colors.BOLD}{Colors.HEADER}=== AGI Runtime Overview ==={Colors.ENDC}")
            print("-" * 80)

            runtime_state = load_json_file(RUNTIME_STATE_PATH)
            if runtime_state:
                print_runtime_state(runtime_state)
            else:
                print("runtime_state.json does not exist yet. Run main.py first.")

            print("\n" + "-" * 80)
            print_memory_counts()
            print("\nPress q = menu, d = dump this view.")

            key = _sleep_with_keys(keyboard, poll_interval)
            if key == "q":
                return
            if key == "d":
                dump_payload("runtime", runtime_state)


def view_memory(db_path: str = "./memory_db", poll_interval: int = 3, top_n: int = 20) -> None:
    print(f"{Colors.CYAN}Waiting for Memory DB...{Colors.ENDC}")
    client = wait_for_client(db_path)
    hot_collection = wait_for_collection(client, "hot_episodic")
    cold_collection = wait_for_collection(client, "cold_archive")

    with NonBlockingKeyboard() as keyboard:
        while True:
            clear_screen()
            hot_count = collection_count(hot_collection)
            cold_count = collection_count(cold_collection)
            total_count = hot_count + cold_count

            print(f"{Colors.BOLD}{Colors.HEADER}=== AGI Memory & Thought View ==={Colors.ENDC}")

            memories = collect_memories(hot_collection, cold_collection)
            if total_count > 0:
                print_memory_summary(memories, hot_count, cold_count)
                print("-" * 80)

                for mem_item in memories[:top_n]:
                    print_memory_item(mem_item)
            else:
                print("No memories stored yet.")

            print("Press q = menu, d = dump memory DB.")
            key = _sleep_with_keys(keyboard, poll_interval)
            if key == "q":
                return
            if key == "d":
                dump_payload("memory", memories)


def view_cold_archive(db_path: str = "./memory_db", poll_interval: int = 3, top_n: int = 50) -> None:
    print(f"{Colors.CYAN}Waiting for Cold Archive...{Colors.ENDC}")
    client = wait_for_client(db_path)
    cold_collection = wait_for_collection(client, "cold_archive")

    with NonBlockingKeyboard() as keyboard:
        while True:
            clear_screen()
            cold_count = collection_count(cold_collection)
            print(f"{Colors.BOLD}{Colors.HEADER}=== AGI Cold Archive View ==={Colors.ENDC}")
            print(f"Cold memories shown: {min(cold_count, top_n)} | DB count: {Colors.BLUE}COLD {cold_count}{Colors.ENDC}")
            print("-" * 80)

            memories = collect_collection_memories(cold_collection, "COLD")
            if memories:
                memories.sort(key=lambda x: x["metadata"].get("time", ""), reverse=True)
                for mem_item in memories[:top_n]:
                    print_memory_item(mem_item)
            else:
                print("No cold memories stored yet.")

            print("Press q = menu, d = dump cold archive.")
            key = _sleep_with_keys(keyboard, poll_interval)
            if key == "q":
                return
            if key == "d":
                dump_payload("cold_archive", memories)


def export_full_text_dump(memory_path: str = "./memory_db", emotion_path: str = "./emotion_db") -> None:
    clear_screen()
    print(f"{Colors.BOLD}{Colors.HEADER}=== Export Full Memory & Emotion Space ==={Colors.ENDC}\n")

    print("Connecting to AGI Databases...")
    mem_client = get_chroma_client(memory_path)
    emo_client = get_chroma_client(emotion_path)

    if not mem_client or not emo_client:
        print(f"{Colors.FAIL}[Error] Database connection failed.{Colors.ENDC}")
        time.sleep(1.5)
        return

    hot_coll = get_collection(mem_client, "hot_episodic")
    cold_coll = get_collection(mem_client, "cold_archive")
    emo_coll = get_collection(emo_client, "emotion_space")

    hot_count = collection_count(hot_coll)
    cold_count = collection_count(cold_coll)
    emo_count = collection_count(emo_coll)
    total_entries = hot_count + cold_count + emo_count

    print(f"Current Database Entry Counts:")
    print(f" - Memory HOT: {hot_count} | Memory COLD: {cold_count} | Emotion Nodes: {emo_count}")
    print(f"Total DB Entries: {Colors.BLUE}{total_entries}{Colors.ENDC}\n")

    dump_limit = total_entries
    if total_entries > 2000:
        print(f"{Colors.WARNING}[Warning] Total entries exceed 2000.{Colors.ENDC}")
        choice = input("Force dump all entries without truncation? (y/n): ").strip().lower()
        if choice != 'y':
            print("Export aborted.")
            time.sleep(1)
            return

    print(f"\n{Colors.CYAN}Extracting databases (Zero truncation mode)...{Colors.ENDC}")

    hot_data = retry_chroma(lambda: hot_coll.get(limit=dump_limit), fallback={}) if hot_coll and dump_limit > 0 else {}
    cold_data = retry_chroma(lambda: cold_coll.get(limit=dump_limit), fallback={}) if cold_coll and dump_limit > 0 else {}
    emo_data = retry_chroma(lambda: emo_coll.get(limit=dump_limit), fallback={}) if emo_coll and dump_limit > 0 else {}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    DUMP_DIR.mkdir(exist_ok=True)
    export_path = DUMP_DIR / f"full_agi_text_dump_{timestamp}.txt"

    try:
        with export_path.open("w", encoding="utf-8") as f:
            f.write(f"=== BABY-AGI FULL DATA TEXT DUMP ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ===\n")
            f.write(f"Summary -> Memory HOT: {hot_count} | Memory COLD: {cold_count} | Emotion Nodes: {emo_count}\n")
            f.write("=" * 100 + "\n\n")

            f.write("--- PART 1. EMOTION SPACE (SYSTEM 0) ---\n")
            if isinstance(emo_data, dict) and emo_data.get("ids"):
                for idx, (doc_id, doc, meta) in enumerate(zip(emo_data["ids"], emo_data.get("documents", []), emo_data.get("metadatas", []))):
                    meta = meta or {}
                    f.write(f"[{idx + 1}] NODE_ID: {doc_id} | Valence Tint: {_read_numeric_field(meta, 'valence', 0.0):+.4f}\n")
                    f.write(f"Raw Text Context:\n{doc}\n")
                    f.write("-" * 60 + "\n")

            f.write("\n" + "=" * 100 + "\n\n")
            f.write("--- PART 2. HOT EPISODIC MEMORY (LEVEL 3) ---\n")
            if isinstance(hot_data, dict) and hot_data.get("ids"):
                for idx, (doc_id, doc, meta) in enumerate(zip(hot_data["ids"], hot_data.get("documents", []), hot_data.get("metadatas", []))):
                    meta = meta or {}
                    f.write(f"[{idx + 1}] ID: {doc_id} | Kind: {meta.get('kind', '')} | Time: {meta.get('time', '')}\n")
                    f.write(
                        f"Affective State: Emotion={meta.get('emotion', '')} | "
                        f"ARO={_read_numeric_field(meta, 'arousal', 0.0):.2f} | "
                        f"VAL={_read_numeric_field(meta, 'valence', 0.0):+.2f} | "
                        f"RPE={_read_numeric_field(meta, 'surprise', 0.0):.2f}\n"
                    )
                    f.write(f"Raw Document:\n{doc}\n")
                    f.write("-" * 60 + "\n")

            f.write("\n" + "=" * 100 + "\n\n")
            f.write("--- PART 3. COLD ARCHIVE MEMORY (LEVEL 3) ---\n")
            if isinstance(cold_data, dict) and cold_data.get("ids"):
                for idx, (doc_id, doc, meta) in enumerate(zip(cold_data["ids"], cold_data.get("documents", []), cold_data.get("metadatas", []))):
                    meta = meta or {}
                    f.write(f"[{idx + 1}] ID: {doc_id} | Kind: {meta.get('kind', '')} | Time: {meta.get('time', '')}\n")
                    f.write(
                        f"Affective State: Emotion={meta.get('emotion', '')} | "
                        f"ARO={_read_numeric_field(meta, 'arousal', 0.0):.2f} | "
                        f"VAL={_read_numeric_field(meta, 'valence', 0.0):+.2f} | "
                        f"RPE={_read_numeric_field(meta, 'surprise', 0.0):.2f}\n"
                    )
                    f.write(f"Raw Document:\n{doc}\n")
                    f.write("-" * 60 + "\n")

        print(f"\n{Colors.GREEN}[Success] Raw text dump saved to: {export_path}{Colors.ENDC}")
    except Exception as error:
        print(f"\n{Colors.FAIL}[Error] Failed to write text dump file: {error}{Colors.ENDC}")

    input("\nPress Enter to return to menu...")


def view_emotion(db_path: str = "./emotion_db", poll_interval: int = 3) -> None:
    print(f"{Colors.CYAN}Waiting for Emotion DB...{Colors.ENDC}")
    client = wait_for_client(db_path)
    emotion_collection = wait_for_collection(client, "emotion_space")
    initial_count = collection_count(emotion_collection)

    with NonBlockingKeyboard() as keyboard:
        while True:
            clear_screen()
            current_count = collection_count(emotion_collection)
            print(f"{Colors.BOLD}{Colors.HEADER}=== AGI Emotion Space View ==={Colors.ENDC}")
            print(f"Total Emotion Nodes: {current_count} (Learned this view: +{current_count - initial_count})")
            print("-" * 80)

            nodes = collect_emotion_nodes(emotion_collection, limit=80)
            if nodes:
                for node in nodes[:30]:
                    valence_color = Colors.GREEN if node["valence"] > 0 else Colors.FAIL
                    print(
                        f"[{valence_color}VAL: {node['valence']:+.2f}{Colors.ENDC}] "
                        f"{node['id']} | {node['doc'][:120]}"
                    )
            else:
                print("No emotion nodes found.")

            print("\nPress q = menu, d = dump emotion DB.")
            key = _sleep_with_keys(keyboard, poll_interval)
            if key == "q":
                return
            if key == "d":
                dump_payload("emotion", nodes)


def view_facts(file_path: str = "facts.json", poll_interval: int = 3) -> None:
    with NonBlockingKeyboard() as keyboard:
        while True:
            clear_screen()
            print(f"{Colors.BOLD}{Colors.HEADER}=== AGI Fact Notepad View ==={Colors.ENDC}")
            print("-" * 80)

            facts = load_json_file(Path(file_path))
            if facts:
                for key, value in facts.items():
                    print(f"{Colors.CYAN}[{key}]{Colors.ENDC} {value}")
            else:
                print("facts.json does not exist yet or is not valid JSON.")

            print("\nPress q = menu, d = dump facts.")
            key_input = _sleep_with_keys(keyboard, poll_interval)
            if key_input == "q":
                return
            if key_input == "d":
                dump_payload("facts", facts)


def cleanup_unsupported_cjk_entries(memory_path: str = "./memory_db", emotion_path: str = "./emotion_db") -> None:
    clear_screen()
    print(f"{Colors.BOLD}{Colors.HEADER}=== Cleanup unsupported CJK entries ==={Colors.ENDC}\n")

    targets = _collect_cleanup_targets(memory_path, emotion_path)
    cjk_entries = []
    for label, collection in targets:
        cjk_entries.extend(_find_cjk_entries(label, collection))

    if not cjk_entries:
        print(f"{Colors.GREEN}No unsupported CJK entries found.{Colors.ENDC}")
        input("\nPress Enter to return to menu...")
        return

    print(f"Found {Colors.FAIL}{len(cjk_entries)}{Colors.ENDC} entrie(s) containing unsupported CJK ideographs.")
    print("Hangul Korean text is not targeted. Review the samples before deletion.\n")

    for entry in cjk_entries[:CLEANUP_SAMPLE_LIMIT]:
        preview = entry["document"].replace("\n", " ")[:180]
        print(f"- {entry['label']} | {entry['id']} | {preview}")

    if len(cjk_entries) > CLEANUP_SAMPLE_LIMIT:
        print(f"... and {len(cjk_entries) - CLEANUP_SAMPLE_LIMIT} more.")

    confirmation = input("\nType DELETE to remove these entries, or press Enter to cancel: ").strip()
    if confirmation != "DELETE":
        print("Cleanup canceled.")
        time.sleep(1)
        return

    deleted_count = 0
    for label, collection in targets:
        ids_to_delete = [entry["id"] for entry in cjk_entries if entry["label"] == label]
        if not ids_to_delete:
            continue
        retry_chroma(lambda ids=ids_to_delete, coll=collection: coll.delete(ids=ids), fallback=None)
        deleted_count += len(ids_to_delete)

    print(f"{Colors.GREEN}Deleted {deleted_count} entrie(s).{Colors.ENDC}")
    input("\nPress Enter to return to menu...")


def _collect_cleanup_targets(memory_path: str, emotion_path: str) -> list[tuple[str, Any]]:
    targets: list[tuple[str, Any]] = []

    memory_client = get_chroma_client(memory_path)
    if memory_client:
        hot_collection = get_collection(memory_client, "hot_episodic")
        cold_collection = get_collection(memory_client, "cold_archive")
        if hot_collection is not None:
            targets.append(("memory.hot_episodic", hot_collection))
        if cold_collection is not None:
            targets.append(("memory.cold_archive", cold_collection))

    emotion_client = get_chroma_client(emotion_path)
    if emotion_client:
        emotion_collection = get_collection(emotion_client, "emotion_space")
        if emotion_collection is not None:
            targets.append(("emotion.emotion_space", emotion_collection))

    return targets


def _find_cjk_entries(label: str, collection: Any) -> list[dict[str, str]]:
    count = collection_count(collection)
    if count == 0:
        return []

    data = retry_chroma(lambda: collection.get(limit=count), fallback={})
    ids = data.get("ids") or []
    documents = data.get("documents") or []
    return [
        {"label": label, "id": doc_id, "document": str(document or "")}
        for doc_id, document in zip(ids, documents)
        if CJK_IDEOGRAPH_RE.search(str(document or ""))
    ]


def wait_for_client(db_path: str) -> Any:
    while True:
        client = get_chroma_client(db_path)
        if client:
            return client
        time.sleep(1)


def wait_for_collection(client: Any, collection_name: str) -> Any:
    while True:
        collection = get_collection(client, collection_name)
        if collection is not None:
            return collection
        time.sleep(1)


def collect_memories(hot_collection: Any, cold_collection: Any) -> list[dict[str, Any]]:
    memories: list[dict[str, Any]] = []
    memories.extend(collect_collection_memories(hot_collection, "HOT"))
    memories.extend(collect_collection_memories(cold_collection, "COLD"))
    memories.sort(key=lambda x: x["metadata"].get("time", ""), reverse=True)
    return memories


def collect_collection_memories(collection: Any, tier_label: str) -> list[dict[str, Any]]:
    if collection is None:
        return []
    if collection_count(collection) == 0:
        return []

    data = retry_chroma(lambda: collection.get(limit=200), fallback={})
    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or []
    ids = data.get("ids") or []
    return [
        {
            "id": doc_id,
            "document": document,
            "metadata": metadata or {},
            "tier": tier_label,
        }
        for doc_id, document, metadata in zip(ids, documents, metadatas)
    ]


def collection_count(collection: Any) -> int:
    if collection is None:
        return 0
    return int(retry_chroma(collection.count, fallback=0) or 0)


def collect_emotion_nodes(collection: Any, limit: int) -> list[dict[str, Any]]:
    if collection is None:
        return []
    if collection_count(collection) == 0:
        return []

    data = retry_chroma(lambda: collection.get(limit=limit), fallback={})
    ids = data.get("ids") or []
    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or []
    nodes = [
        {
            "id": doc_id,
            "doc": document,
            "valence": _read_numeric_field(metadata, "valence", 0.0),
        }
        for doc_id, document, metadata in zip(ids, documents, metadatas)
    ]
    nodes.sort(key=lambda node: abs(node["valence"]), reverse=True)
    return nodes


def print_memory_summary(memories: list[dict[str, Any]], hot_count: int, cold_count: int) -> None:
    emotions = [mem_item["metadata"].get("emotion") for mem_item in memories]
    arousals = [_read_numeric_field(mem_item["metadata"], "arousal", 0.0) for mem_item in memories]
    average_arousal = sum(arousals) / len(arousals) if arousals else 0.0
    dominant_emotion = Counter(emotions).most_common(1)[0][0] if emotions else "None"
    print(
        f"Episodes shown: {len(memories)} | DB counts: "
        f"{Colors.FAIL}HOT {hot_count}{Colors.ENDC} / {Colors.BLUE}COLD {cold_count}{Colors.ENDC} | "
        f"Avg ARO: {average_arousal:.2f} | Dom EMO: {dominant_emotion}"
    )


def _read_numeric_field(source: dict[str, Any] | None, key: str, fallback: float = 0.0) -> float:
    if not source:
        return fallback
    return BodyState.coerce_float(source.get(key, fallback), fallback)


def _format_score(source: dict[str, Any] | None, key: str) -> str:
    return f"{_read_numeric_field(source, key, 0.0):.2f}"


def _estimate_body_arousal(metadata: dict[str, Any] | None) -> float:
    if not metadata:
        return 0.0
    if "body_arousal" in metadata:
        return _read_numeric_field(metadata, "body_arousal", 0.0)
    return BodyState.estimate_arousal(
        arousal=_read_numeric_field(metadata, "arousal", 0.0),
        valence=_read_numeric_field(metadata, "valence", 0.0),
        surprise=_read_numeric_field(metadata, "surprise", 0.0),
    )


def print_memory_item(memory: dict[str, Any]) -> None:
    metadata = memory.get("metadata") or {}
    arousal = _read_numeric_field(metadata, "arousal", 0.0)
    valence = _read_numeric_field(metadata, "valence", 0.0)
    surprise = _read_numeric_field(metadata, "surprise", 0.0)
    body_arousal = _estimate_body_arousal(metadata)
    body_fatigue = _read_numeric_field(metadata, "body_fatigue", 0.0)
    body_pressure = _read_numeric_field(metadata, "body_sleep_pressure", body_fatigue - body_arousal)
    body_label = (
        f"B-ARO:{body_arousal:.2f} FAT:{body_fatigue:.2f} PRESS:{body_pressure:+.2f}"
        if "body_fatigue" in metadata
        else f"B-ARO≈{body_arousal:.2f} FAT:n/a PRESS:n/a"
    )
    memory_kind = str(metadata.get("kind", "episode"))
    time_text = str(metadata.get("time", "unknown"))[-15:]
    content = str(memory.get("document", ""))

    arousal_color = Colors.FAIL if arousal > 0.6 else Colors.CYAN

    print(
        f"[{memory.get('tier', '?')}] {time_text} | id={memory.get('id', '?')} | "
        f"ARO:{arousal_color}{arousal:.2f}{Colors.ENDC} | "
        f"VAL:{valence:+.2f} | RPE:{surprise:.2f} | {body_label} | {memory_kind}"
    )

    if memory_kind == "consolidated":
        consolidated_text = content.replace("[CONSOLIDATED]", "").strip()
        emotion_text = str(metadata.get("emotion", "") or "N/A")

        print(
            f"  Affective Trace: "
            f"ARO={arousal:.2f}, VAL={valence:+.2f}, RPE={surprise:.2f}, EMO={emotion_text}"
        )
        print(f"  Body Trace: {body_label}")
        print(f"  Consolidated: {Colors.CYAN}{consolidated_text[:320]}{Colors.ENDC}")
        print(f"  Raw: {content[:320]}\n")
        return

    context_text = extract_between(content, ["Context:"], ["| Expect:", "| Thought:", "| Spoke:"]) or "N/A"
    baby_expect = extract_between(content, ["Expect:"], ["| Thought:", "| Spoke:"]) or "N/A"
    baby_thought = extract_between(content, ["Thought:"], ["| Spoke:", "|Spoke:"]) or "N/A"
    baby_spoke = extract_between(content, ["Spoke:"], []) or "N/A"

    print(f"  Context: {context_text[:220]}")
    print(f"  Expectations: {Colors.WARNING}{baby_expect}{Colors.ENDC}")
    print(f"  Thought: {Colors.CYAN}{baby_thought[:220]}{Colors.ENDC}")
    print(f"  Spoke: {Colors.GREEN}{baby_spoke[:220]}{Colors.ENDC}")
    print(f"  Raw: {content[:260]}\n")


def print_runtime_state(runtime_state: dict[str, Any]) -> None:
    print(f"Time: {runtime_state.get('time', 'unknown')}")
    print(f"Eyes: {'open' if runtime_state.get('eyes_enabled') else 'closed'}")
    print(f"Sleep source: {runtime_state.get('sleep_source') or 'none'}")
    print(f"Emotion: {runtime_state.get('emotion_token', 'N/A')}")

    body_state = runtime_state.get("body_state") or {}
    body_arousal = _read_numeric_field(body_state, "arousal", _read_numeric_field(runtime_state, "arousal", 0.0))
    body_fatigue = _read_numeric_field(body_state, "fatigue", _read_numeric_field(runtime_state, "fatigue", 0.0))
    arousal_barrier = _read_numeric_field(body_state, "arousal_barrier", min(1.0, body_arousal + body_arousal * body_arousal))
    sleep_pressure = _read_numeric_field(body_state, "sleep_pressure", body_fatigue - arousal_barrier)
    print(
        f"Arousal: {_read_numeric_field(runtime_state, 'arousal', 0.0):.2f} | "
        f"Body ARO: {body_arousal:.2f} | "
        f"Fatigue: {body_fatigue:.2f} | "
        f"Arousal barrier: {arousal_barrier:.2f} | "
        f"Sleep pressure: {sleep_pressure:+.2f} | "
        f"Valence: {_read_numeric_field(runtime_state, 'valence', 0.0):.2f} | "
        f"Mood: {_read_numeric_field(runtime_state, 'mood', 0.0):.2f} | "
        f"RPE: {_read_numeric_field(runtime_state, 'surprise', 0.0):.2f}"
    )

    expect = runtime_state.get("previous_expected_emotions", {})
    print(
        f"Predicted Expect -> {Colors.GREEN}JOY: {_format_score(expect, 'JOY')}{Colors.ENDC} | "
        f"{Colors.BLUE}SAD: {_format_score(expect, 'SAD')}{Colors.ENDC} | "
        f"{Colors.FAIL}ANG: {_format_score(expect, 'ANG')}{Colors.ENDC}"
    )
    print("-" * 40)
    print(f"Last Dad: {runtime_state.get('last_user_message') or 'N/A'}")
    print(f"Last Baby: {runtime_state.get('last_response') or 'N/A'}")
    print(f"Last Thought: {runtime_state.get('last_inner_monologue') or 'N/A'}")
    print(f"Last Screen: {runtime_state.get('last_visual_summary') or 'N/A'}")


def print_memory_counts() -> None:
    memory_client = get_chroma_client("./memory_db")
    emotion_client = get_chroma_client("./emotion_db")
    if memory_client:
        hot_collection = get_collection(memory_client, "hot_episodic")
        cold_collection = get_collection(memory_client, "cold_archive")
        if hot_collection is not None and cold_collection is not None:
            hot_count = collection_count(hot_collection)
            cold_count = collection_count(cold_collection)
            print(f"Memory DB: HOT={hot_count}, COLD={cold_count}")
        else:
            print("Memory DB: temporarily locked")
    else:
        print("Memory DB: not available")

    if emotion_client:
        emotion_collection = get_collection(emotion_client, "emotion_space")
        if emotion_collection is not None:
            emotion_count = collection_count(emotion_collection)
            print(f"Emotion DB: nodes={emotion_count}")
        else:
            print("Emotion DB: temporarily locked")
    else:
        print("Emotion DB: not available")


def extract_between(text: str, start_tokens: list[str], end_tokens: list[str]) -> str:
    start_index = -1
    start_token_length = 0
    for token in start_tokens:
        start_index = text.find(token)
        if start_index != -1:
            start_token_length = len(token)
            break

    if start_index == -1:
        return ""

    value_start = start_index + start_token_length
    if not end_tokens:
        return text[value_start:].strip()

    end_positions = [text.find(token, value_start) for token in end_tokens]
    valid_end_positions = [position for position in end_positions if position != -1]
    value_end = min(valid_end_positions) if valid_end_positions else len(text)
    return text[value_start:value_end].strip()


def load_json_file(file_path: Path) -> dict[str, Any]:
    if not file_path.exists():
        return {}
    try:
        with file_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def dump_payload(view_name: str, payload: Any) -> None:
    DUMP_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dump_path = DUMP_DIR / f"{view_name}_dump_{timestamp}.json"
    with dump_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    print(f"\n[System] Dump saved: {dump_path}")
    time.sleep(1.2)


def _sleep_with_keys(keyboard: NonBlockingKeyboard, total_seconds: int) -> str:
    end_time = time.time() + total_seconds
    while time.time() < end_time:
        key = keyboard.get_key()
        if key in {"q", "d"}:
            return key
        time.sleep(0.1)
    return ""


def run_menu() -> None:
    while True:
        clear_screen()
        print(f"{Colors.BOLD}Select Debug View:{Colors.ENDC}")
        print("1. Runtime Overview")
        print("2. Memory & Thoughts (Hot/Cold Episodic)")
        print("3. Emotion Vector Space (System 0)")
        print("4. Fact Notepad (Deterministic Facts)")
        print("5. Cold Archive only")
        print("6. Export Full Memory & Emotion Space (Text Dump)")
        print("7. Cleanup unsupported CJK entries")
        print("q. Quit")

        choice = input("\nEnter number (1-7): ").strip().lower()
        if choice == "1":
            view_overview()
        elif choice == "2":
            view_memory()
        elif choice == "3":
            view_emotion()
        elif choice == "4":
            view_facts()
        elif choice == "5":
            view_cold_archive()
        elif choice == "6":
            export_full_text_dump()
        elif choice == "7":
            cleanup_unsupported_cjk_entries()
        elif choice == "q":
            return
        else:
            print("Invalid selection.")
            time.sleep(1)


if __name__ == "__main__":
    run_menu()
