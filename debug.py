from __future__ import annotations

import json
import os
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
_NO_CHROMA_FALLBACK = object()


class NonBlockingKeyboard:
    def __enter__(self) -> "NonBlockingKeyboard":
        self.file_descriptor = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.file_descriptor)
        tty.setcbreak(self.file_descriptor)
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        termios.tcsetattr(self.file_descriptor, termios.TCSADRAIN, self.old_settings)

    def get_key(self) -> str:
        readable, _, _ = select.select([sys.stdin], [], [], 0)
        if readable:
            return sys.stdin.read(1).lower()
        return ""


def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def get_chroma_client(db_path: str) -> Any | None:
    return retry_chroma(lambda: chromadb.PersistentClient(path=db_path), fallback=None)


def get_collection(client: Any, collection_name: str) -> Any:
    return retry_chroma(lambda: client.get_or_create_collection(name=collection_name))


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
    hot_collection = get_collection(client, "hot_episodic")
    cold_collection = get_collection(client, "cold_archive")

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

                for memory in memories[:top_n]:
                    print_memory_item(memory)
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
    cold_collection = get_collection(client, "cold_archive")

    with NonBlockingKeyboard() as keyboard:
        while True:
            clear_screen()
            cold_count = collection_count(cold_collection)
            print(f"{Colors.BOLD}{Colors.HEADER}=== AGI Cold Archive View ==={Colors.ENDC}")
            print(f"Cold memories shown: {min(cold_count, top_n)} | DB count: {Colors.BLUE}COLD {cold_count}{Colors.ENDC}")
            print("-" * 80)

            memories = collect_collection_memories(cold_collection, "COLD")
            if memories:
                memories.sort(key=lambda memory: memory["metadata"].get("time", ""), reverse=True)
                for memory in memories[:top_n]:
                    print_memory_item(memory)
            else:
                print("No cold memories stored yet.")

            print("Press q = menu, d = dump cold archive.")
            key = _sleep_with_keys(keyboard, poll_interval)
            if key == "q":
                return
            if key == "d":
                dump_payload("cold_archive", memories)


def view_emotion(db_path: str = "./emotion_db", poll_interval: int = 3) -> None:
    print(f"{Colors.CYAN}Waiting for Emotion DB...{Colors.ENDC}")
    client = wait_for_client(db_path)
    emotion_collection = get_collection(client, "emotion_space")
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


def wait_for_client(db_path: str) -> Any:
    while True:
        client = get_chroma_client(db_path)
        if client:
            return client
        time.sleep(1)


def collect_memories(hot_collection: Any, cold_collection: Any) -> list[dict[str, Any]]:
    memories: list[dict[str, Any]] = []
    memories.extend(collect_collection_memories(hot_collection, "HOT"))
    memories.extend(collect_collection_memories(cold_collection, "COLD"))
    memories.sort(key=lambda memory: memory["metadata"].get("time", ""), reverse=True)
    return memories


def collect_collection_memories(collection: Any, tier_label: str) -> list[dict[str, Any]]:
    if collection_count(collection) == 0:
        return []

    data = retry_chroma(lambda: collection.get(limit=200), fallback={})
    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or []
    ids = data.get("ids") or []
    return [
        {
            "id": ids[index],
            "document": documents[index],
            "metadata": metadatas[index],
            "tier": tier_label,
        }
        for index in range(len(ids))
    ]


def collection_count(collection: Any) -> int:
    return int(retry_chroma(collection.count, fallback=0) or 0)


def collect_emotion_nodes(collection: Any, limit: int) -> list[dict[str, Any]]:
    if collection_count(collection) == 0:
        return []

    data = retry_chroma(lambda: collection.get(limit=limit), fallback={})
    nodes = [
        {
            "id": data["ids"][index],
            "doc": data["documents"][index],
            "valence": float(data["metadatas"][index].get("valence", 0.0)),
        }
        for index in range(len(data.get("ids", [])))
    ]
    nodes.sort(key=lambda node: abs(node["valence"]), reverse=True)
    return nodes


def print_memory_summary(memories: list[dict[str, Any]], hot_count: int, cold_count: int) -> None:
    emotions = [memory["metadata"].get("emotion") for memory in memories]
    arousals = [float(memory["metadata"].get("arousal", 0.0)) for memory in memories]
    average_arousal = sum(arousals) / len(arousals) if arousals else 0.0
    dominant_emotion = Counter(emotions).most_common(1)[0][0] if emotions else "None"
    print(
        f"Episodes shown: {len(memories)} | DB counts: "
        f"{Colors.FAIL}HOT {hot_count}{Colors.ENDC} / {Colors.BLUE}COLD {cold_count}{Colors.ENDC} | "
        f"Avg ARO: {average_arousal:.2f} | Dom EMO: {dominant_emotion}"
    )


def print_memory_item(memory: dict[str, Any]) -> None:
    metadata = memory["metadata"]
    arousal = float(metadata.get("arousal", 0.0))
    valence = float(metadata.get("valence", 0.0))
    surprise = float(metadata.get("surprise", 0.0))
    time_text = str(metadata.get("time", "unknown"))[-15:]
    content = memory["document"]

    context_text = extract_between(content, ["Context:"], ["| Expect:", "| Thought:", "| Spoke:"]) or "N/A"
    baby_expect = extract_between(content, ["Expect:"], ["| Thought:", "| Spoke:"]) or "JOY=0.00, SAD=0.00, ANG=0.00"
    baby_thought = extract_between(content, ["Thought:"], ["| Spoke:", "|Spoke:"]) or "N/A"
    baby_spoke = extract_between(content, ["Spoke:"], []) or "N/A"
    arousal_color = Colors.FAIL if arousal > 0.6 else Colors.CYAN

    print(
        f"[{memory['tier']}] {time_text} | id={memory['id']} | "
        f"ARO:{arousal_color}{arousal:.2f}{Colors.ENDC} | "
        f"VAL:{valence:+.2f} | RPE:{surprise:.2f} | {metadata.get('kind', 'episode')}"
    )
    print(f"  Context: {context_text[:220]}")
    print(f"  Expectations: {Colors.WARNING}{baby_expect}{Colors.ENDC}")
    print(f"  Thought: {Colors.CYAN}{baby_thought[:220]}{Colors.ENDC}")
    print(f"  Spoke: {Colors.GREEN}{baby_spoke[:220]}{Colors.ENDC}")
    print(f"  Raw: {content[:260]}\n")


def print_runtime_state(runtime_state: dict[str, Any]) -> None:
    print(f"Time: {runtime_state.get('time', 'unknown')}")
    print(f"Eyes: {'open' if runtime_state.get('eyes_enabled') else 'closed'}")
    print(f"Emotion: {runtime_state.get('emotion_token', 'N/A')}")
    print(
        f"Arousal: {float(runtime_state.get('arousal', 0.0)):.2f} | "
        f"Valence: {float(runtime_state.get('valence', 0.0)):.2f} | "
        f"Mood: {float(runtime_state.get('mood', 0.0)):.2f} | "
        f"RPE: {float(runtime_state.get('surprise', 0.0)):.2f}"
    )

    expect = runtime_state.get("previous_expected_emotions", {})
    print(
        f"Predicted Expect -> {Colors.GREEN}JOY: {expect.get('JOY', 0.0):.2f}{Colors.ENDC} | "
        f"{Colors.BLUE}SAD: {expect.get('SAD', 0.0):.2f}{Colors.ENDC} | "
        f"{Colors.FAIL}ANG: {expect.get('ANG', 0.0):.2f}{Colors.ENDC}"
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
        hot_count = collection_count(get_collection(memory_client, "hot_episodic"))
        cold_count = collection_count(get_collection(memory_client, "cold_archive"))
        print(f"Memory DB: HOT={hot_count}, COLD={cold_count}")
    else:
        print("Memory DB: not available")

    if emotion_client:
        emotion_count = collection_count(get_collection(emotion_client, "emotion_space"))
        print(f"Emotion DB: nodes={emotion_count}")
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
        print("q. Quit")

        choice = input("\nEnter number (1-5): ").strip().lower()
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
        elif choice == "q":
            return
        else:
            print("Invalid selection.")
            time.sleep(1)


if __name__ == "__main__":
    run_menu()
