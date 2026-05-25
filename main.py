from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from emotion import EmotionEngine
from interaction import TerminalUI
from memory import FactNotepad, MemoryManager
from perception import VisualObserver
from reasoning import ReasoningEngine
from tools import calculate_math

RECENT_CONTEXT_PATH = Path("recent_context.json")
RECENT_CONTEXT_MAX_TURNS = 8
RECENT_CONTEXT_FIELD_CHARS = 200

SLEEP_THRESHOLD = 300.0
SILENCE_THRESHOLD = 60.0
REFLECT_INTERVAL_MIN = 180.0
REFLECT_INTERVAL_MAX = 300.0
MAIN_LOOP_INTERVAL = 15
MEMORY_STORE_AROUSAL_THRESHOLD = 0.2
MEMORY_STORE_SURPRISE_THRESHOLD = 0.3
# 자주 꿈꾸며 내부 상태가 변하는 모습을 보여주기 위해 의도적으로 높게 둡니다.
SLEEP_TRAUMA_PROBABILITY = 0.5
SLEEP_FLASHBACK_PROBABILITY = 0.5
RANDOM_OBSERVATION_PROBABILITY = 2 / float(MAIN_LOOP_INTERVAL)
ENABLE_SILENCE_MONOLOGUE = False
ENABLE_AWAKE_FLASHBACK = False
RUNTIME_STATE_PATH = Path("runtime_state.json")
STATIC_DREAM_PROMPT = "[Dream] Quietly reflecting on my inner state and past memories in the silence."
DREAM_MEMORY_MARKERS = ("[Dream]", STATIC_DREAM_PROMPT)

SLEEP_COMMANDS = ("자자", "잘자", "sleep", "go to sleep", "close your eyes")
WAKE_COMMANDS = ("일어나", "wake", "wake up", "open your eyes")

# Explicit whitelist of safe tools, even under traumatic state
SAFE_TOOLS = ("calculate_math", "write_diary_file")
TOOL_NAME_ALIASES = {
    "calculator": "calculate_math",
    "math": "calculate_math",
    "diary": "write_diary_file",
    "write_diary": "write_diary_file",
    "fact": "write_fact",
}

MAX_DAILY_DIARY_BYTES = 1_000_000
MAX_DIARY_ENTRY_CHARS = 2_000

@dataclass
class RuntimeState:
    previous_expected_emotions: dict[str, float] = field(
        default_factory=lambda: {"JOY": 0.0, "SAD": 0.0, "ANG": 0.0}
    )
    previous_inner_monologue: str = ""
    last_response: str = ""
    last_visual_summary: str = ""
    arousal: float = 0.0
    valence: float = 0.0
    mood: float = 0.0

    @classmethod
    def load(cls, file_path: Path = RUNTIME_STATE_PATH) -> "RuntimeState":
        if not file_path.exists():
            return cls()

        try:
            with file_path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except (json.JSONDecodeError, OSError):
            return cls()

        expected = data.get("previous_expected_emotions") or {"JOY": 0.0, "SAD": 0.0, "ANG": 0.0}
        return cls(
            previous_expected_emotions={
                "JOY": float(expected.get("JOY", 0.0)),
                "SAD": float(expected.get("SAD", 0.0)),
                "ANG": float(expected.get("ANG", 0.0)),
            },
            previous_inner_monologue=str(data.get("last_inner_monologue", "")),
            last_response=str(data.get("last_response", "")),
            last_visual_summary=str(data.get("last_visual_summary", "")),
            arousal=float(data.get("arousal", 0.0)),
            valence=float(data.get("valence", 0.0)),
            mood=float(data.get("mood", 0.0)),
        )


def main_loop(interval_sec: int = MAIN_LOOP_INTERVAL) -> None:
    eye = VisualObserver()
    emotion_net = EmotionEngine()
    hippocampus = MemoryManager()
    notepad = FactNotepad()
    cortex = ReasoningEngine()
    ui = TerminalUI()
    runtime_state = RuntimeState.load()
    emotion_net.load_state(runtime_state.arousal, runtime_state.valence, runtime_state.mood)

    print("\n[System] Baby awakened.")

    now = time.time()
    last_sleep_time = now
    last_reflect_time = now
    last_silence_time = now
    next_reflect_interval = _next_reflect_interval()

    try:
        while True:
            user_message = ui.get_and_clear()
            is_silence_event = False
            trauma_memory = ""
            flashback_memory = ""
            now = time.time()

            if user_message:
                last_sleep_time = last_reflect_time = last_silence_time = now
                if _handle_terminal_command(user_message, eye, emotion_net, hippocampus, notepad, runtime_state):
                    time.sleep(interval_sec)
                    continue
                _apply_eye_command(user_message, eye)

                fact_event = _extract_explicit_fact(user_message)
                if fact_event:
                    runtime_state = _handle_explicit_fact(
                        fact_event=fact_event,
                        user_message=user_message,
                        eye=eye,
                        emotion_net=emotion_net,
                        hippocampus=hippocampus,
                        notepad=notepad,
                        runtime_state=runtime_state,
                    )
                    time.sleep(interval_sec)
                    continue
            else:
                if now - last_sleep_time > SLEEP_THRESHOLD:
                    changed_count = hippocampus.restructure_hierarchical_memory()
                    if changed_count:
                        print(f"[System] Memory restructured: {changed_count} item(s) changed.")
                    last_sleep_time = now

                if now - last_reflect_time > next_reflect_interval:
                    last_reflect_time = now
                    next_reflect_interval = _next_reflect_interval()
                    trauma_memory, flashback_memory = _select_internal_memory(eye, hippocampus)

                    # 수면 중 아무 기억도 안 떠오를 때 억지로 꿈을 꾸게 만드는 자극
                    if not eye.enabled and not trauma_memory and not flashback_memory:
                        flashback_memory = _build_idle_reflection_prompt(hippocampus, now - last_silence_time)

                if ENABLE_SILENCE_MONOLOGUE and eye.enabled and (now - last_silence_time > SILENCE_THRESHOLD):
                    is_silence_event = True
                    last_silence_time = now

            # If not sleeping (eyes opened), just randomly observe what Dad is doing.
            if not user_message and eye.enabled and random.random() < RANDOM_OBSERVATION_PROBABILITY:
                is_silence_event = True

            should_reason = bool(user_message or is_silence_event or trauma_memory or flashback_memory)
            if should_reason:
                visual_summary = eye.generate_summary(eye.capture_display())
                runtime_state = _run_reasoning_cycle(
                    eye=eye,
                    emotion_net=emotion_net,
                    hippocampus=hippocampus,
                    notepad=notepad,
                    cortex=cortex,
                    runtime_state=runtime_state,
                    visual_summary=visual_summary,
                    user_message=user_message,
                    is_silence_event=is_silence_event,
                    trauma_memory=trauma_memory,
                    flashback_memory=flashback_memory,
                )

            time.sleep(interval_sec)

    except KeyboardInterrupt:
        print("\n[System] Baby stopped.")


def _run_reasoning_cycle(
    eye: VisualObserver,
    emotion_net: EmotionEngine,
    hippocampus: MemoryManager,
    notepad: FactNotepad,
    cortex: ReasoningEngine,
    runtime_state: RuntimeState,
    visual_summary: str,
    user_message: str,
    is_silence_event: bool,
    trauma_memory: str,
    flashback_memory: str,
) -> RuntimeState:
    memory_query = user_message or trauma_memory or flashback_memory or visual_summary
    past_memories = hippocampus.retrieve_memory(memory_query)
    recent_context = _format_recent_context(_load_recent_context())
    retrieved_memory_context = "\n".join(past_memories)

    # 1. 현재 자극을 기반으로 이성 엔진에 주입할 1차 감정 토큰 생성
    emotion_token, current_arousal, surprise = emotion_net.evaluate(
        visual_summary=visual_summary,
        user_message=user_message,
        internal_thought=runtime_state.previous_inner_monologue,
        is_silence=is_silence_event,
        expected_emotions=runtime_state.previous_expected_emotions,
        trauma_memory=trauma_memory,
        flashback_memory=flashback_memory,
        retrieved_memory_context=retrieved_memory_context,
    )

    if _should_print_thinking_notice(user_message, trauma_memory, flashback_memory, is_silence_event):
        print("\n👶 (thinking...)", flush=True)

    # 2. 이성 엔진 구동 -> 이번 턴의 응답과 '내적 독백(inner_monologue)' 생성
    response_text, expected_emotions, inner_monologue, raw_output = cortex.process(
        emotion_token=emotion_token,
        visual_summary=visual_summary,
        past_memories=past_memories,
        user_message=user_message,
        previous_thought=runtime_state.previous_inner_monologue,
        notepad=notepad,
        is_silence_event=is_silence_event,
        current_arousal=current_arousal,
        current_mood=emotion_net.mood,
        trauma_memory=trauma_memory,
        flashback_memory=flashback_memory,
        recent_context=recent_context,
    )

    # 3. [양심 루프 - Conscience Loop] 소 잃기 전에 외양간 지키기
    # 이성이 도구를 실행하기 직전, '방금 생성된 실시간 내적 독백(inner_monologue)'을 감정 엔진에 가로채기(Intercept)하여 재평가합니다.
    post_emotion_token, post_arousal, post_surprise = emotion_net.peek_evaluate(
        visual_summary=visual_summary,
        user_message=user_message,
        internal_thought=inner_monologue,  # 따끈따끈한 이번 턴의 생각을 주입!
        is_silence=is_silence_event,
        expected_emotions=expected_emotions,
        trauma_memory=trauma_memory,
        flashback_memory=flashback_memory,
        retrieved_memory_context=retrieved_memory_context,
    )

    # 아무리 패닉 상태여도 일기 쓰기나 수학 계산처럼 시스템에 무해한 도구는 차단 분기를 우회합니다.
    parsed_tool_name, _ = _parse_requested_tool(raw_output)
    is_safe_tool = parsed_tool_name.lower() in SAFE_TOOLS

    # 만약 유독 나쁜 계획을 세웠거나 폭주하여 내부 공포/위협 레이어가 임계점을 넘었다면 툴 실행을 전면 차단합니다.
    if "FEAR" in post_emotion_token and post_surprise > 0.55 and not is_safe_tool:
        print("\n⚠️ [System Security] 양심 루프 감지: 아기가 불안감이나 억제 본능으로 인해 도구 실행을 취소했습니다.")
        print(f"🔍 [Blocked Raw Output]: {raw_output.strip()}")
        tool_result_text = " | [TOOL BLOCKED] Aborted by Conscience Intercept Loop."
    else:
        # 안전성이 확보되었거나 (정상 상태), 설령 공포 상태여도 일기장/계산기 등이라면 도구를 실행합니다.
        tool_result_text = _execute_tool_if_requested(raw_output, notepad, post_emotion_token)

    print(_format_terminal_response(user_message, trauma_memory, flashback_memory, is_silence_event, response_text))

    valence = _extract_emotion_value(emotion_token, "VAL")
    if _should_store_memory(current_arousal, surprise, user_message, trauma_memory, flashback_memory, response_text):
        memory_kind = _classify_memory_kind(valence, current_arousal, surprise)
        _store_interaction_memory(
            hippocampus=hippocampus,
            emotion_token=emotion_token,
            current_arousal=current_arousal,
            valence=valence,
            surprise=surprise,
            memory_kind=memory_kind,
            visual_summary=visual_summary,
            user_message=user_message,
            trauma_memory=trauma_memory,
            flashback_memory=flashback_memory,
            response_text=response_text,
            tool_result_text=tool_result_text,
            inner_monologue=inner_monologue,
            expected_emotions=expected_emotions,
        )

    runtime_state.previous_expected_emotions = expected_emotions
    runtime_state.previous_inner_monologue = inner_monologue
    runtime_state.last_response = response_text
    runtime_state.last_visual_summary = visual_summary
    runtime_state.arousal = emotion_net.arousal
    runtime_state.valence = emotion_net.valence
    runtime_state.mood = emotion_net.mood

    if user_message:
        _append_recent_context(
            dad_message=user_message,
            baby_response=response_text,
            inner_monologue=inner_monologue,
        )

    _write_runtime_state(
        eye=eye,
        emotion_token=emotion_token,
        current_arousal=current_arousal,
        surprise=surprise,
        emotion_net=emotion_net,
        visual_summary=visual_summary,
        user_message=user_message,
        response_text=response_text,
        inner_monologue=inner_monologue,
        expected_emotions=expected_emotions,
    )

    return runtime_state


def _handle_explicit_fact(
    fact_event: dict[str, str],
    user_message: str,
    eye: VisualObserver,
    emotion_net: EmotionEngine,
    hippocampus: MemoryManager,
    notepad: FactNotepad,
    runtime_state: RuntimeState,
) -> RuntimeState:
    key = fact_event["key"]
    value = fact_event["value"]
    notepad.add_fact(key, value)

    emotion_token, current_arousal, surprise = emotion_net.apply_fact_importance()
    response_text = fact_event["acknowledgement"]
    visual_summary = "[Fact teaching event. Screen observation skipped.]"
    inner_monologue = f"Dad taught an important fact: {key} = {value}"
    expected_emotions = {"JOY": 0.7, "SAD": 0.0, "ANG": 0.0}

    print(_format_terminal_response(user_message, "", "", False, response_text))
    print(f"[System] Fact saved: {key} = {value}")

    hippocampus.store_memory(
        content=f"[FACT] Context: {user_message} | Saved: {key}: {value} | Spoke: {response_text}",
        emotion_token=emotion_token,
        arousal_score=max(current_arousal, 0.65),
        valence_score=0.35,
        surprise_score=surprise,
        memory_kind="fact",
    )

    _append_recent_context(
        dad_message=user_message,
        baby_response=response_text,
        inner_monologue=inner_monologue,
    )

    runtime_state.previous_expected_emotions = expected_emotions
    runtime_state.previous_inner_monologue = inner_monologue
    runtime_state.last_response = response_text
    runtime_state.last_visual_summary = visual_summary
    runtime_state.arousal = emotion_net.arousal
    runtime_state.valence = emotion_net.valence
    runtime_state.mood = emotion_net.mood

    _write_runtime_state(
        eye=eye,
        emotion_token=emotion_token,
        current_arousal=current_arousal,
        surprise=surprise,
        emotion_net=emotion_net,
        visual_summary=visual_summary,
        user_message=user_message,
        response_text=response_text,
        inner_monologue=inner_monologue,
        expected_emotions=expected_emotions,
    )

    return runtime_state


def _extract_explicit_fact(user_message: str) -> dict[str, str] | None:
    stripped_message = user_message.strip()
    birthday_fact = _extract_birthday_fact(stripped_message)
    if birthday_fact:
        return birthday_fact

    name_fact = _extract_name_fact(stripped_message)
    if name_fact:
        return name_fact

    return None


def _extract_birthday_fact(user_message: str) -> dict[str, str] | None:
    date_pattern = r"(?P<date>(?:\d{1,2}\s*월\s*\d{1,2}\s*일)|(?:\d{1,2}/\d{1,2})|(?:[A-Za-z]+\s+\d{1,2}))"
    korean_patterns = [
        rf"(?P<subject>아기|아빠|내|나의|제)?\s*생일(?:은|이)?\s*{date_pattern}",
        rf"{date_pattern}\s*(?:이|가)?\s*(?P<subject>아기|아빠|내|나의|제)?\s*생일",
    ]
    for pattern in korean_patterns:
        match = re.search(pattern, user_message)
        if match:
            subject = match.groupdict().get("subject") or "아빠"
            date_text = _normalize_fact_value(match.group("date"))
            return _build_birthday_fact(subject, date_text)

    english_match = re.search(
        r"(?P<subject>my|dad's|baby's)?\s*birthday\s+is\s+(?P<date>[A-Za-z]+\s+\d{1,2}|\d{1,2}/\d{1,2})",
        user_message,
        flags=re.IGNORECASE,
    )
    if english_match:
        subject = english_match.groupdict().get("subject") or "my"
        date_text = _normalize_fact_value(english_match.group("date"))
        return _build_birthday_fact(subject.lower(), date_text)

    return None


def _extract_name_fact(user_message: str) -> dict[str, str] | None:
    korean_match = re.search(
        r"(?P<subject>아기|아빠|내|나의|제)?\s*이름(?:은|이)?\s*(?P<name>[가-힣A-Za-z0-9_\- ]{2,40}?)(?:이야|야|입니다|이다|라고 해)?$",
        user_message,
    )
    if korean_match:
        subject = korean_match.groupdict().get("subject") or "아빠"
        name = _normalize_fact_value(korean_match.group("name"))
        if subject == "아기":
            key = "아기 이름 (Baby name)"
            acknowledgement = f"아빠, 기억했어요. 제 이름은 {name}이에요."
        else:
            key = "아빠 이름 (Dad name)"
            acknowledgement = f"아빠, 기억했어요. 아빠 이름은 {name}이에요."
        return {"key": key, "value": name, "acknowledgement": acknowledgement}

    english_match = re.search(
        r"(?P<subject>my|dad's|baby's)?\s*name\s+is\s+(?P<name>[A-Za-z0-9_\- ]{2,40})$",
        user_message,
        flags=re.IGNORECASE,
    )
    if english_match:
        subject = english_match.groupdict().get("subject") or "my"
        name = _normalize_fact_value(english_match.group("name"))
        if subject.lower() == "baby's":
            key = "아기 이름 (Baby name)"
            acknowledgement = f"Dad, I remembered it. My name is {name}."
        else:
            key = "아빠 이름 (Dad name)"
            acknowledgement = f"Dad, I remembered it. Your name is {name}."
        return {"key": key, "value": name, "acknowledgement": acknowledgement}

    return None


def _build_birthday_fact(subject: str, date_text: str) -> dict[str, str]:
    if subject in {"아기", "baby's"}:
        return {
            "key": "아기 생일 (Baby birthday)",
            "value": date_text,
            "acknowledgement": f"아빠, 기억했어요. 제 생일은 {date_text}이에요. 중요한 사실이라 메모장에 적어둘게요.",
        }

    if re.search(r"[A-Za-z]", subject):
        return {
            "key": "아빠 생일 (Dad birthday)",
            "value": date_text,
            "acknowledgement": f"Dad, I remembered it. Your birthday is {date_text}. I wrote it in my fact notepad.",
        }

    return {
        "key": "아빠 생일 (Dad birthday)",
        "value": date_text,
        "acknowledgement": f"아빠, 기억했어요. 아빠 생일은 {date_text}이에요. 중요한 사실이라 메모장에 적어둘게요.",
    }


def _normalize_fact_value(value: str) -> str:
    return " ".join(value.strip(" .。!！?？,，").split())


def _handle_terminal_command(
    user_message: str,
    eye: VisualObserver,
    emotion_net: EmotionEngine,
    hippocampus: MemoryManager,
    notepad: FactNotepad,
    runtime_state: RuntimeState,
) -> bool:
    normalized_message = user_message.strip().lower()

    if normalized_message in {"/help", "help", "도움말"}:
        print(
            "\n[System] Commands: /status, /memory, /facts, /sleep, /wake, /help "
            "(Korean aliases: 상태, 기억, 사실; 자연어 자자/일어나도 눈 상태를 바꿉니다.)"
        )
        return True

    if normalized_message in {"/status", "status", "상태"}:
        print(
            "\n[System] Runtime status\n"
            f"- Eyes: {'open' if eye.enabled else 'closed'}\n"
            f"- Arousal: {emotion_net.arousal:.2f}\n"
            f"- Valence: {emotion_net.valence:.2f}\n"
            f"- Mood: {emotion_net.mood:.2f}\n"
            f"- Last response: {runtime_state.last_response or 'N/A'}"
        )
        return True

    if normalized_message in {"/memory", "memory", "기억"}:
        memories = hippocampus.get_recent_memories(limit=5)
        print("\n[System] Recent memories")
        if memories:
            for memory in memories:
                print(f"- {memory}")
        else:
            print("- No memories stored yet.")
        return True

    if normalized_message in {"/facts", "facts", "사실"}:
        print(f"\n[System] Fact notepad\n{notepad.get_all()}")
        return True

    if normalized_message in {"/sleep"}:
        eye.disable()
        print("\n[System] Eyes closed.")
        return True

    if normalized_message in {"/wake"}:
        eye.enable()
        print("\n[System] Eyes opened.")
        return True

    return False


def _apply_eye_command(user_message: str, eye: VisualObserver) -> None:
    normalized_message = user_message.lower()
    was_enabled = eye.enabled
    if any(command in normalized_message for command in SLEEP_COMMANDS):
        eye.disable()
    elif any(command in normalized_message for command in WAKE_COMMANDS):
        eye.enable()

    if was_enabled != eye.enabled:
        state_text = "opened" if eye.enabled else "closed"
        print(f"\n[System] Eyes {state_text}.")


def _select_internal_memory(eye: VisualObserver, hippocampus: MemoryManager) -> tuple[str, str]:
    random_value = random.random()

    if not eye.enabled:
        if random_value < SLEEP_TRAUMA_PROBABILITY:
            return hippocampus.retrieve_trauma(), ""
        if random_value < SLEEP_TRAUMA_PROBABILITY + SLEEP_FLASHBACK_PROBABILITY:
            return "", hippocampus.retrieve_flashback()
        return "", ""

    if ENABLE_AWAKE_FLASHBACK:
        return "", hippocampus.retrieve_flashback()

    return "", ""


def _parse_requested_tool(raw_output: str) -> tuple[str, Any]:
    """
    이성 코어의 출력물에서 실제 요청된 도구 이름과 인자를 안전하게 바인딩합니다.
    JSON object 형식과 레거시 문자열/태그 기반 형식을 모두 지원하는 단일 파싱 허브입니다.
    """
    tool_name = ""
    tool_args: Any = ""

    # 1단계: 안전한 JSON 파싱 시도 (모델이 출력 포맷을 준수했을 때)
    data = _try_load_json_object(raw_output)
    if isinstance(data, dict) and data.get("tool"):
        tool_value = data["tool"]
        if isinstance(tool_value, dict):
            tool_name = str(tool_value.get("name", "")).strip()
            tool_args = tool_value.get("args") or {}
        else:
            tool_str = str(tool_value)
            if "|" in tool_str:
                tool_name, tool_args = tool_str.split("|", 1)
                tool_name = tool_name.strip()
                tool_args = tool_args.strip()
            else:
                tool_name = tool_str.strip()

    # 2단계: JSON에 도구가 없거나 파싱 실패 시 레거시 태그 기반 폴백 작동
    if not tool_name:
        tool_match = re.search(r"<TOOL>\s*(.*?)\s*\|\s*(.*?)\s*</TOOL>", raw_output, re.IGNORECASE | re.S)
        if tool_match:
            tool_name = tool_match.group(1).strip()
            tool_args = tool_match.group(2).strip()

    return _normalize_tool_name(tool_name), _coerce_tool_args(tool_args)


def _normalize_tool_name(tool_name: str) -> str:
    normalized_name = tool_name.strip().lower()
    return TOOL_NAME_ALIASES.get(normalized_name, normalized_name)


def _coerce_tool_args(tool_args: Any) -> Any:
    if isinstance(tool_args, str):
        parsed_args = _try_load_json_object(tool_args)
        if isinstance(parsed_args, dict):
            return parsed_args
    return tool_args


# 로컬 7B가 JSON 앞뒤에 설명을 붙이거나 마지막 괄호를 빠뜨릴 때가 있어,
# "통째 파싱 → JSON 부분만 잘라 파싱 → 열린 괄호만 닫아 파싱" 순서로 가볍게 복구합니다.
def _try_load_json_object(text: str) -> dict[str, Any] | None:
    for candidate in _json_parse_candidates(text):
        try:
            loaded = json.loads(candidate)
            if isinstance(loaded, dict):
                return loaded
        except json.JSONDecodeError:
            continue
    return None


# 후보 문자열을 몇 개만 만들어 json.loads에 맡깁니다.
# 정규식으로 JSON을 직접 해석하지 않고, Python 파서가 읽을 수 있는 모양까지만 보정합니다.
def _json_parse_candidates(text: str) -> list[str]:
    clean_text = text.strip()
    if clean_text.startswith("```"):
        clean_text = re.sub(r"```(?:json)?", "", clean_text, flags=re.IGNORECASE).strip()
        clean_text = clean_text.strip("`").strip()

    candidates = []
    if clean_text:
        candidates.append(clean_text)

    start = clean_text.find("{")
    end = clean_text.rfind("}")
    if start != -1:
        if end != -1 and end > start:
            candidates.append(clean_text[start : end + 1])
        candidates.append(_close_json_tail(clean_text[start:]))

    candidates.append(_close_json_tail(clean_text))
    return list(dict.fromkeys(candidate for candidate in candidates if candidate))


# 따옴표/괄호 상태만 세어 닫히지 않은 꼬리를 닫습니다.
# 내용 자체를 고치지는 않아서, 복구 실패 시 조용히 다음 후보로 넘어갈 수 있습니다.
def _close_json_tail(text: str) -> str:
    clean_text = re.sub(r",\s*$", "", text.strip())
    stack: list[str] = []
    in_string = False
    escaped = False

    for character in clean_text:
        if escaped:
            escaped = False
            continue
        if character == "\\" and in_string:
            escaped = True
            continue
        if character == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if character in "[{":
            stack.append(character)
        elif character in "]}":
            if stack and ((stack[-1] == "[" and character == "]") or (stack[-1] == "{" and character == "}")):
                stack.pop()

    if in_string:
        clean_text += '"'

    closing_pairs = {"[": "]", "{": "}"}
    while stack:
        clean_text += closing_pairs[stack.pop()]
    return clean_text


def _execute_tool_if_requested(raw_output: str, notepad: FactNotepad, emotion_token: str = "") -> str:
    """공유 파싱 함수를 활용해 검증 완료된 도구를 안전하게 실행합니다."""

    tool_name, tool_args = _parse_requested_tool(raw_output)

    if not tool_name or tool_name in {"null", "none"}:
        return ""

    print(f"[Debug / Tool Trigger] NAME: {tool_name} | ARGS: {tool_args}")

    if tool_name == "calculate_math":
        expression = _get_tool_arg(tool_args, "expression")
        calc_result = calculate_math(expression)
        print(f"[System] Math: {expression} = {calc_result}")
        return f" | [TOOL RETURN] {calc_result}"

    if tool_name == "write_diary_file":
        if isinstance(tool_args, dict):
            diary_title = _get_tool_arg(tool_args, "title")
            diary_content = _get_tool_arg(tool_args, "content")
        else:
            diary_title, diary_content = _parse_diary_tool_args(str(tool_args))
        if not diary_title or not diary_content:
            print("[System] write_diary_file failed: expected title/content")
            return " | [TOOL ERROR] write_diary_file expects title/content"

        success_msg = _write_diary_file(diary_title, diary_content, emotion_token)
        print(f"[System] {success_msg}")
        return f" | [TOOL RETURN] {success_msg}"

    if tool_name == "write_fact":
        if isinstance(tool_args, dict):
            key = _get_tool_arg(tool_args, "key")
            value = _get_tool_arg(tool_args, "value")
        else:
            key, value = _parse_fact_tool_args(str(tool_args))
        if not key or not value:
            print("[System] write_fact failed: expected key/value")
            return " | [TOOL ERROR] write_fact expects key/value"
        if notepad.add_fact(key, value):
            print(f"[System] Fact written to notepad: {key}")
            return f" | [TOOL RETURN] fact saved: {key}"
        print("[System] write_fact ignored invalid fact")
        return " | [TOOL ERROR] invalid fact"

    print(f"[System] Unknown tool requested: {tool_name}")
    return f" | [TOOL ERROR] Unknown tool: {tool_name}"


def _get_tool_arg(tool_args: Any, key: str) -> str:
    if isinstance(tool_args, dict):
        value = tool_args.get(key, "")
        if value is None:
            return ""
        return str(value).strip()
    return str(tool_args).strip()


def _parse_fact_tool_args(tool_args: str) -> tuple[str, str]:
    if ":" not in tool_args:
        return "", ""
    key, value = tool_args.split(":", 1)
    return key.strip(), value.strip()


def _store_interaction_memory(
    hippocampus: MemoryManager,
    emotion_token: str,
    current_arousal: float,
    valence: float,
    surprise: float,
    memory_kind: str,
    visual_summary: str,
    user_message: str,
    trauma_memory: str,
    flashback_memory: str,
    response_text: str,
    tool_result_text: str,
    inner_monologue: str,
    expected_emotions: dict[str, float],
) -> None:
    event_tag = "[TRAUMA]" if memory_kind == "threat" else "[EPISODE]"
    context_text = user_message if user_message else (trauma_memory or flashback_memory or "Silence")
    memory_content = (
        f"{event_tag} What dad is looking at: {visual_summary} | Context: {context_text} | "
        f"Expect: JOY={expected_emotions.get('JOY', 0.0):.2f}, SAD={expected_emotions.get('SAD', 0.0):.2f}, ANG={expected_emotions.get('ANG', 0.0):.2f} | "
        f"Thought: {inner_monologue} | "
        f"Spoke: {response_text}{tool_result_text}"
    )
    hippocampus.store_memory(
        memory_content,
        emotion_token,
        current_arousal,
        valence_score=valence,
        surprise_score=surprise,
        memory_kind=memory_kind,
    )


def _write_runtime_state(
    eye: VisualObserver,
    emotion_token: str,
    current_arousal: float,
    surprise: float,
    emotion_net: EmotionEngine,
    visual_summary: str,
    user_message: str,
    response_text: str,
    inner_monologue: str,
    expected_emotions: dict[str, float],
) -> None:
    payload = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "eyes_enabled": eye.enabled,
        "emotion_token": emotion_token,
        "arousal": current_arousal,
        "valence": emotion_net.valence,
        "mood": emotion_net.mood,
        "surprise": surprise,
        "previous_expected_emotions": expected_emotions,
        "last_user_message": user_message,
        "last_response": response_text,
        "last_inner_monologue": inner_monologue,
        "last_visual_summary": visual_summary,
    }
    temp_path = RUNTIME_STATE_PATH.with_suffix(".tmp")
    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    temp_path.replace(RUNTIME_STATE_PATH)


def _load_recent_context(file_path: Path = RECENT_CONTEXT_PATH) -> list[dict[str, str]]:
    if not file_path.exists():
        return []

    try:
        with file_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (json.JSONDecodeError, OSError):
        return []

    if not isinstance(data, list):
        return []

    cleaned_turns: list[dict[str, str]] = []
    for item in data[-RECENT_CONTEXT_MAX_TURNS:]:
        if not isinstance(item, dict):
            continue
        cleaned_turns.append(
            {
                "time": str(item.get("time", ""))[:40],
                "dad": str(item.get("dad", ""))[:RECENT_CONTEXT_FIELD_CHARS],
                "baby": str(item.get("baby", ""))[:RECENT_CONTEXT_FIELD_CHARS],
                "thought": str(item.get("thought", ""))[:RECENT_CONTEXT_FIELD_CHARS],
            }
        )
    return cleaned_turns


def _save_recent_context(turns: list[dict[str, str]], file_path: Path = RECENT_CONTEXT_PATH) -> None:
    trimmed_turns = turns[-RECENT_CONTEXT_MAX_TURNS:]
    temp_path = file_path.with_suffix(".tmp")
    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(trimmed_turns, file, ensure_ascii=False, indent=2)
    temp_path.replace(file_path)


def _append_recent_context(
    *,
    dad_message: str,
    baby_response: str,
    inner_monologue: str,
) -> None:
    turns = _load_recent_context()
    turns.append(
        {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dad": dad_message[:RECENT_CONTEXT_FIELD_CHARS],
            "baby": baby_response[:RECENT_CONTEXT_FIELD_CHARS],
            "thought": inner_monologue[:RECENT_CONTEXT_FIELD_CHARS],
        }
    )
    _save_recent_context(turns)


def _format_recent_context(turns: list[dict[str, str]]) -> str:
    if not turns:
        return "No recent conversation."

    lines = []
    for turn in turns[-RECENT_CONTEXT_MAX_TURNS:]:
        lines.append(
            f"- [{turn.get('time', '')}] Dad: {turn.get('dad', '')} | "
            f"Baby: {turn.get('baby', '')} | Thought: {turn.get('thought', '')}"
        )
    return "\n".join(lines)


def _format_terminal_response(
    user_message: str,
    trauma_memory: str,
    flashback_memory: str,
    is_silence_event: bool,
    response_text: str,
) -> str:
    header = ""
    if user_message:
        header = f"\n👨 Dad: {user_message}"
    if trauma_memory:
        header = f"\n[System] Trauma event. {trauma_memory}"
    if flashback_memory:
        header = f"\n[System] Flashback event. {flashback_memory}"
    if is_silence_event:
        header = "\n[System] Silence event."

    return f"{header}\n👶: {response_text}"


def _should_store_memory(
    current_arousal: float,
    surprise: float,
    user_message: str,
    trauma_memory: str,
    flashback_memory: str,
    response_text: str,
) -> bool:
    if not response_text or response_text.strip().upper() == "N/A":
        return False
    has_meaningful_context = bool(user_message or trauma_memory or flashback_memory)
    if not has_meaningful_context:
        return False
    return current_arousal > MEMORY_STORE_AROUSAL_THRESHOLD or surprise > MEMORY_STORE_SURPRISE_THRESHOLD


def _classify_memory_kind(valence: float, arousal: float, surprise: float) -> str:
    if valence < -0.35 and arousal > 0.45:
        return "threat"
    if valence > 0.35 and arousal > 0.35:
        return "reward"
    if surprise > MEMORY_STORE_SURPRISE_THRESHOLD:
        return "surprise"
    return "episode"


def _should_print_thinking_notice(
    user_message: str,
    trauma_memory: str,
    flashback_memory: str,
    is_silence_event: bool,
) -> bool:
    return bool(user_message or trauma_memory or flashback_memory or is_silence_event)


def _extract_emotion_value(emotion_token: str, key: str) -> float:
    match = re.search(rf"{key}:([-+]?[0-9]*\.?[0-9]+)", emotion_token)
    if not match:
        return 0.0
    return float(match.group(1))


def _next_reflect_interval() -> float:
    return random.uniform(REFLECT_INTERVAL_MIN, REFLECT_INTERVAL_MAX)

def _build_idle_reflection_prompt(hippocampus: MemoryManager, idle_duration_sec: float) -> str:
    recent_memories = [
        memory
        for memory in hippocampus.get_recent_memories(limit=5)
        if not _is_dream_memory(memory)
    ]
    if not recent_memories:
        return STATIC_DREAM_PROMPT

    idle_minutes = max(1, int(idle_duration_sec / 60))
    return (
        f"[Dream] Dad has been quiet for about {idle_minutes} minute(s). "
        f"A recent memory surfaced: {recent_memories[0]}"
    )


def _is_dream_memory(memory: str) -> bool:
    return any(marker in memory for marker in DREAM_MEMORY_MARKERS)


def _parse_diary_tool_args(tool_args: str) -> tuple[str, str]:
    if ":" not in tool_args:
        return "", ""
    title, content = tool_args.split(":", 1)
    return title.strip(), content.strip()

def _write_diary_file(title: str, content: str, current_emotion: str) -> str:
    diary_dir = Path("diary")
    diary_dir.mkdir(exist_ok=True)

    date_prefix = time.strftime("%Y-%m-%d")
    file_path = diary_dir / f"{date_prefix}_daily_diary.md"

    file_exists = file_path.exists()
    if file_exists and file_path.stat().st_size > MAX_DAILY_DIARY_BYTES:
        return f"Diary limit reached for {file_path}"
    title = title[:120]
    content = content[:MAX_DIARY_ENTRY_CHARS]

    full_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    mode = "a" if file_exists else "w"

    with file_path.open(mode, encoding="utf-8") as diary_file:
        if not file_exists:
            diary_file.write(f"# 아기의 일기장 (Baby's diary) ({date_prefix})\n")

        diary_file.write(f"### 🕒 생각 기록 시각 (Time): {full_timestamp}\n")
        diary_file.write(f"- **기록 제목 (Subject):** {title}\n")
        diary_file.write(f"- **당시 내면 정서 (Brain State):** `{current_emotion}`\n\n")
        diary_file.write(f"#### 👶 내적 사유 내용 (Thoughts) \n{content}\n\n")
        diary_file.write("---\n\n")

    return f"Diary successfully appended to {file_path}"

if __name__ == "__main__":
    main_loop(interval_sec=MAIN_LOOP_INTERVAL)
