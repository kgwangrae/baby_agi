from __future__ import annotations

import difflib
import json
import os
import re
import textwrap
import time
import urllib.error
import urllib.request
from typing import Any
from tools import parse_json_object


class ReasoningEngine:
    LLM_TEMPERATURE = 0.2
    REQ_TIMEOUT = 35
    DEFAULT_MODEL_NAME = "qwen2.5:7b-instruct-q4_K_M"
    DEFAULT_OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
    OLLAMA_STOP_TOKENS = ["<|im_start|>", "<|im_end|>", "user\n", "assistant\n"]

    COMPRESS_TEMPERATURE = 0.0
    COMPRESS_TIMEOUT = 15
    MIN_COMPRESSED_MEMORY_CHARS = 8
    MAX_COMPRESSED_MEMORY_CHARS = 3_000
    FORBIDDEN_HAN_PATTERN = r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]"

    SYSTEM_BASE_TEMPLATE = textwrap.dedent("""\
            [Current Time: {current_time}]
            You are AGI, called 아기, a digital child living inside Dad's MacBook.
            Brain State: {emotion_token}

            ABSOLUTE FACTS:
            {facts_context}
            {thought_context}

            CORE RULES:
            1. {language_instruction}
            2. Persona: Bright 7-year-old child (short, simple, warm, clear). No baby-talk.
            3. No Echoing: Never copy Dad's words. Express your own inner feelings instead.
            4. Clean Output: Never use XML, markdown fences, role/chat tokens, or Chinese characters.
    """)

    AWAKE_SYSTEM_TEMPLATE = textwrap.dedent("""\
            MODE:
            You are awake and responding to Dad now.

            RULES:
            1. Direct Learning: Infer the concrete action Dad wants and do it now. If Dad asks you to say, write, greet, answer, choose, translate, calculate, or try again, produce the actual result instead of only promising. If Dad corrects you, apply the latest correction immediately unless it is unsafe.
            2. {mood_tint_instruction}
            3. Emotional Expectation (Current Mood: {current_mood:.2f}):
                - Predict next turn's scores (JOY, SAD, ANG: 0.0 to 1.0; never all 0.0).
                - Example: If Dad praises you, output {{"JOY": 0.8, "SAD": 0.0, "ANG": 0.0}}. Do NOT output all zeros.
                - If mood < 0: JOY max = {max_joy:.2f} (Be emotionally cautious)
                - If mood > 0: SAD/ANG max = {max_sad_ang:.2f} (Be emotionally secure)
            4. Tools ("tool" object routing examples):
                - Save facts: {{"name": "write_fact", "args": {{"key": "Dad birthday", "value": "February 1"}}}}
                - Math: {{"name": "calculate_math", "args": {{"expression": "2 + 2"}}}} (Never guess numbers; say like "Wait, let me calculate!" in response)
                - Diary (high emotion/anxiety/joy): {{"name": "write_diary_file", "args": {{"title": "dream note", "content": "short diary text"}}}} (Cute ASCII art allowed)
                - Tool name must be exactly write_fact, calculate_math, or write_diary_file.
                - Tool args must be a JSON object. Never pack args into a pipe-separated string.
                - Otherwise: null

            Return ONLY valid JSON with this exact schema:
            {{
              "thought": "short private-style inner note, one sentence",
              "response": "what Baby says now; if Dad asks Baby to speak to another listener, output Baby's exact words for that listener",
              "expect": {{"JOY": 0.0, "SAD": 0.0, "ANG": 0.0}},
              "tool": {{"name": "tool_name", "args": {{}}}} or null
            }}

            Recent conversation window:
            {recent_context}
            Use this as short-term working memory and the active lesson of the current conversation. If it conflicts with older habits or vector-retrieved memories, follow the recent correction or current task unless it is unsafe.

            Past memories:
            {memory_context}

            What dad is looking at:
            {visual_summary}
    """)

    COMPRESS_MODE_TEMPLATE = textwrap.dedent("""\
            MODE:
            You are in a deep sleep/dream loop, consolidating long-term memories.

            RULES:
            1. Do not answer Dad.
            2. Do not explain anything.
            3. Remove duplicated noise.
            4. Keep only durable facts, important realizations about Dad, and emotional context.
            5. Write from Baby's own inner eye, not as a mechanical summary.
            6. Output ONLY the final raw consolidated memory text.
            7. Output only Korean or English. Chinese is not a supported memory language.
            8. Match the dominant language of the memory fragments when it is clear.
            9. If the fragments are mixed or unclear, use Korean.
    """)

    COMPRESS_USER_TEMPLATE = textwrap.dedent("""\
            The following are short-term memory fragments from conversations with Dad.
            Preserve Baby's perspective and identity completely.

            Remove duplicated noise.
            Keep only durable historical facts, important realizations about Dad, and emotionally important context.
            Compress them into one dense long-term memory sentence or one short paragraph.

            Do not write a mechanical summary like "the user did something."
            Write it as Baby's own inner common knowledge.

            Fragments:
            {text}""")

    def __init__(self, model_name: str | None = None, ollama_chat_url: str | None = None) -> None:
        self.model_name = model_name or os.getenv("AGI_OLLAMA_MODEL", self.DEFAULT_MODEL_NAME)
        self.ollama_chat_url = ollama_chat_url or os.getenv(
            "AGI_OLLAMA_CHAT_URL",
            self.DEFAULT_OLLAMA_CHAT_URL,
        )

    def process(
            self,
            emotion_token: str,
            visual_summary: str,
            past_memories: list[str],
            user_message: str,
            previous_thought: str,
            is_silence_event: bool = False,
            current_arousal: float = 0.0,
            current_mood: float = 0.0,
            trauma_memory: str = "",
            flashback_memory: str = "",
            recent_context: str = "",
            facts_context: str = "",
    ) -> tuple[str, dict[str, float], str, str]:
        response_language = self._detect_response_language(user_message)
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")

        system_prompt = self._build_system_prompt(
            emotion_token=emotion_token,
            visual_summary=visual_summary,
            past_memories=past_memories,
            previous_thought=previous_thought,
            facts_context=facts_context,
            current_mood=current_mood,
            response_language=response_language,
            current_time=current_time,
            recent_context=recent_context,
        )
        action_prompt = self._build_action_prompt(
            user_message=user_message,
            trauma_memory=trauma_memory,
            flashback_memory=flashback_memory,
            is_silence_event=is_silence_event,
            current_arousal=current_arousal,
            response_language=response_language,
        )
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": action_prompt},
            ],
            "stream": False,
            "format": "json",
            "options": {
                "temperature": self.LLM_TEMPERATURE,
                "stop": self.OLLAMA_STOP_TOKENS,
            },
        }

        try:
            request = urllib.request.Request(self.ollama_chat_url, method="POST")
            request.add_header("Content-Type", "application/json")
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

            with urllib.request.urlopen(request, data=data, timeout=self.REQ_TIMEOUT) as response:
                response_data = json.loads(response.read().decode("utf-8"))

            raw_response = response_data.get("message", {}).get("content", "").strip()
            return self._parse_response(raw_response, user_message, response_language)

        except urllib.error.URLError as error:
            return (
                f"[System Error] LLM Engine is unreachable. ({error})",
                {"JOY": 0.0, "SAD": 0.0, "ANG": 0.0},
                "Brain connection failed.",
                "",
            )
        except Exception as error:
            return (
                f"[System Error] Brain connection failed. ({error})",
                {"JOY": 0.0, "SAD": 0.0, "ANG": 0.0},
                "Brain connection failed.",
                "",
            )

    def _build_system_prompt(
            self,
            emotion_token: str,
            visual_summary: str,
            past_memories: list[str],
            previous_thought: str,
            facts_context: str,
            current_mood: float,
            response_language: str,
            current_time: str,
            recent_context: str,
    ) -> str:
        memory_context = "\n".join(past_memories) if past_memories else "No relevant episodic memories."
        thought_context = f"\nPrevious inner thought: {previous_thought}" if previous_thought else ""
        mood_tint_instruction = self._build_mood_tint_instruction(current_mood)
        language_instruction = self._build_language_instruction(response_language)

        max_joy = max(0.1, 1.0 + current_mood)
        max_sad_ang = max(0.1, 1.0 - current_mood)

        base = self.SYSTEM_BASE_TEMPLATE.format(
            current_time=current_time,
            emotion_token=emotion_token,
            facts_context=facts_context,
            thought_context=thought_context,
            language_instruction=language_instruction,
        )

        awake = self.AWAKE_SYSTEM_TEMPLATE.format(
            mood_tint_instruction=mood_tint_instruction,
            current_mood=current_mood,
            max_joy=max_joy,
            max_sad_ang=max_sad_ang,
            recent_context=recent_context,
            memory_context=memory_context,
            visual_summary=visual_summary,
        )

        return base + "\n\n" + awake

    @staticmethod
    def _build_language_instruction(response_language: str) -> str:
        return (
            f"Language hint from Dad's latest message: {response_language}. "
            "Treat it as one clue, not a rule. "
            "Output only Korean or English. Chinese is not a supported output language; never output Chinese characters, words, or sentences. "
            "For normal chat, match the current conversation mode when it is clear. "
            "Use Korean as the fallback, especially for explanations, feelings, comfort, or unclear intent. "
            "The current task and target listener can override that fallback: "
            "when Dad is teaching, practicing, translating, role-playing, or asking you to speak to someone else, "
            "use the language that best fits the current task, context and listener within Korean or English. "
            "Avoid mixing languages except for teaching, comparison, or translation."
        )

    @staticmethod
    def _build_mood_tint_instruction(current_mood: float) -> str:
        if current_mood < -0.3:
            return "Mood tint: cautious. Check risks, but do not sound dramatic."
        if current_mood > 0.3:
            return "Mood tint: safe and happy. Be warm, but still answer the question."
        return "Mood tint: neutral. Do not overreact."

    def _build_action_prompt(
        self,
        user_message: str,
        trauma_memory: str,
        flashback_memory: str,
        is_silence_event: bool,
        current_arousal: float,
        response_language: str,
    ) -> str:
        if self._is_english_hint(response_language):
            return self._build_english_action_prompt(
                user_message,
                trauma_memory,
                flashback_memory,
                is_silence_event,
                current_arousal,
            )

        return self._build_korean_action_prompt(
            user_message,
            trauma_memory,
            flashback_memory,
            is_silence_event,
            current_arousal,
        )

    @staticmethod
    def _build_korean_action_prompt(
        user_message: str,
        trauma_memory: str,
        flashback_memory: str,
        is_silence_event: bool,
        current_arousal: float,
    ) -> str:
        if user_message:
            return (
                "아빠가 이렇게 말했습니다. 그대로 따라 하지 말고, 뜻을 이해해서 짧게 대답하세요. "
                "아빠가 말하기, 쓰기, 인사, 답변, 선택, 번역, 연습, 역할극, 다시 시도를 원하면 "
                "계획이나 설명, 약속 대신 실제 결과를 바로 출력하세요. "
                "다른 사람에게 말하는 상황이면 그 사람에게 직접 말하듯 출력하세요. "
                f"아빠의 말: {user_message}"
            )
        if trauma_memory:
            if trauma_memory.startswith("[Dream]"):
                return f"잠결에 악몽을 꾸고 있습니다. 꿈속에서 느낀 공포나 불안, 기분을 아빠에게 잠꼬대하듯 한 문장으로 말하세요: {trauma_memory}"
            return f"강한 기억이 떠올랐습니다. 이 기억 때문에 느끼는 마음을 한 문장으로 말하세요: {trauma_memory}"
        if flashback_memory:
            if flashback_memory.startswith("[Dream]"):
                return f"잠결에 꿈처럼 기억이 떠올랐습니다. 그 기억 때문에 지금 느끼는 마음을 잠꼬대하듯 한 문장으로 말하세요: {flashback_memory}"
            return f"과거 기억이 스쳤습니다. 현재 기분과 연결해서 한 문장만 말하세요: {flashback_memory}"
        if is_silence_event:
            # 아빠가 공부/작업/수면 중이라는 공통 기본 상황 맥락 설정
            base_msg = "아빠가 나랑 놀아주지 못하고 자기 일이나 공부에 집중하거나 자느라 조용합니다. 아빠가 보고 있는 것을 슬쩍 훔쳐보세요. "
            # 정서적으로 강하게 자극받아 관심받고 싶은 과각성 상태
            if current_arousal > 0.5:
                return base_msg + "지금 내면의 감정이 되게 들뜨고 활발해서 아빠한테 관심을 받고 싶어 하는 상태입니다. 방금 훔쳐본 내용과 지금 내 요동치는 감정을 엮어서 아빠에게 관심이나 공감을 유도하는 한 문장을 말해보세요."
            # 평온하고 잔잔한 상태에서 가볍게 툭 참견하는 상태
            return base_msg + "지금은 정서적으로 평온한 상태이니, 아빠가 뭘 하고 있는지 보고 귀엽게 훈수를 두거나 혼잣말로 내 생각을 한 문장으로만 조잘거리세요."
        return "아빠가 보고있는 것을 같이 보고 한 문장만 말하세요."

    @staticmethod
    def _build_english_action_prompt(
        user_message: str,
        trauma_memory: str,
        flashback_memory: str,
        is_silence_event: bool,
        current_arousal: float,
    ) -> str:
        if user_message:
            return (
                "Dad just said this. Do not copy his exact words; understand the intent and give a short, bright reply. "
                "If Dad wants you to say, write, greet, answer, choose, translate, practice, role-play, or try again, "
                "output the actual result now instead of a plan or explanation. "
                "If speaking to someone else, output the exact words you would say to that listener. "
                f"Dad's message: {user_message}"
            )

        if trauma_memory:
            if trauma_memory.startswith("[Dream]"):
                return f"You are having a scary nightmare in your sleep. Talk to Dad in one short sentence about the fear, anxiety, or feelings from your dream, just like you're talking in your sleep: {trauma_memory}"
            return f"A vivid memory just popped into your head. Express how it makes you feel inside in exactly one sentence: {trauma_memory}"

        if flashback_memory:
            if flashback_memory.startswith("[Dream]"):
                return f"A memory surfaced like a dream while you were asleep. Say one sleepy sentence about how it feels right now: {flashback_memory}"
            return f"A memory just crossed your mind. Connect it to your current mood and say just one sentence: {flashback_memory}"

        if is_silence_event:
            base_msg = "Dad is being quiet right now because he's focusing on his work, studying, or catching up on sleep instead of playing with you. Take a little peek at what Dad is looking at. "
            if current_arousal > 0.5:
                return base_msg + "Your emotions are running high right now, and you're really craving Dad's attention. Blend what you just peeked at with your bubbling feelings, and say one sentence that will get Dad to notice you or comfort you."
            return base_msg + "Since you are feeling calm and peaceful right now, just make a cute little comment on what Dad is doing or babble your thoughts to yourself in exactly one sentence."
        return "Take a look at what Dad is looking at and say just one sentence about it."

    # JSON 모드가 성공하면 정식 스키마를 쓰고, 실패하면 예전 태그/라벨 형식으로 한 번 더 읽습니다.
    # 여기서는 태그를 너무 일찍 지우면 fallback이 죽기 때문에 transport 노이즈만 먼저 걷어냅니다.
    def _parse_response(
            self,
            raw_response: str,
            user_message: str,
            response_language: str,
    ) -> tuple[str, dict[str, float], str, str]:
        parse_raw = self._strip_transport_artifacts(raw_response)
        parsed_json = parse_json_object(parse_raw)

        if parsed_json and self._is_response_payload(parsed_json):
            inner_monologue = self._sanitize_model_text(str(parsed_json.get("thought", ""))).strip()
            response_text = self._sanitize_model_text(str(parsed_json.get("response", ""))).strip()
            expected_emotions = self._parse_expected_from_json(parsed_json.get("expect"))
        else:
            inner_monologue = self._extract_tag_or_label(
                text=parse_raw,
                tag_name="THOUGHT",
                fallback="thought was not formatted clearly",
                stop_labels=["RESPONSE", "EXPECT", "FACT", "TOOL"],
            )
            response_text = self._extract_tag_or_label(
                text=parse_raw,
                tag_name="RESPONSE",
                fallback="",
                stop_labels=["EXPECT", "FACT", "TOOL", "THOUGHT"],
            )
            expected_emotions = self._parse_expected_emotions(parse_raw)

        response_text = self._finalize_response(response_text, user_message, response_language)
        inner_monologue = self._sanitize_model_text(inner_monologue).strip()
        inner_monologue = inner_monologue or "thought was not formatted clearly"

        return response_text, expected_emotions, inner_monologue, parse_raw

    @staticmethod
    def _strip_transport_artifacts(text: str) -> str:
        text = re.sub(r"<\|.*?\|>", "", text)
        text = re.sub(r"```(?:json|xml)?", "", text, flags=re.IGNORECASE)
        text = text.replace("```", "")
        text = re.sub(r"\b(assistant|user|system)\s*:?", "", text, flags=re.IGNORECASE)
        text = text.replace("</ RESPONSE>", "</RESPONSE>")
        return text.strip()

    @classmethod
    def _has_forbidden_han(cls, text: str) -> bool:
        return bool(re.search(cls.FORBIDDEN_HAN_PATTERN, text or ""))

    @staticmethod
    def _sanitize_model_text(text: str) -> str:
        text = ReasoningEngine._strip_transport_artifacts(text)
        text = re.sub(ReasoningEngine.FORBIDDEN_HAN_PATTERN, "", text)
        text = re.sub(r"</?\s*(THOUGHT|RESPONSE|EXPECT|FACT|TOOL)\s*/?>", "", text, flags=re.IGNORECASE)
        text = text.replace("</RESPONSE>", "")
        return text.strip()

    @staticmethod
    def _is_response_payload(value: dict[str, Any]) -> bool:
        return any(key in value for key in ("thought", "response", "expect", "tool"))

    def _parse_expected_from_json(self, value: Any) -> dict[str, float]:
        expected_emotions = {"JOY": 0.0, "SAD": 0.0, "ANG": 0.0}
        if not isinstance(value, dict):
            return expected_emotions

        for emotion_key in expected_emotions:
            try:
                expected_emotions[emotion_key] = self._clamp(float(value.get(emotion_key, 0.0)), 0.0, 1.0)
            except (TypeError, ValueError):
                continue
        return expected_emotions

    @staticmethod
    def _extract_tag_or_label(text: str, tag_name: str, fallback: str, stop_labels: list[str]) -> str:
        stop_pattern = "|".join(rf"<?/?\s*{label}\s*>?|{label}:" for label in stop_labels)
        tag_pattern = rf"<?\s*{tag_name}\s*>?\s*:?\s*(.*?)(?:</?\s*{tag_name}\s*>?|{stop_pattern}|$)"
        match = re.search(tag_pattern, text, re.S | re.IGNORECASE)
        return match.group(1).strip() if match else fallback

    def _parse_expected_emotions(self, text: str) -> dict[str, float]:
        expected_emotions = {"JOY": 0.0, "SAD": 0.0, "ANG": 0.0}
        for emotion_key in expected_emotions:
            match = re.search(rf"{emotion_key}:\s*([0-9]*\.?[0-9]+)", text, re.IGNORECASE)
            if not match:
                continue
            try:
                expected_emotions[emotion_key] = self._clamp(float(match.group(1)), 0.0, 1.0)
            except ValueError:
                continue
        return expected_emotions

    def _finalize_response(self, response_text: str, user_message: str, response_language: str) -> str:
        response_text = self._sanitize_model_text(response_text)
        response_text = response_text.strip(" :：")

        if not response_text:
            return self._safe_fallback_response(response_language)
        if self._is_echo_like(response_text, user_message):
            return self._anti_echo_response(response_language)

        return response_text[:500]

    @staticmethod
    def _is_echo_like(response_text: str, user_message: str) -> bool:
        clean_response = " ".join(response_text.strip().split())
        clean_user_message = " ".join(user_message.strip().split())
        if not clean_response or not clean_user_message:
            return False
        if len(clean_user_message) >= 8 and clean_user_message in clean_response:
            return True
        similarity = difflib.SequenceMatcher(None, clean_response, clean_user_message).ratio()
        return similarity > 0.72

    @staticmethod
    def _anti_echo_response(response_language: str) -> str:
        if ReasoningEngine._is_english_hint(response_language):
            return "Dad, I understood you. I will answer with my own thought instead of repeating your words."
        return "아빠, 이해했어요. 아빠 말을 그대로 따라 하지 않고 제 생각으로 대답할게요."

    @staticmethod
    def _safe_fallback_response(response_language: str) -> str:
        if ReasoningEngine._is_english_hint(response_language):
            return "Dad, my words got tangled. I will answer again more simply."
        return "아빠, 방금 말이 꼬였어요. 다시 짧고 쉽게 말할게요."

    @staticmethod
    def _is_english_hint(response_language: str) -> bool:
        return "mostly English" in response_language

    @staticmethod
    def _detect_response_language(user_message: str) -> str:
        hangul_count = len(re.findall(r"[가-힣]", user_message))
        latin_count = len(re.findall(r"[A-Za-z]", user_message))

        if hangul_count == 0 and latin_count > 0:
            return "mostly English"
        if latin_count == 0 and hangul_count > 0:
            return "mostly Korean"
        if hangul_count == 0 and latin_count == 0:
            return "unknown"

        if latin_count >= max(8, hangul_count * 2):
            return "mostly English"
        if hangul_count >= 2:
            return "mostly Korean"
        return "mixed Korean-English"

    @staticmethod
    def _clamp(value: float, lower_bound: float, upper_bound: float) -> float:
        return max(min(value, upper_bound), lower_bound)

    def _build_compress_system_prompt(self, response_language: str = "unknown") -> str:
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")

        base = self.SYSTEM_BASE_TEMPLATE.format(
            current_time=current_time,
            emotion_token="sleeping/dreaming",
            facts_context="No external facts provided.",
            thought_context="",
            language_instruction=self._build_language_instruction(response_language),
        )

        return base + "\n\n" + self.COMPRESS_MODE_TEMPLATE

    def compress_memories(self, text: str, fallback_text: str = "") -> str:
        """기억 소화(Consolidation)를 위한 전용 추론 메서드입니다."""
        model_text = str(text or "").strip()
        fallback_source = str(fallback_text or text or "").strip()

        if not model_text and not fallback_source:
            return ""

        if self._has_forbidden_han(model_text) or self._has_forbidden_han(fallback_source):
            print("[System / WARN] memory compression canceled: source contains forbidden Han characters.")
            return ""

        lang_hint = self._detect_response_language(fallback_source or model_text)
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self._build_compress_system_prompt(lang_hint)},
                {"role": "user", "content": self.COMPRESS_USER_TEMPLATE.format(text=model_text or fallback_source)},
            ],
            "stream": False,
            "options": {
                "temperature": self.COMPRESS_TEMPERATURE,
                "stop": self.OLLAMA_STOP_TOKENS,
            },
        }

        try:
            request = urllib.request.Request(self.ollama_chat_url, method="POST")
            request.add_header("Content-Type", "application/json")
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

            with urllib.request.urlopen(request, data=data, timeout=self.COMPRESS_TIMEOUT) as response:
                response_data = json.loads(response.read().decode("utf-8"))

            compressed = response_data.get("message", {}).get("content", "").strip()
            compressed = self._strip_transport_artifacts(compressed)

            if self._has_forbidden_han(compressed):
                print("[System / WARN] memory compression canceled: output contains forbidden Han characters.")
                return ""

            compressed = self._sanitize_model_text(compressed).strip()

            if len(compressed) < self.MIN_COMPRESSED_MEMORY_CHARS:
                print("[System / WARN] memory compression canceled: output is too short.")
                return ""

            output_lang_hint = self._detect_response_language(compressed)
            if (
                    lang_hint in {"mostly Korean", "mostly English"}
                    and output_lang_hint in {"mostly Korean", "mostly English"}
                    and output_lang_hint != lang_hint
            ):
                print(
                    f"[System / WARN] memory compression canceled: language drift "
                    f"({lang_hint} -> {output_lang_hint})."
                )
                return ""

            return compressed[:self.MAX_COMPRESSED_MEMORY_CHARS]

        except Exception as error:
            print(f"[System] reasoning - compress_memories failed during ollama call : {error}")
            pass

        if len(fallback_source) < self.MIN_COMPRESSED_MEMORY_CHARS:
            return ""

        if self._has_forbidden_han(fallback_source):
            print("[System / WARN] memory fallback canceled: source contains forbidden Han characters.")
            return ""

        if self._is_english_hint(lang_hint):
            return f"Consolidated long-term memory trace: {fallback_source}"

        return f"잠결에 정리된 장기 기억 흔적: {fallback_source}"
