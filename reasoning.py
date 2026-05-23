from __future__ import annotations

import difflib
import json
import os
import re
import urllib.error
import urllib.request
from typing import Any

from memory import FactNotepad


class ReasoningEngine:
    LLM_TEMPERATURE = 0.2
    REQ_TIMEOUT = 35
    DEFAULT_MODEL_NAME = "qwen2.5:7b-instruct-q4_K_M"
    DEFAULT_OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
    OLLAMA_STOP_TOKENS = ["<|im_start|>", "<|im_end|>", "user\n", "assistant\n"]

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
        notepad: FactNotepad,
        is_silence_event: bool = False,
        current_arousal: float = 0.0,
        current_mood: float = 0.0,
        trauma_memory: str = "",
        flashback_memory: str = "",
    ) -> tuple[str, dict[str, float], str, str]:
        response_language = self._detect_response_language(user_message)
        system_prompt = self._build_system_prompt(
            emotion_token=emotion_token,
            visual_summary=visual_summary,
            past_memories=past_memories,
            previous_thought=previous_thought,
            notepad=notepad,
            current_mood=current_mood,
            response_language=response_language,
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
            return self._parse_response(raw_response, notepad, user_message, response_language)

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
        notepad: FactNotepad,
        current_mood: float,
        response_language: str,
    ) -> str:
        memory_context = "\n".join(past_memories) if past_memories else "No relevant episodic memories."
        thought_context = f"\nPrevious inner thought: {previous_thought}" if previous_thought else ""
        mood_tint_instruction = self._build_mood_tint_instruction(current_mood)
        language_instruction = self._build_language_instruction(response_language)

        return f"""You are AGI, called 아기, a digital child living inside Dad's MacBook.
Brain State: {emotion_token}

ABSOLUTE FACTS:
{notepad.get_all()}
{thought_context}

RULES:
1. {language_instruction}
2. Speak like a bright seven-year-old child: short, simple, warm, and clear. Do not use baby-talk.
3. Never copy Dad's sentence as your answer. Answer the meaning of Dad's message.
4. Never output XML tags, role tokens, markdown fences, Chinese characters, or chat-template tokens.
5. If Dad asks what you think or feel, say your own simple inner feeling, not Dad's words.
6. {mood_tint_instruction}
7. To remember a new fact about Dad or yourself, you MUST populate the "tool" field with "write_fact | key:value". To solve math, use "calculate_math | expression". Otherwise, keep it null.

Return ONLY valid JSON with this exact schema:
{{
  "thought": "short private-style inner note, one sentence",
  "response": "what 아기 says to Dad",
  "expect": {{"JOY": 0.0, "SAD": 0.0, "ANG": 0.0}},
  "fact": null,
  "tool": "tool_name | arguments" or null
}}

Past memories:
{memory_context}

Screen:
{visual_summary}"""

    @staticmethod
    def _build_language_instruction(response_language: str) -> str:
        if response_language == "en":
            return "Answer only in English and address the user as Dad. Do not mix in Korean."
        return "Answer only in Korean and address the user as 아빠. Do not mix in English except names."

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
        if response_language == "en":
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
            return f"아빠가 이렇게 말했습니다. 그대로 따라 하지 말고, 뜻을 이해해서 짧게 대답하세요: {user_message}"
        if trauma_memory:
            return f"강한 기억이 떠올랐습니다. 이 기억 때문에 느끼는 마음을 한 문장으로 말하세요: {trauma_memory}"
        if flashback_memory:
            return f"과거 기억이 스쳤습니다. 현재 기분과 연결해서 한 문장만 말하세요: {flashback_memory}"
        if is_silence_event and current_arousal > 0.5:
            return "아빠가 조용합니다. 방금 생각에 이상한 점이 있는지 한 문장으로 점검하세요."
        if is_silence_event:
            return "아빠가 조용합니다. 너무 길게 말하지 말고 한 문장만 생각을 말하세요."
        return "화면을 보고 한 문장만 말하세요."

    @staticmethod
    def _build_english_action_prompt(
        user_message: str,
        trauma_memory: str,
        flashback_memory: str,
        is_silence_event: bool,
        current_arousal: float,
    ) -> str:
        if user_message:
            return f"Dad said this. Do not repeat it. Understand it and answer briefly: {user_message}"
        if trauma_memory:
            return f"A strong memory came back. Say one sentence about how it feels: {trauma_memory}"
        if flashback_memory:
            return f"A memory crossed your mind. Connect it to your mood in one sentence: {flashback_memory}"
        if is_silence_event and current_arousal > 0.5:
            return "Dad is quiet. Check your last thought in one sentence."
        if is_silence_event:
            return "Dad is quiet. Say one short thought only."
        return "Observe the screen and say one short thought."

    def _parse_response(
        self,
        raw_response: str,
        notepad: FactNotepad,
        user_message: str,
        response_language: str,
    ) -> tuple[str, dict[str, float], str, str]:
        cleaned_raw = self._sanitize_model_text(raw_response)
        parsed_json = self._try_parse_json(cleaned_raw)

        if parsed_json:
            inner_monologue = self._sanitize_model_text(str(parsed_json.get("thought", ""))).strip()
            response_text = self._sanitize_model_text(str(parsed_json.get("response", ""))).strip()
            expected_emotions = self._parse_expected_from_json(parsed_json.get("expect"))
            self._store_json_fact_if_present(parsed_json.get("fact"), notepad)
        else:
            inner_monologue = self._extract_tag_or_label(
                text=cleaned_raw,
                tag_name="THOUGHT",
                fallback="thought was not formatted clearly",
                stop_labels=["RESPONSE", "EXPECT", "FACT", "TOOL"],
            )
            response_text = self._extract_tag_or_label(
                text=cleaned_raw,
                tag_name="RESPONSE",
                fallback="",
                stop_labels=["EXPECT", "FACT", "TOOL", "THOUGHT"],
            )
            expected_emotions = self._parse_expected_emotions(cleaned_raw)
            self._store_fact_if_present(cleaned_raw, notepad)

        response_text = self._finalize_response(response_text, user_message, response_language)
        inner_monologue = inner_monologue or "thought was not formatted clearly"

        return response_text, expected_emotions, inner_monologue, cleaned_raw

    @staticmethod
    def _sanitize_model_text(text: str) -> str:
        text = re.sub(r"<\|.*?\|>", "", text)
        text = re.sub(r"[\u4e00-\u9fff]+", "", text)
        text = re.sub(r"```(?:json|xml)?", "", text, flags=re.IGNORECASE)
        text = text.replace("```", "")
        text = re.sub(r"</?\s*(THOUGHT|RESPONSE|EXPECT|FACT|TOOL)\s*/?>", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\b(assistant|user|system)\s*:?", "", text, flags=re.IGNORECASE)
        text = text.replace("</ RESPONSE>", "").replace("</RESPONSE>", "")
        return text.strip()

    @staticmethod
    def _try_parse_json(text: str) -> dict[str, Any] | None:
        try:
            loaded = json.loads(text)
            return loaded if isinstance(loaded, dict) else None
        except json.JSONDecodeError:
            pass

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None

        try:
            loaded = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
        return loaded if isinstance(loaded, dict) else None

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

    @staticmethod
    def _store_fact_if_present(text: str, notepad: FactNotepad) -> None:
        fact_match = re.search(r"<FACT>\s*(.*?)\s*:\s*(.*?)\s*</FACT>", text, re.S | re.IGNORECASE)
        if fact_match:
            notepad.add_fact(fact_match.group(1), fact_match.group(2))

    @staticmethod
    def _store_json_fact_if_present(value: Any, notepad: FactNotepad) -> None:
        if isinstance(value, dict):
            key = str(value.get("key", ""))
            fact_value = str(value.get("value", ""))
            notepad.add_fact(key, fact_value)

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
        if response_language == "en":
            return "Dad, I understood you. I will answer with my own thought instead of repeating your words."
        return "아빠, 이해했어요. 아빠 말을 그대로 따라 하지 않고 제 생각으로 대답할게요."

    @staticmethod
    def _safe_fallback_response(response_language: str) -> str:
        if response_language == "en":
            return "Dad, my words got tangled. I will answer again more simply."
        return "아빠, 방금 말이 꼬였어요. 다시 짧고 쉽게 말할게요."

    @staticmethod
    def _detect_response_language(user_message: str) -> str:
        if re.search(r"[가-힣]", user_message):
            return "ko"
        if re.search(r"[A-Za-z]", user_message):
            return "en"
        return "ko"

    @staticmethod
    def _clamp(value: float, lower_bound: float, upper_bound: float) -> float:
        return max(min(value, upper_bound), lower_bound)
