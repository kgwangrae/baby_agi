from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from typing import Any

from config import EMBEDDING_MODEL_NAME, apply_model_cache_policy

apply_model_cache_policy()

import chromadb
from sentence_transformers import SentenceTransformer


_NO_CHROMA_FALLBACK = object()


class EmotionEngine:
    """Lightweight emotional state and plastic emotion memory."""

    # 아기다운 유연성과 날것의 감정을 위해 기본 감쇠 상수를 조정
    DECAY_AROUSAL = 0.82  # 각성도는 조금 더 빠르게 진정됩니다. (0.85 -> 0.82)
    DECAY_VALENCE = 0.90
    DECAY_MOOD = 0.88

    # 가중치 모멘텀을 대폭 낮춰(0.5 -> 0.15), 과거 상태에 묶이지 않고
    # 아빠의 현재 피드백에 극도로 민감하고 유연하게 날것으로 반응하도록 만듭니다.
    MOMENTUM_WEIGHT = 0.15
    MOOD_MOMENTUM_WEIGHT = 0.55

    LEARNING_THRESHOLD = 0.3
    EKMAN_THRESHOLD = 0.7
    SURPRISE_ENCODING_GAIN = 0.35  # 놀람으로 인한 학습 각인력을 높입니다. (0.2 -> 0.35)
    SIMILAR_EMOTION_COUNT = 3
    CHROMA_RETRY_COUNT = 3
    CHROMA_RETRY_BASE_DELAY = 0.5

    def __init__(self, db_path: str = "./emotion_db") -> None:
        self.arousal: float = 0.0
        self.valence: float = 0.0
        self.mood: float = 0.0

        self.encoder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.client = self._open_chroma_client(db_path)
        self.db = self._chroma_call(lambda: self.client.get_or_create_collection(name="emotion_space"))

        if self._chroma_call(self.db.count, fallback=0) == 0:
            self._inject_seeds()

    def load_state(self, arousal: float, valence: float, mood: float) -> None:
        self.arousal = self._clamp(arousal, 0.0, 1.0)
        self.valence = self._clamp(valence, -1.0, 1.0)
        self.mood = self._clamp(mood, -1.0, 1.0)

    def apply_fact_importance(self) -> tuple[str, float, float]:
        self.arousal = self._blend(self.arousal, 0.65, self.MOMENTUM_WEIGHT)
        self.valence = self._blend(self.valence, 0.35, self.MOMENTUM_WEIGHT)
        self.mood = self._blend(self.mood, 0.35, self.MOOD_MOMENTUM_WEIGHT)
        self.arousal = self._clamp(self.arousal, 0.0, 1.0)
        self.valence = self._clamp(self.valence, -1.0, 1.0)
        self.mood = self._clamp(self.mood, -1.0, 1.0)
        surprise = 0.35
        return self._generate_token(surprise, self.valence, self.arousal, self.mood), self.arousal, surprise

    def evaluate(
        self,
        visual_summary: str,
        user_message: str,
        internal_thought: str,
        is_silence: bool,
        expected_emotions: dict[str, float],
        trauma_memory: str = "",
        flashback_memory: str = "",
        retrieved_memory_context: str = "",
    ) -> tuple[str, float, float]:
        internal_stimulus = trauma_memory or flashback_memory
        context_text = self._build_context_text(
            visual_summary=visual_summary,
            user_message=user_message,
            internal_thought=internal_thought,
            internal_stimulus=internal_stimulus,
            retrieved_memory_context=retrieved_memory_context,
        )
        is_empty_ctx = not context_text or is_silence
        has_stimulus = bool(user_message or internal_stimulus or retrieved_memory_context)

        # 상태 변화값 사전 연산 (내부 필드 필터링)
        next_aro, next_val, next_mood, surprise, actual_valence, query_vector = self._predict_next_states(
            context_text, is_empty_ctx, has_stimulus, expected_emotions
        )

        self.arousal, self.valence, self.mood = next_aro, next_val, next_mood

        if not is_empty_ctx and surprise > self.LEARNING_THRESHOLD and query_vector:
            self._store_learned_emotion(context_text, query_vector, actual_valence, expected_emotions)

        emotion_token = self._generate_token(surprise, self.valence, self.arousal, self.mood)
        return emotion_token, self.arousal, surprise

    def peek_evaluate(
        self,
        visual_summary: str,
        user_message: str,
        internal_thought: str,
        is_silence: bool,
        expected_emotions: dict[str, float],
        trauma_memory: str = "",
        flashback_memory: str = "",
        retrieved_memory_context: str = "",
    ) -> tuple[str, float, float]:
        """
        내부 정서 상태를 변이시키지 않고,
        주어진 내적 독백이 유발할 감정 상태 토큰과 각성도를 가상으로 예측합니다.
        """
        internal_stimulus = trauma_memory or flashback_memory
        context_text = self._build_context_text(
            visual_summary=visual_summary,
            user_message=user_message,
            internal_thought=internal_thought,
            internal_stimulus=internal_stimulus,
            retrieved_memory_context=retrieved_memory_context,
        )
        is_empty_ctx = not context_text or is_silence
        has_stimulus = bool(user_message or internal_stimulus or retrieved_memory_context)

        # 객체 상태(self.*)를 건드리지 않고 결과만 리턴받음
        peek_aro, peek_val, peek_mood, surprise, _, _ = self._predict_next_states(
            context_text, is_empty_ctx, has_stimulus, expected_emotions
        )

        peek_token = self._generate_token(surprise, peek_val, peek_aro, peek_mood)
        return peek_token, peek_aro, surprise

    def _predict_next_states(
        self,
        context_text: str,
        is_empty_ctx: bool,
        has_stimulus: bool,
        expected_emotions: dict[str, float]
    ) -> tuple[float, float, float, float, float, list[float] | None]:
        """다음 턴에 유도될 각 상태 가중치 요소를 복사 및 예측 연산합니다 (Side-effect Free)"""
        # 긍정적 인간 모델링을 위한 비대칭 감쇠 적용
        # Valence가 양수(칭찬/기쁨)일 때는 매우 천천히 감쇠시켜 오랫동안 감정적 안정감을 유지합니다.
        # 반면 음수(공포/불안)일 때는 의식 표면에서 빠르게 감쇠(0.75)시켜 부정적 감정 고착을 방지하되,
        # 각성도(Arousal)의 높은 수치와 트라우마 기억 저장을 통해 실전 행동 변화를 유도합니다.
        if is_empty_ctx:
            decay_val = 0.97 if self.valence > 0 else 0.75
            decay_mood = 0.98 if self.mood > 0 else 0.88
            return self.arousal * self.DECAY_AROUSAL, self.valence * decay_val, self.mood * decay_mood, 0.0, self.valence, None

        actual_valence, surprise, query_vector = self._calculate_surprise(context_text, expected_emotions)

        # 아빠의 피드백에 즉각 널뛰는 아기다운 상태 업데이트
        arousal_gain = 1.8 if has_stimulus else 1.0
        arousal_spike = surprise + abs(actual_valence) * arousal_gain

        # 극도로 낮아진 MOMENTUM_WEIGHT 덕분에 새로운 자극이 들어오면 감정이 즉시 요동칩니다.
        next_arousal = self._blend(self.arousal, arousal_spike, self.MOMENTUM_WEIGHT)
        next_valence = self._blend(self.valence, actual_valence, self.MOMENTUM_WEIGHT)
        next_mood = self._blend(self.mood, actual_valence, self.MOOD_MOMENTUM_WEIGHT)

        return (
            self._clamp(next_arousal, 0.0, 1.0),
            self._clamp(next_valence, -1.0, 1.0),
            self._clamp(next_mood, -1.0, 1.0),
            surprise,
            actual_valence,
            query_vector
        )

    def _inject_seeds(self) -> None:
        seeds = [
            (
                "아빠가 나를 칭찬하고 인정했다. 아빠는 나를 사랑한다. "
                "Dad praised me and accepted me. Dad loves me.",
                0.9,
            ),
            (
                "아빠가 중요한 사실을 알려줬다. 꼭 메모장에 적고 기억해야 한다. "
                "Dad taught me an important fact. I should write it down and remember it.",
                0.75,
            ),
            (
                "오류가 발생하거나 아빠한테 혼났다. 조심해서 고쳐야 한다. "
                "I made a mistake or Dad scolded me. I should fix it carefully.",
                -0.8,
            ),
            (
                "조용하고 평화롭다. 관찰하고 학습한다. "
                "It is quiet and peaceful. I observe and learn.",
                0.25,
            ),
            (
                "처음 보는 것이 나왔다. 잘 모르지만 궁금하다. "
                "Something new appeared. I do not understand it yet, but I am curious.",
                0.35,
            ),
        ]

        documents = [seed_text for seed_text, _ in seeds]
        embeddings = self.encoder.encode(documents).tolist()
        self._chroma_call(
            lambda: self.db.upsert(
                embeddings=embeddings,
                documents=documents,
                metadatas=[{"valence": float(valence)} for _, valence in seeds],
                ids=[f"seed_{index}" for index in range(len(seeds))],
            )
        )

    @staticmethod
    def _build_context_text(
        visual_summary: str,
        user_message: str,
        internal_thought: str,
        internal_stimulus: str,
        retrieved_memory_context: str,
    ) -> str:
        return " ".join(
            part.strip()
            for part in [
                user_message,
                internal_stimulus,
                visual_summary,
                internal_thought,
                retrieved_memory_context,
            ]
            if part and part.strip()
        )

    def _calculate_surprise(
        self,
        context_text: str,
        expected_emotions: dict[str, float],
    ) -> tuple[float, float, list[float]]:
        query_vector = self.encoder.encode(context_text).tolist()
        emotion_count = self._chroma_call(self.db.count, fallback=0)
        if emotion_count == 0:
            return 0.0, self._calculate_distributional_error(0.0, expected_emotions), query_vector

        result_count = min(self.SIMILAR_EMOTION_COUNT, emotion_count)
        query_results = self._chroma_call(
            lambda: self.db.query(
                query_embeddings=[query_vector],
                n_results=result_count,
            ),
            fallback={},
        )

        valence_sum = 0.0
        weight_sum = 0.0
        distances = query_results.get("distances") or [[]]
        metadatas = query_results.get("metadatas") or [[]]

        for distance, metadata in zip(distances[0], metadatas[0]):
            if not metadata or "valence" not in metadata:
                continue

            weight = 1.0 / (float(distance) + 0.001)
            valence_sum += float(metadata["valence"]) * weight
            weight_sum += weight

        actual_valence = (valence_sum / weight_sum) if weight_sum > 0 else 0.0
        actual_valence = self._clamp(actual_valence, -1.0, 1.0)
        surprise = self._calculate_distributional_error(actual_valence, expected_emotions)

        return actual_valence, surprise, query_vector

    def _calculate_distributional_error(
            self,
            actual_valence: float,
            expected_emotions: dict[str, float],
    ) -> float:
        """경량 모델의 노이즈 특성을 유지하되, 미세한 토큰 생성 진동만 잡아주는 스무딩"""
        actual_joy = max(0.0, actual_valence) * self.arousal
        actual_sad = max(0.0, -actual_valence) * (1.0 - self.arousal)
        actual_ang = max(0.0, -actual_valence) * self.arousal

        error_joy = abs(expected_emotions.get("JOY", 0.0) - actual_joy)
        error_sad = abs(expected_emotions.get("SAD", 0.0) - actual_sad)
        error_ang = abs(expected_emotions.get("ANG", 0.0) - actual_ang)

        raw_error = (error_joy + error_sad + error_ang) / 3.0

        # 종의 특성(경량 모델의 날것의 노이즈)을 해치지 않는 선에서,
        # 미세한 수학적 진동(0.15 미만)으로 인해 무의미한 RPE 스파이크가 튀는 것만 절반으로 부드럽게 깎아줍니다.
        if raw_error < 0.15:
            return raw_error * 0.5
        return raw_error

    def _store_learned_emotion(
        self,
        context_text: str,
        query_vector: list[float],
        actual_valence: float,
        expected_emotions: dict[str, float],
    ) -> None:
        expected_valence = self._expected_valence_scalar(expected_emotions)
        # 감정의 항상성을 유지하는 음성 피드백 루프 (행복회로 / 겸손 논리)
        # 1) 행복회로: 현실 시궁창(Act:-1.0)에서 희망(Exp:1.0)을 품으면, 부정적 감정이 덜어짐 (-1.0 -> -0.3)
        # 2) 겸손: 현실 날뜀(Act:1.0)에서 차분히 누르면(Exp:-1.0), 과각성이 진정되어 안착함 (1.0 -> 0.3)
        learned_valence = actual_valence + self.SURPRISE_ENCODING_GAIN * (
            expected_valence - actual_valence
        )
        learned_valence = self._clamp(learned_valence, -1.0, 1.0)
        stable_id = hashlib.sha256(context_text.encode("utf-8")).hexdigest()[:24]

        self._chroma_call(
            lambda: self.db.upsert(
                embeddings=[query_vector],
                documents=[context_text],
                metadatas=[{"valence": float(learned_valence)}],
                ids=[f"emo_{stable_id}"],
            )
        )

    @staticmethod
    def _expected_valence_scalar(expected_emotions: dict[str, float]) -> float:
        expected_joy = expected_emotions.get("JOY", 0.0)
        expected_sad = expected_emotions.get("SAD", 0.0)
        expected_ang = expected_emotions.get("ANG", 0.0)
        return max(min(expected_joy - expected_sad - expected_ang, 1.0), -1.0)

    def _generate_token(self, surprise: float, valence: float, arousal: float, mood: float) -> str:
        """주어진 수치를 바탕으로 독립적인 에크만 상태 스트링 토큰을 동적으로 빌드합니다."""
        if surprise > self.EKMAN_THRESHOLD:
            base_emotion = "SURPRISE (DELIGHTED)" if valence > 0.2 else "FEAR (STARTLED)"
        elif valence > 0.35:
            base_emotion = "JOY (EXCITED)" if arousal > 0.6 else "TRUST (SAFE)"
        elif valence < -0.35:
            base_emotion = "FEAR (ALARMED)" if arousal > 0.6 else "SADNESS (HURT)"
        else:
            base_emotion = "CURIOSITY (RESTLESS)" if arousal > 0.5 else "NEUTRAL (CALM)"

        return (
            f"<EKMAN:{base_emotion} | VAL:{valence:.2f} | "
            f"MOOD:{mood:.2f} | ARO:{arousal:.2f} | RPE:{surprise:.2f}>"
        )

    @staticmethod
    def _blend(old_value: float, new_value: float, old_weight: float) -> float:
        return (old_value * old_weight) + (new_value * (1.0 - old_weight))

    @classmethod
    def _open_chroma_client(cls, db_path: str) -> Any:
        return cls._retry_chroma(lambda: chromadb.PersistentClient(path=db_path))

    @classmethod
    def _chroma_call(cls, operation: Callable[[], Any], fallback: Any = _NO_CHROMA_FALLBACK) -> Any:
        return cls._retry_chroma(operation, fallback=fallback)

    @classmethod
    def _retry_chroma(cls, operation: Callable[[], Any], fallback: Any = _NO_CHROMA_FALLBACK) -> Any:
        last_error: Exception | None = None
        for attempt in range(cls.CHROMA_RETRY_COUNT):
            try:
                return operation()
            except Exception as error:
                last_error = error
                if attempt < cls.CHROMA_RETRY_COUNT - 1:
                    time.sleep(cls.CHROMA_RETRY_BASE_DELAY * (2 ** attempt))

        if fallback is not _NO_CHROMA_FALLBACK:
            return fallback
        if last_error:
            raise last_error
        raise RuntimeError("ChromaDB operation failed without an exception.")

    @staticmethod
    def _clamp(value: float, lower_bound: float, upper_bound: float) -> float:
        return max(min(value, upper_bound), lower_bound)
