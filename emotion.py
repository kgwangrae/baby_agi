from __future__ import annotations

import hashlib

from config import EMBEDDING_MODEL_NAME, apply_model_cache_policy

apply_model_cache_policy()

import chromadb
from sentence_transformers import SentenceTransformer


class EmotionEngine:
    """Lightweight emotional state and plastic emotion memory."""

    DECAY_AROUSAL = 0.85
    DECAY_VALENCE = 0.90
    DECAY_MOOD = 0.95

    MOMENTUM_WEIGHT = 0.5
    MOOD_MOMENTUM_WEIGHT = 0.8

    LEARNING_THRESHOLD = 0.3
    EKMAN_THRESHOLD = 0.7
    SURPRISE_ENCODING_GAIN = 0.2
    SIMILAR_EMOTION_COUNT = 5

    def __init__(self, db_path: str = "./emotion_db") -> None:
        self.arousal: float = 0.0
        self.valence: float = 0.0
        self.mood: float = 0.0

        self.encoder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.client = chromadb.PersistentClient(path=db_path)
        self.db = self.client.get_or_create_collection(name="emotion_space")

        if self.db.count() == 0:
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
        return self._generate_token(surprise), self.arousal, surprise

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

        if not context_text or is_silence:
            self._decay_states()
            surprise = 0.0
        else:
            actual_valence, surprise, query_vector = self._calculate_surprise(
                context_text=context_text,
                expected_emotions=expected_emotions,
            )
            has_stimulus = bool(user_message or internal_stimulus or retrieved_memory_context)
            self._update_states(actual_valence, surprise, has_stimulus)

            if surprise > self.LEARNING_THRESHOLD:
                self._store_learned_emotion(
                    context_text=context_text,
                    query_vector=query_vector,
                    actual_valence=actual_valence,
                    expected_emotions=expected_emotions,
                )

        emotion_token = self._generate_token(surprise)
        return emotion_token, self.arousal, surprise

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
        self.db.upsert(
            embeddings=embeddings,
            documents=documents,
            metadatas=[{"valence": float(valence)} for _, valence in seeds],
            ids=[f"seed_{index}" for index in range(len(seeds))],
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

    def _decay_states(self) -> None:
        self.arousal *= self.DECAY_AROUSAL
        self.valence *= self.DECAY_VALENCE
        self.mood *= self.DECAY_MOOD

    def _calculate_surprise(
        self,
        context_text: str,
        expected_emotions: dict[str, float],
    ) -> tuple[float, float, list[float]]:
        query_vector = self.encoder.encode(context_text).tolist()
        result_count = min(self.SIMILAR_EMOTION_COUNT, max(1, self.db.count()))
        query_results = self.db.query(
            query_embeddings=[query_vector],
            n_results=result_count,
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
        actual_joy = max(0.0, actual_valence) * self.arousal
        actual_sad = max(0.0, -actual_valence) * (1.0 - self.arousal)
        actual_ang = max(0.0, -actual_valence) * self.arousal

        error_joy = abs(expected_emotions.get("JOY", 0.0) - actual_joy)
        error_sad = abs(expected_emotions.get("SAD", 0.0) - actual_sad)
        error_ang = abs(expected_emotions.get("ANG", 0.0) - actual_ang)

        return (error_joy + error_sad + error_ang) / 3.0

    def _update_states(self, actual_valence: float, surprise: float, has_stimulus: bool) -> None:
        arousal_gain = 1.5 if has_stimulus else 1.0
        arousal_spike = surprise + abs(actual_valence) * arousal_gain

        self.arousal = self._blend(self.arousal, arousal_spike, self.MOMENTUM_WEIGHT)
        self.valence = self._blend(self.valence, actual_valence, self.MOMENTUM_WEIGHT)
        self.mood = self._blend(self.mood, actual_valence, self.MOOD_MOMENTUM_WEIGHT)

        self.arousal = self._clamp(self.arousal, 0.0, 1.0)
        self.valence = self._clamp(self.valence, -1.0, 1.0)
        self.mood = self._clamp(self.mood, -1.0, 1.0)

    def _store_learned_emotion(
        self,
        context_text: str,
        query_vector: list[float],
        actual_valence: float,
        expected_emotions: dict[str, float],
    ) -> None:
        expected_valence = self._expected_valence_scalar(expected_emotions)
        learned_valence = actual_valence + self.SURPRISE_ENCODING_GAIN * (
            actual_valence - expected_valence
        )
        learned_valence = self._clamp(learned_valence, -1.0, 1.0)
        stable_id = hashlib.sha256(context_text.encode("utf-8")).hexdigest()[:24]

        self.db.upsert(
            embeddings=[query_vector],
            documents=[context_text],
            metadatas=[{"valence": float(learned_valence)}],
            ids=[f"emo_{stable_id}"],
        )

    @staticmethod
    def _expected_valence_scalar(expected_emotions: dict[str, float]) -> float:
        expected_joy = expected_emotions.get("JOY", 0.0)
        expected_sad = expected_emotions.get("SAD", 0.0)
        expected_ang = expected_emotions.get("ANG", 0.0)
        return max(min(expected_joy - expected_sad - expected_ang, 1.0), -1.0)

    def _generate_token(self, surprise: float) -> str:
        if surprise > self.EKMAN_THRESHOLD:
            base_emotion = "SURPRISE (DELIGHTED)" if self.valence > 0.2 else "FEAR (STARTLED)"
        elif self.valence > 0.35:
            base_emotion = "JOY (EXCITED)" if self.arousal > 0.6 else "TRUST (SAFE)"
        elif self.valence < -0.35:
            base_emotion = "FEAR (ALARMED)" if self.arousal > 0.6 else "SADNESS (HURT)"
        else:
            base_emotion = "CURIOSITY (RESTLESS)" if self.arousal > 0.5 else "NEUTRAL (CALM)"

        return (
            f"<EKMAN:{base_emotion} | VAL:{self.valence:.2f} | "
            f"MOOD:{self.mood:.2f} | ARO:{self.arousal:.2f} | RPE:{surprise:.2f}>"
        )

    @staticmethod
    def _blend(old_value: float, new_value: float, old_weight: float) -> float:
        return (old_value * old_weight) + (new_value * (1.0 - old_weight))

    @staticmethod
    def _clamp(value: float, lower_bound: float, upper_bound: float) -> float:
        return max(min(value, upper_bound), lower_bound)
