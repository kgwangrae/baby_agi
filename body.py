from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class BodyState:
    """Shared physical arousal/fatigue state driven by emotion and memory work."""

    # --- 기억 작업과 피로도 ---
    MEMORY_WORK_UNIT = 0.018  # 기억 입출력 1단위가 피로도에 남기는 기본 흔적
    DIGESTION_UNIT = 0.055  # 단기 기억이 정리되어 비워질 때 줄어드는 피로도 단위
    MIN_MEMORY_AROUSAL_LOAD = 0.25  # 낮은 각성 상태에서도 꿈/기억 정리는 완전히 공짜가 아님
    MAX_WRITE_STIMULUS_LOAD = 3.00  # arousal/valence/surprise 합산 자극이 숫자 폭주로 번지지 않게 막는 물리 상한

    # 기억 종류별 저장 부담. 사실/위협/놀람은 더 오래 붙잡히므로 저장 흔적도 조금 더 무겁게 봅니다.
    KIND_WRITE_FATIGUE_BONUS = {
        "fact": 0.50,
        "threat": 0.20,
        "surprise": 0.10,
        "diary": 0.10,
        "reward": 0.08,
        "episode": 0.00,
        "consolidated": 0.00,
    }

    # --- 수면 압력 ---
    SLEEP_PRESSURE_SWITCH_POINT = 0.0  # fatigue가 각성 장벽을 넘으면 수면 압력이 양수로 전환됨
    FATIGUE_REBOUND_WAKE_MARGIN = 0.0  # 수면 중 피로도가 저점을 찍고 다시 오르는지 보는 기준
    PROMPT_LEVEL_COUNT = 5

    arousal: float = 0.0
    fatigue: float = 0.0
    fatigue_delta: float = 0.0
    asleep: bool = False
    sleep_source: str = ""
    sleep_fatigue_floor: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BodyState":
        if not isinstance(data, dict):
            return cls()

        fatigue = cls._clamp(cls.coerce_float(data.get("fatigue", 0.0)), 0.0, 1.0)
        return cls(
            arousal=cls._clamp(cls.coerce_float(data.get("arousal", 0.0)), 0.0, 1.0),
            fatigue=fatigue,
            fatigue_delta=cls._clamp(cls.coerce_float(data.get("fatigue_delta", 0.0)), -1.0, 1.0),
            asleep=bool(data.get("asleep", False)),
            sleep_source=str(data.get("sleep_source", "")),
            sleep_fatigue_floor=cls._clamp(cls.coerce_float(data.get("sleep_fatigue_floor", fatigue)), 0.0, 1.0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "arousal": round(self.arousal, 4),
            "fatigue": round(self.fatigue, 4),
            "fatigue_delta": round(self.fatigue_delta, 4),
            "arousal_barrier": round(self.arousal_barrier, 4),
            "sleep_pressure": round(self.sleep_pressure, 4),
            "asleep": self.asleep,
            "sleep_source": self.sleep_source,
            "sleep_fatigue_floor": round(self.sleep_fatigue_floor, 4),
        }

    def to_memory_metadata(self) -> dict[str, Any]:
        return {
            "body_arousal": round(self.arousal, 4),
            "body_fatigue": round(self.fatigue, 4),
            "body_sleep_pressure": round(self.sleep_pressure, 4),
            "body_asleep": self.asleep,
        }

    def absorb_emotional_state(self, *, valence: float, surprise: float, arousal_hint: float = 0.0) -> None:
        # 감정 엔진의 ARO, 정서 강도(|VAL|), 놀람(RPE)을 독립 자극으로 합칩니다.
        # 하나만 강해도 몸은 각성되고, 여러 자극이 겹치면 1.0에 부드럽게 수렴합니다.
        self.arousal = self.estimate_arousal(
            arousal=arousal_hint,
            valence=valence,
            surprise=surprise,
        )

    @classmethod
    def estimate_arousal(cls, *, arousal: float = 0.0, valence: float = 0.0, surprise: float = 0.0) -> float:
        clean_arousal = cls._clamp(arousal, 0.0, 1.0)
        clean_valence = cls._clamp(abs(cls.coerce_float(valence, 0.0)), 0.0, 1.0)
        clean_surprise = cls._clamp(surprise, 0.0, 1.0)
        return cls._clamp(
            1.0 - ((1.0 - clean_arousal) * (1.0 - clean_valence) * (1.0 - clean_surprise)),
            0.0,
            1.0,
        )

    def on_memory_read(self, item_count: int, *, effort: float = 1.0) -> None:
        safe_count = max(0, int(self.coerce_float(item_count, 0.0)))
        if safe_count <= 0:
            return
        self._add_fatigue(self._work_cost(math.log1p(safe_count) * max(0.0, self.coerce_float(effort, 1.0))))

    def on_memory_write(
            self,
            *,
            memory_kind: str = "episode",
            arousal: float = 0.0,
            valence: float = 0.0,
            surprise: float = 0.0,
    ) -> None:
        clean_kind = str(memory_kind or "episode").lower()
        stimulus_load = (
            abs(self.coerce_float(valence, 0.0))
            + self.coerce_float(surprise, 0.0)
            + self.coerce_float(arousal, 0.0)
        )
        intensity = 1.0 + self._clamp(stimulus_load, 0.0, self.MAX_WRITE_STIMULUS_LOAD)
        intensity += self.KIND_WRITE_FATIGUE_BONUS.get(clean_kind, 0.0)
        self._add_fatigue(self._work_cost(intensity))

    def on_memory_digest(self, *, removed_hot_count: int, refined_count: int = 0) -> None:
        total_change = max(0, int(self.coerce_float(removed_hot_count, 0.0))) + max(0, int(self.coerce_float(refined_count, 0.0)))
        if total_change <= 0:
            return

        recovery = self.DIGESTION_UNIT * math.log1p(total_change) * (1.0 - self.arousal)
        self._add_fatigue(-recovery)

    def mark_sleeping(self, source: str) -> None:
        if not self.asleep:
            self.sleep_fatigue_floor = self.fatigue
        self.asleep = True
        self.sleep_source = str(source or "")

    def mark_awake(self) -> None:
        self.asleep = False
        self.sleep_source = ""
        self.sleep_fatigue_floor = self.fatigue

    @property
    def arousal_barrier(self) -> float:
        # 과각성은 선형보다 급하게 잠을 밀어냅니다. arousal^2가 높은 각성 구간만 강하게 막습니다.
        return self._clamp(self.arousal + (self.arousal * self.arousal), 0.0, 1.0)

    @property
    def sleep_pressure(self) -> float:
        return self.fatigue - self.arousal_barrier

    def should_auto_sleep(self) -> bool:
        return not self.asleep and self.sleep_pressure > self.SLEEP_PRESSURE_SWITCH_POINT

    def should_auto_wake(self) -> bool:
        if not self.asleep or self.sleep_source == "manual":
            return False
        if self.sleep_pressure < self.SLEEP_PRESSURE_SWITCH_POINT:
            return True
        return (
            self.fatigue_delta > self.FATIGUE_REBOUND_WAKE_MARGIN
            and self.fatigue > self.sleep_fatigue_floor
        )

    def to_prompt_context(self) -> str:
        return (
            f"Arousal: {self.arousal:.2f} ({self._level_text(self.arousal)}); "
            f"Fatigue: {self.fatigue:.2f} ({self._level_text(self.fatigue)}); "
            f"Sleep pressure: {self.sleep_pressure:+.2f}; "
            f"State: {'sleeping' if self.asleep else 'awake'}"
        )

    def _work_cost(self, intensity: float) -> float:
        arousal_load = self.MIN_MEMORY_AROUSAL_LOAD + (self.arousal * self.arousal)
        return self.MEMORY_WORK_UNIT * max(0.0, self.coerce_float(intensity, 0.0)) * arousal_load

    def _add_fatigue(self, amount: float) -> None:
        old_fatigue = self.fatigue
        self.fatigue = self._clamp(self.fatigue + self.coerce_float(amount, 0.0), 0.0, 1.0)
        self.fatigue_delta = self.fatigue - old_fatigue
        if self.asleep:
            self.sleep_fatigue_floor = min(self.sleep_fatigue_floor, self.fatigue)

    @classmethod
    def _level_text(cls, value: float) -> str:
        safe_value = cls._clamp(value, 0.0, 1.0)
        level = min(cls.PROMPT_LEVEL_COUNT, max(1, int(safe_value * cls.PROMPT_LEVEL_COUNT) + 1))
        label = ["very low", "low", "medium", "high", "very high"][level - 1]
        return f"level {level}/{cls.PROMPT_LEVEL_COUNT}, {label}"

    @classmethod
    def _clamp(cls, value: float, lower_bound: float, upper_bound: float) -> float:
        safe_value = cls.coerce_float(value, lower_bound)
        return max(min(safe_value, upper_bound), lower_bound)

    @staticmethod
    def coerce_float(value: Any, fallback: float = 0.0) -> float:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return fallback
        if not math.isfinite(numeric_value):
            return fallback
        return numeric_value
