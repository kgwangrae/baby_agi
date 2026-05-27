from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
import math
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from config import EMBEDDING_MODEL_NAME, apply_model_cache_policy

apply_model_cache_policy()

import chromadb
from sentence_transformers import SentenceTransformer


_NO_CHROMA_FALLBACK = object()


class FactNotepad:
    """Deterministic key-value store for facts that should bypass vector search."""

    DEFAULT_FACTS = {
        "정체성 (Identity)": (
            "아빠의 맥북 안에 사는 귀여운 아기 "
            "(A cute AI baby living in Dad's MacBook)"
        ),
        "호칭 (Title)": "아빠 (Dad)",
    }
    INVALID_FACT_KEYS = {
        "",
        "key",
        "fact",
        "emotion",
        "thought",
        "response",
        "expect",
        "tool",
        "none",
        "null",
    }
    MAX_FACT_KEY_CHARS = 80
    MAX_FACT_VALUE_CHARS = 300

    def __init__(self, file_path: str = "facts.json") -> None:
        self.file_path = Path(file_path)
        self.facts = self._load()
        self._save()

    @staticmethod
    def _normalize_key(key: str) -> str:
        # 1단계 기본 정규화: 모든 공백을 제거하고 영문은 소문자 처리하여 완전 매칭 유도
        clean_k = "".join(key.split()).lower()

        # 모델이 임의로 생성하기 쉬운 유사 키 목록 고정 하드매칭 테이블
        mapping = {
            "아빠생일": "아빠 생일 (Dad birthday)",
            "dadbirthday": "아빠 생일 (Dad birthday)",
            "dadsbirthday": "아빠 생일 (Dad birthday)",
            "아기생일": "아기 생일 (Baby birthday)",
            "babybirthday": "아기 생일 (Baby birthday)",
            "babysbirthday": "아기 생일 (Baby birthday)",
            "아빠이름": "아빠 이름 (Dad name)",
            "dadname": "아빠 이름 (Dad name)",
            "dadsname": "아빠 이름 (Dad name)",
            "아기이름": "아기 이름 (Baby name)",
            "babyname": "아기 이름 (Baby name)",
            "babysname": "아기 이름 (Baby name)",
        }
        return mapping.get(clean_k, key.strip())

    def add_fact(self, key: str, value: str) -> bool:
        normalized_key = self._normalize_key(key)
        clean_key = self._clean_fact_text(normalized_key)
        clean_value = self._clean_fact_text(value)
        if not self._is_valid_fact(clean_key, clean_value):
            return False

        self.facts[clean_key] = clean_value
        self._save()
        return True

    def get_all(self) -> str:
        return "\n".join(f"- {key}: {value}" for key, value in self.facts.items())

    def _load(self) -> dict[str, str]:
        if not self.file_path.exists():
            return dict(self.DEFAULT_FACTS)

        try:
            with self.file_path.open("r", encoding="utf-8") as file:
                loaded_facts = json.load(file)
        except (json.JSONDecodeError, OSError):
            return dict(self.DEFAULT_FACTS)

        if not isinstance(loaded_facts, dict):
            return dict(self.DEFAULT_FACTS)

        cleaned_facts = dict(self.DEFAULT_FACTS)
        for key, value in loaded_facts.items():
            normalized_key = self._normalize_key(str(key))
            clean_key = self._clean_fact_text(normalized_key)
            clean_value = self._clean_fact_text(str(value))
            if self._is_valid_fact(clean_key, clean_value):
                cleaned_facts[clean_key] = clean_value

        return cleaned_facts

    def _save(self) -> None:
        temp_path = self.file_path.with_suffix(f"{self.file_path.suffix}.tmp")
        with temp_path.open("w", encoding="utf-8") as file:
            json.dump(self.facts, file, ensure_ascii=False, indent=4)
        os.replace(temp_path, self.file_path)

    @classmethod
    def _is_valid_fact(cls, key: str, value: str) -> bool:
        lower_key = key.strip().lower()
        lower_value = value.strip().lower()
        if lower_key in cls.INVALID_FACT_KEYS:
            return False
        if lower_value in {"", "none", "null", "n/a"}:
            return False
        if len(key) > cls.MAX_FACT_KEY_CHARS or len(value) > cls.MAX_FACT_VALUE_CHARS:
            return False
        if re.fullmatch(r"[-+]?\d+(\.\d+)?", value.strip()):
            return False
        return True

    @staticmethod
    def _clean_fact_text(text: str) -> str:
        text = re.sub(r"[\u4e00-\u9fff]+", "", text)
        text = re.sub(r"<\|.*?\|>", "", text)
        text = text.replace("<FACT>", "").replace("</FACT>", "")
        return " ".join(text.strip().split())


class MemoryManager:
    """Tiered episodic memory backed by ChromaDB hot and cold collections."""
    FORGET_THRESHOLD = 0.15  # 기본 삭제 문턱값 (이 점수보다 각성도가 낮으면 기억 삭제)
    ARCHIVE_MARGIN = 0.10  # 기본 아카이브 마진 (삭제는 면했지만 이 마진보다 낮으면 장기 기억으로 이동)

    # 실험해보니 인위적으로 이 값을 제한하는 건 기억의 자유로운 정리를 심하게 막는 것으로 보여짐
    MAX_FORGET_THRESHOLD = 1.00  # 삭제 문턱값의 상한선 (1.00으로 설정해 사실상 제한 해제)
    MAX_ARCHIVE_THRESHOLD = 1.00  # 아카이브 문턱값의 상한선 (1.00으로 설정해 사실상 제한 해제)

    # --- 인지 부하 및 감정 가드레일 ---
    MIN_ARCHIVE_MARGIN = 0.04  # 과부하로 마진이 압축될 때, 삭제와 보존 구역이 겹치지 않게 막는 최소 버퍼
    ADAPTIVE_FORGET_GAIN = 0.35  # 정보 과부하(밀린 작업) 발생 시 망각 속도를 얼마나 가속할지 결정하는 민감도
    LOW_VALENCE_KEEP_THRESHOLD = -0.25  # 감정 수치(Valence)가 이 기준보다 낮으면 '강한 부정 감정(상처)'으로 판단해 삭제 방지

    # --- 기억 종류별 망각 저항 보너스 (각성도에 더해져서 잘 안 잊히게 만듦) ---
    KIND_RETENTION_BONUS = {
        "fact": 0.35,  # 사실성 정보 보존력 최상
        "threat": 0.30,  # 위협/경고 정보 보존력 상
        "consolidated": 0.20,  # 이미 한 번 압축된 기억 보존력 중
        "reward": 0.16,  # 보상/칭찬 기억 보존력 중하
        "surprise": 0.12,  # 놀라운 사건 기억 보존력 하
        "diary": 0.12,  # 일기 형태 기억 보존력 하
        "episode": 0.00,  # 일반 일상 대화는 보너스 없음 (가장 쉽게 잊힘)
    }

    # --- 기억 종류별 장기 기억(Cold) 전환 압력 가중치 (지속성 점수) ---
    KIND_CONSOLIDATION_BONUS = {
        "fact": 0.80,  # 지식성 팩트는 무조건 장기 기억으로 넘기도록 강력 유도
        "threat": 0.70,  # 위험 신호도 장기 기억 전환 최우선 순위
        "consolidated": 0.10,  # 이미 압축된 것은 중복 처리 방지를 위해 최하위
        "reward": 0.35,  # 긍정 보상 기억 전환율 중
        "surprise": 0.50,  # 놀라운 사건은 장기 기억으로 비교적 잘 넘어감
        "diary": 0.45,  # 기록 가치가 있는 일기 전환율 중상
        "episode": 0.20,  # 일반 일상은 장기 기억 전환율 낮음
    }

    # --- 기억 종류별 조회 가능성 보너스 (이성 엔진에 먼저 떠오르게 만드는 정도) ---
    KIND_RETRIEVAL_BONUS = {
        "fact": 0.08,
        "threat": 0.10,
        "consolidated": 0.04,
        "reward": 0.04,
        "surprise": 0.06,
        "diary": 0.04,
        "episode": 0.00,
    }

    # --- 구조화 배정 및 연산 범위 ---
    RESTRUCTURE_BATCH_SIZE = 300  # 한 번에 스캔하고 정리할 단기 기억(Hot)의 기본 묶음 크기
    RESTRUCTURE_FETCH_MULTIPLIER = 3  # 과부하 시 기본 묶음의 최대 몇 배 영역까지 스캔 오프셋 범위를 넓힐지 결정

    # --- 인지 임계치 및 검색 제한 ---
    TRAUMA_THRESHOLD = 0.6  # 트라우마성 각인 기준 (이 각성도를 넘으면 위협 기억으로 강하게 보존 유도)
    LEARNING_SURPRISE_THRESHOLD = 0.3  # 새로 배운 것(깨달음)으로 인정하여 삭제를 면하게 해주는 놀라움 기준점
    HOT_RESULTS_PER_QUERY = 2  # 대화 맥락 탐색 시 단기 기억(Hot)에서 가져올 유사 기억 개수
    COLD_RESULTS_PER_QUERY = 2  # 대화 맥락 탐색 시 장기 기억(Cold)에서 가져올 유사 기억 개수
    FLASHBACK_SAMPLE_ATTEMPTS = 10  # 수면/대기 중 유효한 꿈(플래시백)을 찾기 위해 무작위 찌르기를 시도할 횟수
    RECENT_MEMORY_SCAN_LIMIT = 300  # 플래시백 및 대화 흐름 추적 시 훑어볼 최신 기억의 물리적 한계선

    # --- 기억 조회 관련 감정 영향도 ---
    RETRIEVAL_FETCH_MULTIPLIER = 4  # 유사도 후보를 조금 넓게 뽑은 뒤 감정 점수로 재정렬
    RETRIEVAL_SURPRISE_TRAUMA_WEIGHT = 0.2
    RETRIEVAL_MOOD_RESONANCE_WEIGHT = 0.08  # 현재 기분과 같은 방향의 정서 기억은 더 쉽게 떠오름 (기분 좋을땐 좋은 생각)
    RETRIEVAL_AROUSAL_RESONANCE_WEIGHT = 0.04  # 현재 각성도와 비슷한 강도의 기억은 약하게 공명
    # 아래 상수 합은 1로 맞추는 것을 권장 (KIND_RETRIEVAL_BONUS와 적정 스케일로 맞출 것)
    RETRIEVAL_SIMILARITY_WEIGHT = 0.62  # 의미적 관련성은 항상 1순위로 유지 (이성 판단)
    RETRIEVAL_AROUSAL_WEIGHT = 0.18  # 각성도가 높은 기억은 현재 판단에 더 잘 끼어듦
    RETRIEVAL_SURPRISE_WEIGHT = 0.14  # 놀라웠던 기억은 학습 신호로 더 잘 떠오름
    RETRIEVAL_VALENCE_WEIGHT = 0.06  # 좋든 싫든 감정 절댓값이 큰 기억은 약하게 보정 (negative loop, 안정성 확보)

    # --- 시간 감쇄 상수 ---
    EMOTIONAL_TIME_DECAY_LAMBDA = 0.015  # 정서 기억의 지수 감쇄율
    FACT_TIME_DECAY_RATE = 0.005  # 팩트 기억의 선형 감쇄율
    EPISODE_TIME_DECAY_RATE = 0.02  # 일반 일상 기억의 선형 감쇄율
    MIN_TIME_DECAY = 0.10  # 시간이 오래 지나도 남겨둘 최소 흔적

    # --- 압축 및 재정리 상수 ---
    ARCHIVE_CHUNK_SIZE = 7  # 한 번에 압축할 기억 묶음 크기
    LONG_TERM_REFINE_PROBABILITY = 0.05  # 수면 루프에서 기존 장기기억을 재압축할 확률

    # --- 데이터 파이프라인 및 DB 가드레일 ---
    CONSOLIDATED_PREVIEW_CHARS = 2_000  # 장기 기억으로 압축(요약)할 때 컨텍스트로 밀어 넣을 최대 텍스트 길이
    MAX_MEMORY_CONTENT_CHARS = 3_000  # 크로마 DB에 들어가는 단일 기억 조각의 최대 글자 수 제한
    CHROMA_RETRY_COUNT = 3  # Chroma DB 연결 지연 또는 에러 발생 시 최대 재시도 횟수
    CHROMA_RETRY_BASE_DELAY = 0.5  # DB 재시도 간격의 시작 대기 시간(초 단위, 지수 백오프 적용)

    def __init__(self, db_path: str = "./memory_db") -> None:
        self.encoder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.client = self._open_chroma_client(db_path)
        self.hot_storage = self._chroma_call(lambda: self.client.get_or_create_collection(name="hot_episodic"))
        self.cold_storage = self._chroma_call(lambda: self.client.get_or_create_collection(name="cold_archive"))

    def store_memory(
        self,
        content: str,
        emotion_token: str,
        arousal_score: float,
        *,
        valence_score: float = 0.0,
        surprise_score: float = 0.0,
        memory_kind: str = "episode",
    ) -> None:
        clean_content = content.strip()[:self.MAX_MEMORY_CONTENT_CHARS]
        if not clean_content or self._is_empty_memory(clean_content):
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        content_hash = hashlib.sha256(clean_content.encode("utf-8")).hexdigest()[:12]
        doc_id = f"mem_{timestamp}_{content_hash}"

        self._chroma_call(
            lambda: self.hot_storage.add(
                documents=[clean_content],
                embeddings=[self._embed_text(clean_content)],
                metadatas=[
                    {
                        "emotion": emotion_token,
                        "arousal": float(arousal_score),
                        "valence": float(valence_score),
                        "surprise": float(surprise_score),
                        "kind": memory_kind,
                        "time": timestamp,
                    }
                ],
                ids=[doc_id],
            )
        )

    def retrieve_memory(
            self,
            query: str,
            current_arousal: float = 0.0,
            current_mood: float = 0.0,
    ) -> list[str]:
        clean_query = query.strip()
        if not clean_query:
            return []

        query_embedding = self._embed_text(clean_query)
        retrieved_candidates: list[tuple[float, str]] = []
        retrieved_candidates.extend(
            self._query_collection(
                self.hot_storage,
                query_embedding,
                self.HOT_RESULTS_PER_QUERY,
                "VIVID",
                current_arousal,
                current_mood,
            )
        )
        retrieved_candidates.extend(
            self._query_collection(
                self.cold_storage,
                query_embedding,
                self.COLD_RESULTS_PER_QUERY,
                "DISTANT",
                current_arousal,
                current_mood,
            )
        )

        retrieved_candidates.sort(key=lambda item: item[0], reverse=True)
        result_limit = self.HOT_RESULTS_PER_QUERY + self.COLD_RESULTS_PER_QUERY
        return [memory for _, memory in retrieved_candidates[:result_limit]]

    def retrieve_trauma(self) -> str:
        query_results = self._chroma_call(
            lambda: self.hot_storage.get(
                where={"kind": "threat"},
                limit=50,
                include=["documents", "metadatas"],
            ),
            fallback={},
        )
        documents = query_results.get("documents") if query_results else None
        metadatas = query_results.get("metadatas") if query_results else None
        if not documents or not metadatas:
            return ""

        weighted_documents = []
        for document, metadata in zip(documents, metadatas):
            if self._is_empty_memory(document):
                continue
            valence = float(metadata.get("valence", 0.0))
            arousal = float(metadata.get("arousal", 0.0))
            surprise = float(metadata.get("surprise", 0.0))
            weight = arousal + max(0.0, -valence) + self.RETRIEVAL_SURPRISE_TRAUMA_WEIGHT * surprise
            weighted_documents.append((document, max(weight, 0.01)))

        return self._weighted_choice(weighted_documents) if weighted_documents else ""

    def retrieve_flashback(self) -> str:
        collections = [self.hot_storage, self.cold_storage]
        for _ in range(self.FLASHBACK_SAMPLE_ATTEMPTS):
            collection = random.choice(collections)
            document = self._sample_document(collection)
            if document and "[CONSOLIDATED]" not in document and not self._is_empty_memory(document):
                return document
        return ""

    def get_recent_memories(self, limit: int = 5) -> list[str]:
        memories: list[tuple[str, str]] = []
        memories.extend(self._recent_collection_items(self.hot_storage, limit, "HOT"))
        memories.extend(self._recent_collection_items(self.cold_storage, limit, "COLD"))
        memories.sort(key=lambda item: item[0], reverse=True)
        return [memory for _, memory in memories[:limit]]

    def restructure_hierarchical_memory(self, cortex: Any, verbose = False) -> int:
        hot_count = self._chroma_call(self.hot_storage.count, fallback=0)
        if hot_count <= 0:
            return 0

        # [1. 정보 간섭과 평균 회귀]
        # 인지 모방 : 평소 대비 많은 일이나 정보 과부하가 있었으면, 망각도 더 많이. 안정 시 기본 상태로 부드럽게 복원됩니다.
        # 즉, 단기(HOT) 기억이 일반적 망각 배치 크기를 초과할 때만 정리 압력이 증가합니다.
        backlog_pressure = max(
            0.0,
            (hot_count - self.RESTRUCTURE_BATCH_SIZE) / max(1, self.RESTRUCTURE_BATCH_SIZE),
        )
        adaptive_multiplier = 1.0 + self.ADAPTIVE_FORGET_GAIN * math.log1p(backlog_pressure)
        dynamic_forget_threshold = min(self.MAX_FORGET_THRESHOLD, self.FORGET_THRESHOLD * adaptive_multiplier)
        dynamic_margin = max(self.MIN_ARCHIVE_MARGIN, self.ARCHIVE_MARGIN / adaptive_multiplier)

        fetch_limit = min(
            max(self.RESTRUCTURE_BATCH_SIZE, hot_count),
            self.RESTRUCTURE_BATCH_SIZE * self.RESTRUCTURE_FETCH_MULTIPLIER,
        )

        # 인지 모사 구현: 전체를 균등하게 찌르지 않고, 앞쪽(최신/과부하 구간)을 집중 스캔합니다.
        # random.betavariate(1.0, 3.0)은 무조건 0~1 사이를 주되, 0에 가까운 값이 나올 확률이 압도적으로 높습니다.
        # 이를 통해 Empty Scan을 원천 차단하고, 데이터가 밀집된 앞쪽을 주로 타겟팅 합니다.
        max_space = max(0, hot_count - self.RESTRUCTURE_BATCH_SIZE)
        bias_ratio = random.betavariate(1.0, 3.0)
        scan_offset = int(bias_ratio * max_space)

        if verbose:
            print(
                f"[Memory Restructure] DB Count: {hot_count} | Scan Range: 0~{max_space} | "
                f"Selected Spot: {scan_offset} (Ratio: {bias_ratio:.2f})"
            )

        hot_memories = self._chroma_call(
            lambda: self.hot_storage.get(
                limit=fetch_limit,
                offset=scan_offset,
                include=["documents", "metadatas"],
            ),
            fallback={},
        )

        ids = hot_memories.get("ids") if hot_memories else None
        documents = hot_memories.get("documents") if hot_memories else None
        metadatas = hot_memories.get("metadatas") if hot_memories else None
        if not ids or not documents or not metadatas:
            if verbose:
                print(f"[Memory Restructure] Early Exit | No memories fetched at offset {scan_offset}.")
            return 0

        parsed_items = []
        decayed_arousals = []
        now_dt = datetime.now()

        # [2. 이중 경로 감쇄(Dual-path Decay) 적용]
        for doc_id, document, metadata in zip(ids, documents, metadatas):
            arousal = float(metadata.get("arousal", 0.0))
            kind = str(metadata.get("kind", "episode")).lower()
            ts_str = metadata.get("time", "")

            hours_passed = 0.0
            if ts_str and '_' in ts_str:
                try:
                    parts = ts_str.split('_')
                    dt = datetime.strptime(parts[0] + '_' + parts[1], "%Y%m%d_%H%M%S")
                    hours_passed = (now_dt - dt).total_seconds() / 3600.0
                except Exception as error:
                    if verbose:
                        print(f"[System] Memory decay - timestamp parse failure {error}")
                    pass

            # 정서 데이터(위협, 보상, 놀람 등)는 지수 감쇄로 은은하게 보존
            if kind in {"threat", "reward", "surprise", "diary"}:
                time_decay = math.exp(-self.EMOTIONAL_TIME_DECAY_LAMBDA * hours_passed)
            # 일상/팩트 데이터는 선형 감쇄로 시스템 자원 관리를 위해 단호하게 삭제
            elif kind == "fact":
                time_decay = max(self.MIN_TIME_DECAY, 1.0 - (self.FACT_TIME_DECAY_RATE * hours_passed))
            else:
                time_decay = max(self.MIN_TIME_DECAY, 1.0 - (self.EPISODE_TIME_DECAY_RATE * hours_passed))

            decayed_arousals.append(arousal * time_decay)
            parsed_items.append((doc_id, document, metadata, time_decay))

        average_arousal = sum(decayed_arousals) / max(1, len(decayed_arousals))
        # 인지 부하 비중에 따라 망각 문턱값과 마진을 동적으로 스케일링 (생물학적 간섭 메커니즘 모사)
        archive_threshold = min(
            self.MAX_ARCHIVE_THRESHOLD,
            max(dynamic_forget_threshold + 0.03, average_arousal - dynamic_margin),
        )

        if verbose:
            print(
                f"[Memory Restructure] Fetched: {len(ids)} | Avg Arousal: {average_arousal:.4f} -> Archive Thresh: {archive_threshold:.4f}"
            )

        ids_to_delete: list[str] = []
        archive_candidates: list[tuple[str, str, dict]] = []

        for doc_id, document, metadata, time_decay in parsed_items:
            kind = str(metadata.get("kind", "episode")).lower()
            arousal = float(metadata.get("arousal", 0.0))
            surprise = float(metadata.get("surprise", 0.0))
            valence = float(metadata.get("valence", 0.0))

            if self._is_empty_memory(document):
                ids_to_delete.append(doc_id)
                continue

            retention_bonus = self.KIND_RETENTION_BONUS.get(kind, 0.0)
            effective_arousal = (arousal + retention_bonus) * time_decay

            should_delete = (
                    effective_arousal < dynamic_forget_threshold
                    and surprise < self.LEARNING_SURPRISE_THRESHOLD
                    and valence > self.LOW_VALENCE_KEEP_THRESHOLD
            )

            if should_delete:
                ids_to_delete.append(doc_id)
                continue

            if effective_arousal < archive_threshold:
                archive_candidates.append((doc_id, document, metadata))

        if ids_to_delete:
            self._chroma_call(lambda: self.hot_storage.delete(ids=ids_to_delete))

        if archive_candidates:
            self._archive_hot_memories(archive_candidates, cortex)

        if verbose:
            print(
                f"[Memory Restructure] Complete | Deleted: {len(ids_to_delete)}, Archived: {len(archive_candidates)}"
            )

        # [3. 장기 기억 재압축 (Cold Memory Refinement)]
        # 단기 기억 정리가 끝난 후 일정 확률로 기존 장기기억 하나를 더 단단하게 재압축
        if random.random() < self.LONG_TERM_REFINE_PROBABILITY:
            self._refine_long_term_memory(cortex, verbose)

        return len(ids_to_delete) + len(archive_candidates)

    def _refine_long_term_memory(self, cortex: Any, verbose = False) -> int:
        """장기 기억 하나를 무작위로 꺼내어 재압축한 뒤 원본을 덮어씁니다."""
        cold_count = self._chroma_call(self.cold_storage.count, fallback=0)
        if cold_count == 0:
            return 0

        # 완전히 무작위로 하나를 뽑음 (장기기억은 고르게 중요하다는 가정)
        offset = random.randint(0, cold_count - 1)
        results = self._chroma_call(
            lambda: self.cold_storage.get(limit=1, offset=offset, include=["documents", "metadatas"]),
            fallback={}
        )

        ids = results.get("ids")
        docs = results.get("documents")
        metas = results.get("metadatas")

        if not ids or not docs or not metas:
            return 0

        old_id, old_doc, old_meta = ids[0], docs[0], metas[0]

        # 이미 충분히 짧다면 스킵 (토큰 낭비 방지)
        if len(old_doc) < 200:
            return 0

        # 추론 엔진을 통해 압축 진행
        new_doc = cortex.compress_memories(text=old_doc, fallback_text=old_doc)

        if new_doc and new_doc != old_doc:
            old_meta["refined"] = True  # 재압축되었음을 메타데이터에 흔적으로 남김
            self._chroma_call(
                lambda: self.cold_storage.upsert(
                    ids=[old_id],  # 동일한 ID로 덮어쓰기
                    documents=[new_doc],
                    embeddings=[self._embed_text(new_doc)],
                    metadatas=[old_meta]
                )
            )
            if verbose:
                print(f"[System] Long-term memory refined & densified (ID: {old_id[:12]} / DOC: {old_doc[:120]}...)")
            return 1
        return 0

    def _parse_time_ago(self, ts_str: str) -> str:
        """타임스탬프 문자열을 계산하여 최신성(Recency) 인지용 거리 문자열로 변환합니다."""
        if not ts_str or not isinstance(ts_str, str) or ts_str in ("unknown", "N/A"):
            return "sometime ago"

        try:
            parts = ts_str.split('_')
            if len(parts) < 2 or len(parts[0]) != 8 or len(parts[1]) != 6:
                return "sometime ago"
            dt = datetime.strptime(parts[0] + '_' + parts[1], "%Y%m%d_%H%M%S")
            diff = datetime.now() - dt

            if diff.days > 0:
                return f"{diff.days} days ago"
            hours = diff.seconds // 3600
            if hours > 0:
                return f"{hours} hours ago"
            minutes = diff.seconds // 60
            if minutes > 0:
                return f"{minutes} mins ago"
            return "just now"
        except Exception as error:
            print(f"[System] memory / _parse_time_ago failed : {error}")
            return "sometime ago"

    def _query_collection(
        self,
        collection: Any,
        query_embedding: list[float],
        result_count: int,
        label: str,
        current_arousal: float = 0.0,
        current_mood: float = 0.0,
    ) -> list[tuple[float, str]]:
        collection_count = self._chroma_call(collection.count, fallback=0)
        if collection_count == 0:
            return []

        fetch_count = min(
            result_count * self.RETRIEVAL_FETCH_MULTIPLIER,
            collection_count,
        )

        query_results = self._chroma_call(
            lambda: collection.query(
                query_embeddings=[query_embedding],
                n_results=fetch_count,
                include=["documents", "metadatas", "distances"],
            ),
            fallback={},
        )
        documents = query_results.get("documents") or [[]]
        metadatas = query_results.get("metadatas") or [[]]
        distances = query_results.get("distances") or [[]]

        memories: list[tuple[float, str]] = []
        for document, metadata, distance in zip(documents[0], metadatas[0], distances[0]):
            if self._is_empty_memory(document):
                continue

            timestamp = metadata.get("time", "unknown") if metadata else "unknown"
            kind = str(metadata.get("kind", "episode")).lower() if metadata else "episode"
            arousal = float(metadata.get("arousal", 0.0)) if metadata else 0.0
            surprise = float(metadata.get("surprise", 0.0)) if metadata else 0.0
            valence = float(metadata.get("valence", 0.0)) if metadata else 0.0

            # Chroma distance는 낮을수록 가깝기 때문에 0~1에 가까운 유사도 점수로 바꿉니다.
            # TODO : Personality Drift 구현의 시작점?
            # 가끔 감정적으로 연상되는 기억이 튀어나옴 → 좋음. 개성/선호/상태 반영.
            # 그런데 계속 관련 없는 기억이 답변을 오염함 → 나쁨. retrieval 스케일 조정 필요 (지나친 딴소리)
            # 혹은, 특정 threat/reward가 거의 항상 튀어나옴 → kind/arousal 가중치 과함 (지나친 과민 반응)
            similarity_score = 1.0 / (1.0 + max(0.0, float(distance)))
            mood_resonance = max(0.0, current_mood * valence)
            arousal_resonance = 1.0 - min(1.0, abs(current_arousal - arousal))
            affective_score = (
                    self.RETRIEVAL_SIMILARITY_WEIGHT * similarity_score
                    + self.RETRIEVAL_AROUSAL_WEIGHT * arousal
                    + self.RETRIEVAL_SURPRISE_WEIGHT * surprise
                    + self.RETRIEVAL_VALENCE_WEIGHT * abs(valence)
                    + self.RETRIEVAL_MOOD_RESONANCE_WEIGHT * mood_resonance
                    + self.RETRIEVAL_AROUSAL_RESONANCE_WEIGHT * arousal_resonance
                    + self.KIND_RETRIEVAL_BONUS.get(kind, 0.0)
            )

            time_ago = self._parse_time_ago(timestamp)
            importance = "CRITICAL" if arousal > 0.6 else "SIGNIFICANT" if arousal > 0.3 else "NORMAL"

            memories.append((
                affective_score,
                f"[{label}/{kind}: {timestamp}] "
                f"(Recency: {time_ago}, Importance: {importance}, Arousal: {arousal:.2f}) {document}"
            ))

        memories.sort(key=lambda item: item[0], reverse=True)
        return memories[:result_count]

    def _archive_hot_memories(
            self,
            archive_candidates: list[tuple[str, str, dict]],
            cortex: Any
    ) -> None:
        archive_candidates = [
            candidate for candidate in archive_candidates if not self._is_empty_memory(candidate[1])
        ]
        if not archive_candidates:
            return

        # 뇌 모사 개선: 수백 개의 기억을 한 번에 뭉뚱그리다 버려지는 현상 방지.
        # 미니 배치로 쪼개어 각 맥락의 밀도를 유지하며 장기 기억화합니다.
        for i in range(0, len(archive_candidates), self.ARCHIVE_CHUNK_SIZE):
            chunk = archive_candidates[i: i + self.ARCHIVE_CHUNK_SIZE]

            ids = [candidate[0] for candidate in chunk]
            documents = [candidate[1] for candidate in chunk]
            metadatas = [candidate[2] for candidate in chunk]

            # 청크 내에 기억이 단 하나라면 압축 없이 바로 Cold로 이동
            if len(ids) == 1:
                self._chroma_call(
                    lambda: self.cold_storage.upsert(
                        ids=ids,
                        documents=documents,
                        embeddings=[self._embed_text(documents[0])],
                        metadatas=metadatas,
                    )
                )
                self._chroma_call(lambda: self.hot_storage.delete(ids=ids))
                continue

            # 작은 묶음이므로 2,000자 글자수 한계선 내에서 대체로 핵심 기억이 압축 엔진으로 진입함
            consolidated_document, consolidated_metadata, consolidated_id = self._consolidate_memories(
                documents,
                metadatas,
                cortex,
            )

            # 압축 실패 시 원본 유실 방지를 위해 이번 청크 처리를 건너뜀 (다음 수면에 재시도)
            if not consolidated_document or not consolidated_metadata or not consolidated_id:
                sample_doc = documents[0][:200] if documents and documents[0] else "No documents"
                print(f"[System / ERROR] Chunk memory consolidation failed. "
                      f"Preserved {len(ids)} hot memories. (Sample: {sample_doc}...)")
                continue

            # 성공한 청크 단위만 Cold로 안전하게 이관 후 Hot에서 삭제
            self._chroma_call(
                lambda: self.cold_storage.upsert(
                    ids=[consolidated_id],
                    documents=[consolidated_document],
                    embeddings=[self._embed_text(consolidated_document)],
                    metadatas=[consolidated_metadata],
                )
            )
            self._chroma_call(lambda: self.hot_storage.delete(ids=ids))

    def _consolidate_memories(
            self,
            documents: list[str],
            metadatas: list[dict],
            cortex: Any,
    ) -> tuple[str, dict, str]:
        scored_fragments: list[tuple[float, str, str, dict]] = []

        for document, metadata in zip(documents, metadatas):
            clean_document = (
                document.replace("[EPISODE]", "")
                .replace("[TRAUMA]", "")
                .replace("[CONSOLIDATED]", "")
                .strip()
            )
            if self._is_empty_memory(clean_document):
                continue

            kind = str(metadata.get("kind", "episode")).lower()
            arousal = float(metadata.get("arousal", 0.0))
            surprise = float(metadata.get("surprise", 0.0))
            valence = float(metadata.get("valence", 0.0))

            priority_score = (
                    0.50 * arousal
                    + 0.70 * surprise
                    + 0.35 * abs(valence)
                    + self.KIND_CONSOLIDATION_BONUS.get(kind, 0.0)
            )

            model_fragment = (
                f"[kind={kind}, arousal={arousal:.2f}, "
                f"surprise={surprise:.2f}, valence={valence:.2f}] "
                f"{clean_document}"
            )

            scored_fragments.append((priority_score, model_fragment, clean_document, metadata))

        scored_fragments.sort(key=lambda item: item[0], reverse=True)

        model_parts: list[str] = []
        fallback_parts: list[str] = []
        total_chars = 0

        for _, model_fragment, clean_document, _ in scored_fragments:
            if total_chars >= self.CONSOLIDATED_PREVIEW_CHARS:
                break

            remaining_chars = self.CONSOLIDATED_PREVIEW_CHARS - total_chars
            model_preview = model_fragment[:remaining_chars]
            clean_preview = clean_document[:remaining_chars]

            model_parts.append(model_preview)
            fallback_parts.append(clean_preview)
            total_chars += len(model_preview)

        model_input = " ".join(model_parts).strip()
        fallback_text = " ".join(fallback_parts).strip()

        if not model_input and not fallback_text:
            return "", {}, ""

        compressed_text = cortex.compress_memories(
            model_input,
            fallback_text=fallback_text,
        ).strip()

        if not compressed_text:
            return "", {}, ""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        consolidated_document = f"[CONSOLIDATED] {compressed_text}"

        divisor = max(1, len(metadatas))
        average_arousal = sum(float(m.get("arousal", 0.0)) for m in metadatas) / divisor
        average_valence = sum(float(m.get("valence", 0.0)) for m in metadatas) / divisor
        average_surprise = sum(float(m.get("surprise", 0.0)) for m in metadatas) / divisor

        consolidated_metadata = {
            "emotion": metadatas[-1].get("emotion", "") if metadatas else "",
            "arousal": float(average_arousal),
            "valence": float(average_valence),
            "surprise": float(average_surprise),
            "kind": "consolidated",
            "time": timestamp,
        }

        consolidated_id = f"mem_consolidated_{timestamp}"
        return consolidated_document, consolidated_metadata, consolidated_id

    def _sample_document(self, collection: Any) -> str:
        collection_count = self._chroma_call(collection.count, fallback=0)
        if collection_count == 0:
            return ""

        offset = random.randint(0, collection_count - 1)
        result = self._chroma_call(
            lambda: collection.get(limit=1, offset=offset, include=["documents"]),
            fallback={},
        )
        documents = result.get("documents") or []
        return documents[0] if documents else ""

    def _recent_collection_items(self, collection: Any, limit: int, label: str) -> list[tuple[str, str]]:
        collection_count = self._chroma_call(collection.count, fallback=0)
        if collection_count == 0:
            return []

        scan_limit = min(collection_count, max(limit * 4, limit), self.RECENT_MEMORY_SCAN_LIMIT)
        offset = max(collection_count - scan_limit, 0)
        data = self._chroma_call(
            lambda: collection.get(limit=scan_limit, offset=offset, include=["documents", "metadatas"]),
            fallback={},
        )
        documents = data.get("documents") or []
        metadatas = data.get("metadatas") or []
        items = []
        for document, metadata in zip(documents, metadatas):
            if self._is_empty_memory(document):
                continue
            timestamp = metadata.get("time", "unknown") if metadata else "unknown"
            arousal = float(metadata.get("arousal", 0.0)) if metadata else 0.0
            items.append((timestamp, f"{timestamp} [{label}] ARO={arousal:.2f} {document[:400]}"))
        return items

    def _embed_text(self, text: str) -> list[float]:
        return self.encoder.encode(text).tolist()

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
    def _weighted_choice(weighted_documents: list[tuple[str, float]]) -> str:
        total_weight = sum(weight for _, weight in weighted_documents)
        threshold = random.uniform(0.0, total_weight)
        cumulative_weight = 0.0
        for document, weight in weighted_documents:
            cumulative_weight += weight
            if cumulative_weight >= threshold:
                return document
        return weighted_documents[-1][0]

    @staticmethod
    def _is_empty_memory(content: str) -> bool:
        normalized = content.strip().lower()
        if not normalized:
            return True
        empty_markers = ("context: n/a", "spoke: n/a", "ctx:none", "ctx: none")
        return any(marker in normalized for marker in empty_markers)
