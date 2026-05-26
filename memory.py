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

    FORGET_THRESHOLD = 0.15
    ARCHIVE_MARGIN = 0.10

    MAX_FORGET_THRESHOLD = 0.25
    MAX_ARCHIVE_THRESHOLD = 0.45
    MIN_ARCHIVE_MARGIN = 0.04
    ADAPTIVE_FORGET_GAIN = 0.35
    LOW_VALENCE_KEEP_THRESHOLD = -0.25

    # 인지 모방 : 위협, 강한 각인, 새로 배운 것 등에 대한 고민을 반영한 상수들
    KIND_RETENTION_BONUS = {
        "fact": 0.35,
        "threat": 0.30,
        "consolidated": 0.20,
        "reward": 0.16,
        "surprise": 0.12,
        "diary": 0.12,
        "episode": 0.00,
    }

    KIND_CONSOLIDATION_BONUS = {
        "fact": 0.80,
        "threat": 0.70,
        "consolidated": 0.10,
        "reward": 0.35,
        "surprise": 0.50,
        "diary": 0.45,
        "episode": 0.20,
    }

    RESTRUCTURE_BATCH_SIZE = 300
    RESTRUCTURE_FETCH_MULTIPLIER = 3

    TRAUMA_THRESHOLD = 0.6
    LEARNING_SURPRISE_THRESHOLD = 0.3
    HOT_RESULTS_PER_QUERY = 2
    COLD_RESULTS_PER_QUERY = 2
    FLASHBACK_SAMPLE_ATTEMPTS = 10
    RECENT_MEMORY_SCAN_LIMIT = 300
    CONSOLIDATED_PREVIEW_CHARS = 2_000
    MAX_MEMORY_CONTENT_CHARS = 3_000
    CHROMA_RETRY_COUNT = 3
    CHROMA_RETRY_BASE_DELAY = 0.5

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

    def retrieve_memory(self, query: str) -> list[str]:
        clean_query = query.strip()
        if not clean_query:
            return []

        query_embedding = self._embed_text(clean_query)
        retrieved_memories: list[str] = []
        retrieved_memories.extend(
            self._query_collection(self.hot_storage, query_embedding, self.HOT_RESULTS_PER_QUERY, "VIVID")
        )
        retrieved_memories.extend(
            self._query_collection(self.cold_storage, query_embedding, self.COLD_RESULTS_PER_QUERY, "DISTANT")
        )

        return retrieved_memories

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
            weight = arousal + max(0.0, -valence)
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

    def restructure_hierarchical_memory(self, cortex: Any) -> int:
        hot_count = self._chroma_call(self.hot_storage.count, fallback=0)
        if hot_count <= 0:
            return 0

        # 인지 모방 : 평소 대비 많은 일이나 정보 과부하가 있었으면, 망각도 더 많이.
        # 즉, 단기(HOT) 기억이 일반적 망각 배치 크기를 초과할 때만 정리 압력이 증가합니다.
        # cold 개수와의 비율을 쓰지 않는 이유: 초기 cold=0 상태에서 망각 문턱값이 폭주하기 때문입니다.
        # 장기기억이 많다고 해서 더 많이 잊지 않는 것과 비슷. (TODO : 장기 기억도 시간이 지나면 흐려지는데, 아직 이런 구조는 아님)
        backlog_pressure = max(
            0.0,
            (hot_count - self.RESTRUCTURE_BATCH_SIZE) / max(1, self.RESTRUCTURE_BATCH_SIZE),
        )

        adaptive_multiplier = 1.0 + self.ADAPTIVE_FORGET_GAIN * math.log1p(backlog_pressure)

        dynamic_forget_threshold = min(
            self.MAX_FORGET_THRESHOLD,
            self.FORGET_THRESHOLD * adaptive_multiplier,
        )

        dynamic_margin = max(
            self.MIN_ARCHIVE_MARGIN,
            self.ARCHIVE_MARGIN / adaptive_multiplier,
        )

        fetch_limit = min(
            max(self.RESTRUCTURE_BATCH_SIZE, hot_count),
            self.RESTRUCTURE_BATCH_SIZE * self.RESTRUCTURE_FETCH_MULTIPLIER,
        )

        # 수면 중 큐 앞쪽에만 검사가 몰리는 병목 방지. 무작위 구간을 찔러서 스캔함.
        scan_offset = random.randint(0, max(0, hot_count - fetch_limit))

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
            return 0

        arousal_values = [float(metadata.get("arousal", 0.0)) for metadata in metadatas]
        average_arousal = sum(arousal_values) / max(1, len(arousal_values))

        # 인지 부하 비중에 따라 망각 문턱값과 마진을 동적으로 스케일링 (생물학적 간섭 메커니즘 모사)
        archive_threshold = min(
            self.MAX_ARCHIVE_THRESHOLD,
            max(dynamic_forget_threshold + 0.03, average_arousal - dynamic_margin),
        )

        ids_to_delete: list[str] = []
        archive_candidates: list[tuple[str, str, dict]] = []

        for doc_id, document, metadata in zip(ids, documents, metadatas):
            kind = str(metadata.get("kind", "episode")).lower()
            arousal = float(metadata.get("arousal", 0.0))
            surprise = float(metadata.get("surprise", 0.0))
            valence = float(metadata.get("valence", 0.0))

            if self._is_empty_memory(document):
                ids_to_delete.append(doc_id)
                continue

            retention_bonus = self.KIND_RETENTION_BONUS.get(kind, 0.0)
            effective_arousal = arousal + retention_bonus

            should_delete = (
                    effective_arousal < dynamic_forget_threshold
                    and surprise < self.LEARNING_SURPRISE_THRESHOLD
                    and valence > self.LOW_VALENCE_KEEP_THRESHOLD
            )

            if should_delete:
                ids_to_delete.append(doc_id)
                continue

            should_archive = effective_arousal < archive_threshold

            if should_archive:
                archive_candidates.append((doc_id, document, metadata))

        if ids_to_delete:
            self._chroma_call(lambda: self.hot_storage.delete(ids=ids_to_delete))

        if archive_candidates:
            self._archive_hot_memories(archive_candidates, cortex)

        return len(ids_to_delete) + len(archive_candidates)

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
        except Exception:
            return "sometime ago"

    def _query_collection(
        self,
        collection: Any,
        query_embedding: list[float],
        result_count: int,
        label: str,
    ) -> list[str]:
        collection_count = self._chroma_call(collection.count, fallback=0)
        if collection_count == 0:
            return []

        query_results = self._chroma_call(
            lambda: collection.query(
                query_embeddings=[query_embedding],
                n_results=min(result_count, collection_count),
            ),
            fallback={},
        )
        documents = query_results.get("documents") or [[]]
        metadatas = query_results.get("metadatas") or [[]]

        memories = []
        for document, metadata in zip(documents[0], metadatas[0]):
            if self._is_empty_memory(document):
                continue
            timestamp = metadata.get("time", "unknown") if metadata else "unknown"
            kind = metadata.get("kind", "episode") if metadata else "episode"
            arousal = float(metadata.get("arousal", 0.0)) if metadata else 0.0

            time_ago = self._parse_time_ago(timestamp)
            importance = "CRITICAL" if arousal > 0.6 else "SIGNIFICANT" if arousal > 0.3 else "NORMAL"

            memories.append(
                f"[{label}/{kind}: {timestamp}] (Recency: {time_ago}, Importance: {importance}, Arousal: {arousal:.2f}) {document}"
            )
        return memories

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

        ids = [candidate[0] for candidate in archive_candidates]
        documents = [candidate[1] for candidate in archive_candidates]
        metadatas = [candidate[2] for candidate in archive_candidates]

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
            return

        consolidated_document, consolidated_metadata, consolidated_id = self._consolidate_memories(
            documents,
            metadatas,
            cortex,
        )

        # 여러 이유로 압축 실패 시, 원본 HOT을 지우지 않고 방치하여 다음 수면에 재시도.
        if not consolidated_document or not consolidated_metadata or not consolidated_id:
            sample_doc = documents[0][:200] if documents and documents[0] else "No documents"
            print(f"[System / ERROR] Memory consolidation failed. "
                  f"Preserved {len(ids)} hot memories. (Sample: {sample_doc}...)")
            return

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
