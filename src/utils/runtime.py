from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Sequence

try:
    from .video_utils import extract_frames
except ImportError:
    from video_utils import extract_frames

if TYPE_CHECKING:
    try:
        from .analysis import AnalysisManager
        from .retriever import RetrievalManager
    except ImportError:
        from analysis import AnalysisManager
        from retriever import RetrievalManager


CHOICE_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

SAFE_BUILTINS: dict[str, Any] = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "Exception": Exception,
    "filter": filter,
    "float": float,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "print": print,
    "range": range,
    "reversed": reversed,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}


@dataclass
class StructuredAnswer:
    answer: str = ""
    confidence: float = 0.0
    raw_output: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "confidence": self.confidence,
            "raw_output": self.raw_output,
            "metadata": self.metadata,
        }


def _clip_confidence(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(numeric) or math.isinf(numeric):
        return 0.0
    return min(1.0, max(0.0, numeric))


def _first_non_empty_text(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _extract_answer_letter(text: str, choices: Sequence[str] | None = None) -> str:
    if not text:
        return ""

    stripped = text.strip()
    match = re.match(r"^(?:Answer\s*[:\-]\s*)?([A-Z])\b", stripped, flags=re.IGNORECASE)
    if match:
        letter = match.group(1).upper()
        if letter in CHOICE_LETTERS:
            return letter

    lowered = stripped.lower()
    if lowered in {"yes", "no"}:
        return lowered.title()

    if choices:
        normalized_choices = [choice.strip() for choice in choices]
        for index, choice in enumerate(normalized_choices):
            if choice and choice.lower() in lowered:
                return CHOICE_LETTERS[index]

    return ""


def normalize_model_result(
    result: Any,
    choices: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    base_metadata = dict(metadata or {})

    if isinstance(result, StructuredAnswer):
        payload = result.to_dict()
        payload["metadata"] = {**base_metadata, **payload.get("metadata", {})}
        payload["confidence"] = _clip_confidence(payload.get("confidence"))
        payload["answer"] = _extract_answer_letter(payload.get("answer", ""), choices) or payload.get("answer", "")
        return payload

    if isinstance(result, Mapping):
        raw_output = _first_non_empty_text(
            result.get("raw_output"),
            result.get("output_text"),
            result.get("text"),
            result.get("result"),
            result.get("answer"),
        )
        answer = _extract_answer_letter(str(result.get("answer", "")), choices)
        if not answer:
            answer = _extract_answer_letter(raw_output, choices)
        if not answer and not choices and raw_output:
            answer = raw_output
        return {
            "answer": answer,
            "confidence": _clip_confidence(result.get("confidence", result.get("score", 0.0))),
            "raw_output": raw_output,
            "metadata": {
                **base_metadata,
                **dict(result.get("metadata", {})),
            },
        }

    if isinstance(result, (tuple, list)):
        values = list(result)
        if not values:
            return StructuredAnswer(metadata=base_metadata).to_dict()

        if len(values) == 1:
            raw_output = _first_non_empty_text(values[0])
            answer = _extract_answer_letter(raw_output, choices)
            if not answer and not choices:
                answer = raw_output
            return StructuredAnswer(
                answer=answer,
                confidence=0.0,
                raw_output=raw_output,
                metadata=base_metadata,
            ).to_dict()

        if len(values) >= 2:
            first, second = values[0], values[1]
            third = values[2] if len(values) >= 3 else ""
            raw_output = _first_non_empty_text(third, first)
            answer = _extract_answer_letter(_first_non_empty_text(first, raw_output), choices)
            if not answer:
                answer = _extract_answer_letter(raw_output, choices)
            if not answer and not choices and raw_output:
                answer = raw_output
            return StructuredAnswer(
                answer=answer,
                confidence=_clip_confidence(second),
                raw_output=raw_output,
                metadata=base_metadata,
            ).to_dict()

    raw_output = _first_non_empty_text(result)
    answer = _extract_answer_letter(raw_output, choices)
    if not answer and not choices and raw_output:
        answer = raw_output
    return StructuredAnswer(
        answer=answer,
        confidence=0.0,
        raw_output=raw_output,
        metadata=base_metadata,
    ).to_dict()


def make_result(
    answer: str = "",
    confidence: float = 0.0,
    raw_output: str = "",
    **metadata: Any,
) -> dict[str, Any]:
    return StructuredAnswer(
        answer=answer,
        confidence=confidence,
        raw_output=raw_output or answer,
        metadata=metadata,
    ).to_dict()


class VideoRuntimeAPI:
    def __init__(
        self,
        clip_save_folder: str,
        clip_duration: int = 10,
        dataset_folder: str = "dataset",
        current_video_path: str = "",
        current_clip_dir: str = "",
        current_clip_paths: Sequence[str] | None = None,
    ):
        self.clip_save_folder = clip_save_folder
        self.clip_duration = clip_duration
        self.current_video_path = current_video_path
        self.current_clip_dir = current_clip_dir
        self.current_clip_paths = list(current_clip_paths or [])
        self.dataset_folder = dataset_folder
        self._retrieval: "RetrievalManager | None" = None
        self._analysis: "AnalysisManager | None" = None

    def _create_retrieval(self):
        try:
            from .retriever import RetrievalManager
        except ImportError:
            from retriever import RetrievalManager

        retrieval = RetrievalManager(
            clip_save_folder=self.clip_save_folder,
            clip_duration=self.clip_duration,
            dataset_folder=self.dataset_folder,
        )
        retrieval.load_model_to_gpu(0)
        return retrieval

    def _create_analysis(self):
        try:
            from .analysis import AnalysisManager
        except ImportError:
            from analysis import AnalysisManager

        return AnalysisManager(retrieval=self._retrieval)

    @property
    def retrieval(self):
        if self._retrieval is None:
            self._retrieval = self._create_retrieval()
            if self._analysis is not None:
                self._analysis.retrieval = self._retrieval
        return self._retrieval

    @property
    def analysis(self):
        if self._analysis is None:
            self._analysis = self._create_analysis()
        return self._analysis

    def _normalize_clip_result(self, payload: Any) -> tuple[list[tuple[float, float]], list[str]]:
        intervals: list[tuple[float, float]] = []
        clip_paths: list[str] = []

        if isinstance(payload, Mapping):
            raw_intervals = payload.get("intervals") or payload.get("time_ranges") or []
            raw_clip_paths = payload.get("clip_paths") or payload.get("clips") or payload.get("paths") or []
        elif isinstance(payload, tuple):
            raw_intervals = payload[0] if len(payload) >= 1 else []
            raw_clip_paths = payload[1] if len(payload) >= 2 else []
        elif isinstance(payload, list):
            raw_intervals = []
            raw_clip_paths = payload
        else:
            raw_intervals = []
            raw_clip_paths = []

        for item in raw_intervals or []:
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                try:
                    intervals.append((float(item[0]), float(item[1])))
                except (TypeError, ValueError):
                    continue

        for path in raw_clip_paths or []:
            if path:
                clip_paths.append(str(path))

        return intervals, clip_paths

    def _call_with_fallbacks(self, func: Callable[..., Any], calls: Iterable[Callable[[], Any]]) -> Any:
        errors: list[Exception] = []
        for call in calls:
            try:
                return call()
            except TypeError as exc:
                errors.append(exc)
        if errors:
            raise errors[-1]
        return func()

    def query_native(self, video_path: str, question: str, choices: Sequence[str]) -> dict[str, Any]:
        result = self.analysis.query_native(video_path, question, list(choices))
        return normalize_model_result(result, choices, metadata={"mode": "native"})

    def query_mc(self, frames: Sequence[Any], question: str, choices: Sequence[str]) -> dict[str, Any]:
        result = self.analysis.query_mc(list(frames), question, list(choices))
        return normalize_model_result(result, choices, metadata={"mode": "mc"})

    def query_frames(self, frames: Sequence[Any], question: str) -> dict[str, Any]:
        result = self.analysis.query_frames(list(frames), question)
        return normalize_model_result(result, metadata={"mode": "frames"})

    def query_yn(self, frames: Sequence[Any], question: str) -> dict[str, Any]:
        result = self.analysis.query_yn(list(frames), question)
        return normalize_model_result(result, metadata={"mode": "yes_no"})

    def get_informative_clips(
        self,
        video_path: str,
        query: str,
        top_k: int = 3,
        total_duration: int | float | None = None,
    ) -> tuple[list[tuple[float, float]], list[str]]:
        method = getattr(self.retrieval, "get_informative_clips", None)
        if method is None:
            raise AttributeError("RetrievalManager.get_informative_clips is not available.")

        payload = self._call_with_fallbacks(
            method,
            [
                lambda: method(video_path, query, top_k=top_k, total_duration=total_duration),
                lambda: method(query, video_path, top_k=top_k, total_duration=total_duration),
                lambda: method(video_path=video_path, query=query, top_k=top_k, total_duration=total_duration),
                lambda: method(query=query, video_path=video_path, top_k=top_k, total_duration=total_duration),
                lambda: method(video_path, query, top_k),
                lambda: method(query, video_path, top_k),
            ],
        )
        return self._normalize_clip_result(payload)

    def get_subtitle_hints(
        self,
        video_path: str,
        question: str,
        choices: Sequence[str],
        duration: float,
    ) -> str:
        _ = self.retrieval
        return self.analysis.get_subtitle_hints(video_path, question, list(choices), duration)

    def detect_object(self, frame: Any, text: str) -> list[list[float]]:
        return self.analysis.detect_object(frame, text)

    def crop(self, frame: Any, box: tuple[int, int, int, int]) -> Any:
        return self.analysis.crop(frame, box)

    def crop_left(self, frame: Any) -> Any:
        return self.analysis.crop_left(frame)

    def crop_right(self, frame: Any) -> Any:
        return self.analysis.crop_right(frame)

    def crop_top(self, frame: Any) -> Any:
        return self.analysis.crop_top(frame)

    def crop_bottom(self, frame: Any) -> Any:
        return self.analysis.crop_bottom(frame)

    def crop_left_top(self, frame: Any) -> Any:
        return self.analysis.crop_left_top(frame)

    def crop_right_top(self, frame: Any) -> Any:
        return self.analysis.crop_right_top(frame)

    def crop_left_bottom(self, frame: Any) -> Any:
        return self.analysis.crop_left_bottom(frame)

    def crop_right_bottom(self, frame: Any) -> Any:
        return self.analysis.crop_right_bottom(frame)

    def trim_frames(self, video_path: str, start: float, end: float, num_frames: int = 64) -> list[Any]:
        return self.analysis.trim_frames(video_path, start, end, num_frames=num_frames)

    def trim_around(
        self,
        video_path: str,
        timestamp: float,
        intervals: int = 30,
        num_frames: int = 64,
    ) -> list[Any]:
        return self.analysis.trim_around(video_path, timestamp, intervals=intervals, num_frames=num_frames)

    def trim_before(
        self,
        video_path: str,
        timestamp: float,
        intervals: int = 30,
        num_frames: int = 64,
    ) -> list[Any]:
        return self.analysis.trim_before(video_path, timestamp, intervals=intervals, num_frames=num_frames)

    def trim_after(
        self,
        video_path: str,
        timestamp: float,
        intervals: int = 30,
        num_frames: int = 64,
    ) -> list[Any]:
        return self.analysis.trim_after(video_path, timestamp, intervals=intervals, num_frames=num_frames)

    def extract_frames(self, video_path: str, num_frames: int = 32) -> list[Any]:
        return extract_frames(video_path, num_frames=num_frames)

    def build_globals(self) -> dict[str, Any]:
        return {
            "make_result": make_result,
            "finalize_result": make_result,
            "normalize_model_result": normalize_model_result,
            "query_native": self.query_native,
            "query_mc": self.query_mc,
            "query_frames": self.query_frames,
            "query_yn": self.query_yn,
            "get_informative_clips": self.get_informative_clips,
            "get_subtitle_hints": self.get_subtitle_hints,
            "detect_object": self.detect_object,
            "extract_frames": self.extract_frames,
            "trim_frames": self.trim_frames,
            "trim_around": self.trim_around,
            "trim_before": self.trim_before,
            "trim_after": self.trim_after,
            "crop": self.crop,
            "crop_left": self.crop_left,
            "crop_right": self.crop_right,
            "crop_top": self.crop_top,
            "crop_bottom": self.crop_bottom,
            "crop_left_top": self.crop_left_top,
            "crop_right_top": self.crop_right_top,
            "crop_left_bottom": self.crop_left_bottom,
            "crop_right_bottom": self.crop_right_bottom,
            "current_video_path": self.current_video_path,
            "current_clip_dir": self.current_clip_dir,
            "current_clip_paths": self.current_clip_paths,
        }


__all__ = [
    "SAFE_BUILTINS",
    "StructuredAnswer",
    "VideoRuntimeAPI",
    "make_result",
    "normalize_model_result",
]
