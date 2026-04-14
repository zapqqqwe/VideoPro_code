import math
import os
import re
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
import requests
import torch
from openai import OpenAI
from PIL import Image
from moviepy import VideoFileClip
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from video_utils import (
    extract_frames,
    get_oai_chat_response,
    get_oai_chat_response_qwen3,
    get_subtitles_in_range,
)


VALID_SHORT_ANSWERS = {
    "A", "B", "C", "D", "E",
    "F", "G", "H", "I", "J",
    "Yes", "No",
}


DEFAULT_API_BASE = os.getenv("VIDEOPRO_API_BASE", "http://0.0.0.0:8007/v1")
DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")
DEFAULT_MODEL_NAME = os.getenv("VIDEOPRO_MODEL", "qwen3vl")


def build_messages_with_local_jpg(
    frames: List[Image.Image],
    question: str,
    sample_k: int = 64,
    max_side: int = 500,
    max_pixels: int = 500 * 300,
    min_side: int = 224,
    images_root: str = "images",
    use_file_url: bool = False,
    jpeg_quality: int = 95,
) -> List[dict]:
    """
    Build multimodal messages by saving frames to local JPEG files.

    Steps:
    1) Compute a unified base_size under size/pixel limits.
    2) Resize all frames to the same size (stretch resize, keep old behavior).
    3) Save them to ./images/<run_id>/*.jpg
    4) Uniformly sample up to sample_k frames.
    5) Build OpenAI-style messages.

    Returns:
        messages: List[dict]
    """
    if not frames:
        return [
            {
                "role": "user",
                "content": [{"type": "text", "text": question}],
            }
        ]

    run_id = uuid.uuid4().hex[:12]
    out_dir = Path(images_root) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    max_w = max(frame.size[0] for frame in frames)
    max_h = max(frame.size[1] for frame in frames)

    max_w = min(max_w, max_side)
    max_h = min(max_h, 300)

    pixels = max_w * max_h
    if pixels > max_pixels:
        scale = math.sqrt(max_pixels / float(pixels))
        max_w = max(1, int(max_w * scale))
        max_h = max(1, int(max_h * scale))

    max_w = max(max_w, min_side)
    max_h = max(max_h, min_side)
    base_size = (max_w, max_h)

    saved_paths: List[Path] = []
    for idx, frame in enumerate(frames):
        frame = frame.convert("RGB")
        if frame.size != base_size:
            frame = frame.resize(base_size, Image.LANCZOS)

        jpg_path = out_dir / f"{idx:06d}.jpg"
        frame.save(jpg_path, format="JPEG", quality=jpeg_quality, optimize=True)
        saved_paths.append(jpg_path)

    selected_paths = saved_paths
    n = len(selected_paths)
    if sample_k is not None and sample_k > 0 and n > sample_k:
        step = n / float(sample_k)
        indices = [min(n - 1, int(i * step)) for i in range(sample_k)]
        indices = sorted(set(indices))
        selected_paths = [selected_paths[i] for i in indices]

    content: List[dict] = []
    for path in selected_paths:
        abs_path = str(path.resolve())
        image_path = f"file://{abs_path}" if use_file_url else abs_path
        content.append(
            {
                "type": "image",
                "image": image_path,
            }
        )

    content.append(
        {
            "type": "text",
            "text": f"Based on the frames, {question}",
        }
    )

    return [{"role": "user", "content": content}]


def _to_image(obj: Any) -> Optional[Image.Image]:
    """
    Convert supported input into PIL.Image in RGB mode.

    Supported:
    - PIL.Image
    - str / Path
    - np.ndarray (OpenCV BGR HWC)
    - bytes / bytearray
    """
    img: Image.Image

    if isinstance(obj, Image.Image):
        img = obj
    elif isinstance(obj, (str, Path)):
        img = Image.open(obj)
    elif isinstance(obj, np.ndarray):
        if obj.ndim != 3 or obj.shape[2] != 3:
            return None
        img = Image.fromarray(cv2.cvtColor(obj, cv2.COLOR_BGR2RGB))
    elif isinstance(obj, (bytes, bytearray)):
        img = Image.open(BytesIO(obj))
    else:
        return None

    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def _extract_short_answer_and_confidence(chat_response) -> Tuple[str, float, str]:
    output_text = chat_response.choices[0].message.content or ""

    confidence = 0.0
    for choice in chat_response.choices:
        if hasattr(choice, "logprobs") and choice.logprobs and choice.logprobs.content:
            first_token_info = choice.logprobs.content[0]
            confidence = float(np.exp(first_token_info.logprob))
            break

    match = re.match(r"^\s*([A-J]|Yes|No)\b", output_text.strip())
    final_answer = match.group(1) if match else ""
    if final_answer not in VALID_SHORT_ANSWERS:
        final_answer = ""

    score = confidence if final_answer else 0.0
    return final_answer, score, output_text


class AnalysisManager:
    def __init__(self, device_qwen: str = "0", retrieval=None):
        self.device_track = "cuda:0"
        self.device_qwen = device_qwen
        self.dtype = torch.float16
        self.retrieval = retrieval

        self.llm = OpenAI(
            api_key=DEFAULT_API_KEY,
            base_url=DEFAULT_API_BASE,
        )
        self.model_name = DEFAULT_MODEL_NAME

        self.gdino_cfg_path = (
            "models/Grounded-SAM-2/grounding_dino/groundingdino/config/"
            "GroundingDINO_SwinT_OGC.py"
        )
        self.gdino_ckpt_path = (
            "models/Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth"
        )

        self.sam2_cfg_path = (
            "models/Grounded-SAM-2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
        )
        self.sam2_ckpt_path = (
            "models/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"
        )

        self.gdino_proc = None
        self.gdino_model = None
        self.sam2_video_predictor = None

    def run_ocr(self, frame: Any) -> str:
        """
        Placeholder OCR method.
        Keep interface for compatibility.
        """
        _ = _to_image(frame)
        return ""

    def detect_object(
        self,
        frame: Any,
        text: str,
        box_threshold: float = 0.5,
        text_threshold: float = 0.25,
    ) -> List[List[float]]:
        """
        Detect objects with Grounding DINO and return xyxy boxes.
        """
        if not text or not text.strip():
            return []

        caption = text.strip().lower()
        if not caption.endswith("."):
            caption += "."

        if getattr(self, "gdino_model", None) is None:
            model_id = "models/grounding-dino-base"
            self.gdino_proc = AutoProcessor.from_pretrained(model_id)
            self.gdino_model = (
                AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to("cuda")
            )

        try:
            img = _to_image(frame)
        except Exception as exc:
            print(f"[detect_object] failed to convert frame to image: {exc}")
            return []

        if img is None:
            return []

        try:
            inputs = self.gdino_proc(
                images=img,
                text=caption,
                return_tensors="pt",
            ).to("cuda")

            with torch.no_grad():
                outputs = self.gdino_model(**inputs)

            processed = self.gdino_proc.post_process_grounded_object_detection(
                outputs=outputs,
                input_ids=inputs.input_ids,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[img.size[::-1]],
            )[0]

            boxes = processed.get("boxes")
            if boxes is None or len(boxes) == 0:
                return []

            results: List[List[float]] = []
            for box in boxes:
                x1, y1, x2, y2 = box.tolist()
                if x2 > x1 and y2 > y1:
                    results.append([float(x1), float(y1), float(x2), float(y2)])
            return results
        except Exception as exc:
            print(f"[detect_object] inference failed: {exc}")
            return []

    def crop(self, frame: Any, box: Tuple[int, int, int, int]) -> Any:
        img = _to_image(frame)
        if img is None:
            return frame
        return img.crop(box)

    def crop_left(self, frame: Any) -> Any:
        img = _to_image(frame)
        if img is None:
            return frame
        w, h = img.size
        return img.crop((0, 0, w // 2, h))

    def crop_right(self, frame: Any) -> Any:
        img = _to_image(frame)
        if img is None:
            return frame
        w, h = img.size
        return img.crop((w // 2, 0, w, h))

    def crop_top(self, frame: Any) -> Any:
        img = _to_image(frame)
        if img is None:
            return frame
        w, h = img.size
        return img.crop((0, 0, w, h // 2))

    def crop_bottom(self, frame: Any) -> Any:
        img = _to_image(frame)
        if img is None:
            return frame
        w, h = img.size
        return img.crop((0, h // 2, w, h))

    def crop_left_top(self, frame: Any) -> Any:
        img = _to_image(frame)
        if img is None:
            return frame
        w, h = img.size
        return img.crop((0, 0, w // 2, h // 2))

    def crop_right_top(self, frame: Any) -> Any:
        img = _to_image(frame)
        if img is None:
            return frame
        w, h = img.size
        return img.crop((w // 2, 0, w, h // 2))

    def crop_left_bottom(self, frame: Any) -> Any:
        img = _to_image(frame)
        if img is None:
            return frame
        w, h = img.size
        return img.crop((0, h // 2, w // 2, h))

    def crop_right_bottom(self, frame: Any) -> Any:
        img = _to_image(frame)
        if img is None:
            return frame
        w, h = img.size
        return img.crop((w // 2, h // 2, w, h))

    def query_video(self, frames: List[Any], question: str):
        return self.query_frames(frames, question)

    def query_mc(self, frames: List[Any], query: str, choices: List[str]):
        letters = [chr(65 + i) for i in range(len(choices))]
        choices_str = "\n".join(
            f"{letter}. {choice}" for letter, choice in zip(letters, choices)
        )
        prompt = (
            f"Question: {query}\n"
            f"Choices:\n{choices_str}\n"
            f"You must just output a single letter. Best option:"
        )

        final_answer, score, output_text = self.query_frames(frames, prompt)
        return output_text, score

    def query_native(
        self,
        video_path: str,
        query: str,
        choices: List[str],
        num_frames: int = 64,
        threshold: float = 0.75,
    ) -> Tuple[str, float]:
        _ = num_frames
        _ = threshold

        letters = [chr(65 + i) for i in range(len(choices))]
        choices_str = "\n".join(
            f"{letter}. {choice}" for letter, choice in zip(letters, choices)
        )
        prompt = (
            "Select the best answer to the following multiple-choice question "
            "based on the video. Respond with only the letter of the correct option.\n"
            f"Question: {query}\n"
            f"Possible answer choices:\n{choices_str}\n"
            "Output a single letter. The best answer is: "
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        chat_response = self.llm.chat.completions.create(
            model=self.model_name,
            messages=messages,
            n=1,
            temperature=0,
            max_tokens=20000,
            logprobs=True,
            top_logprobs=5,
        )

        final_answer, score, _ = _extract_short_answer_and_confidence(chat_response)
        return final_answer, score

    def query_frames(
        self,
        frames: List[Any],
        question: str,
        video_fps: int = 8,
    ) -> Tuple[str, float, str]:
        _ = video_fps

        pil_frames: List[Image.Image] = []
        for frame in frames:
            img = _to_image(frame)
            if img is not None:
                pil_frames.append(img)

        messages = build_messages_with_local_jpg(pil_frames, question)

        chat_response = self.llm.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0,
            top_p=0.01,
            logprobs=True,
            top_logprobs=5,
            max_tokens=20000,
        )

        return _extract_short_answer_and_confidence(chat_response)

    def query_yn(self, frames: List[Any], query: str) -> str:
        prompt = f"{query}\nOutput 'Yes' or 'No': Answer:"
        _, _, output_text = self.query_frames(frames, prompt)
        return output_text

    def get_subtitle_hints(
        self,
        video_path: str,
        question: str,
        choices: List[str],
        duration: float,
        word_number: int = 300,
    ) -> str:
        _ = word_number
        if self.retrieval is None:
            raise ValueError("Retrieval manager is required for get_subtitle_hints.")

        subtitles = get_subtitles_in_range(video_path, (0, duration))

        letters = [chr(65 + i) for i in range(len(choices))]
        choices_str = "\n".join(
            f"{letter}. {choice}" for letter, choice in zip(letters, choices)
        )
        question_with_choices = f"Question: {question}\nChoices:\n{choices_str}"
    
        intervals = self.retrieval.get_informative_subtitles(
            video_path,
            question_with_choices,
            top_k=50,
        )
        retrieval_subtitles = get_subtitles_in_range(video_path, intervals)
        prompt = (
            f"{question_with_choices}\n\n"
            "### Please summarize the relevant information from the following "
            "video subtitles that could help answer the above question. "
            "The summary must be clear and accurate. ###\n\n"
            f"The retrieval subtitles start:\n{retrieval_subtitles}\n"
            "Subtitles end.\n"
            "The important summary tip of the question: "
        )
        response = get_oai_chat_response_qwen3(prompt)
          

        if "</think>" in response:
            return response.split("</think>")[-1].strip()
        return response.strip()

    def _generate_trim_path(self, original_path: str, tag: str) -> str:
        base_dir = os.path.dirname(original_path)
        filename = os.path.splitext(os.path.basename(original_path))[0]
        new_filename = f"{filename}_trim-{tag}.mp4"
        return os.path.join(base_dir, new_filename)

    def trim_frames(
        self,
        video_path: str,
        start: float,
        end: float,
        num_frames: int = 64,
    ):
        start = max(0.0, start)
        end = max(start, end)

        out_path = self._generate_trim_path(
            video_path,
            f"between-{int(start)}-{int(end)}",
        )

        if not os.path.exists(out_path):
            try:
                with VideoFileClip(video_path) as clip:
                    subclip = clip.subclipped(start, end)
                    subclip.write_videofile(out_path, codec="libx264", audio_codec="aac")
            except Exception:
                return extract_frames(video_path, num_frames)

        return extract_frames(out_path, num_frames)

    def trim_around(
        self,
        video_path: str,
        timestamp: float,
        intervals: int = 30,
        num_frames: int = 64,
    ):
        half = intervals / 2
        start = max(0.0, timestamp - half)

        with VideoFileClip(video_path) as clip:
            video_duration = clip.duration

        end = min(timestamp + half, max(0.0, video_duration - 1))
        out_path = self._generate_trim_path(video_path, f"around-{int(timestamp)}")

        if not os.path.exists(out_path):
            with VideoFileClip(video_path) as clip:
                subclip = clip.subclipped(start, end)
                subclip.write_videofile(out_path, codec="libx264", audio_codec="aac")

        return extract_frames(out_path, num_frames)

    def trim_before(
        self,
        video_path: str,
        timestamp: float,
        intervals: int = 30,
        num_frames: int = 64,
    ):
        start = max(0.0, timestamp - intervals)
        end = max(start, timestamp)

        out_path = self._generate_trim_path(video_path, f"before-{int(timestamp)}")
        if not os.path.exists(out_path):
            with VideoFileClip(video_path) as clip:
                subclip = clip.subclipped(start, end)
                subclip.write_videofile(out_path, codec="libx264", audio_codec="aac")

        return extract_frames(out_path, num_frames)

    def trim_after(
        self,
        video_path: str,
        timestamp: float,
        intervals: int = 30,
        num_frames: int = 64,
    ):
        with VideoFileClip(video_path) as clip:
            video_duration = clip.duration

        end = min(timestamp + intervals, max(0.0, video_duration - 1))
        out_path = self._generate_trim_path(video_path, f"after-{int(timestamp)}")

        if not os.path.exists(out_path):
            with VideoFileClip(video_path) as clip:
                subclip = clip.subclipped(timestamp, end)
                subclip.write_videofile(out_path, codec="libx264", audio_codec="aac")

        return extract_frames(out_path, num_frames)
