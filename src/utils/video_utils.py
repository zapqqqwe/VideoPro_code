import base64
import json
import math
import os
import re
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from decord import VideoReader, cpu
from moviepy.video.io.VideoFileClip import VideoFileClip
from openai import OpenAI
from PIL import Image
from scenedetect import SceneManager, VideoManager
from scenedetect.detectors import ContentDetector


MAX_DS_ROUND = 20
MAX_FRAMES = 32
MIN_FRAMES = 10
MAX_IMAGE_RESOLUTION = 748
EPS = 1


client1 = OpenAI(api_key="", base_url="http://0.0.0.0:8007/v1")


def get_oai_chat_response(prompt: str, model: str = "qwen3vl") -> str:
    """General-purpose text-only chat response."""
    response = client1.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    result = response.choices[0].message.content if response.choices else ""
    return result


def get_oai_chat_response_qwen3(prompt: str) -> str:
    print(prompt)
    response = client1.chat.completions.create(
        model="qwen3vl",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    results = [choice.message.content for choice in response.choices]
    print("summary", results[0])
    return results[0]


def _safe_open_image(path: str) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            img.load()
        return True
    except Exception:
        return False


def _sample_list(values: List[Any], target_num: int) -> List[Any]:
    if not values:
        return values

    if len(values) <= target_num:
        return values

    interval = len(values) / target_num
    return [values[int(i * interval)] for i in range(target_num)]


def _resize_keep_aspect(frame: np.ndarray, max_dimension: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if height > width:
        new_height = max_dimension
        new_width = int(width * (max_dimension / height))
    else:
        new_width = max_dimension
        new_height = int(height * (max_dimension / width))
    return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)


def timestamp_to_clip_path(
    dataset_folder: str,
    begin_time_stamp: float,
    end_time_stamp: float,
    video_path: str,
    fps: int = 2,
) -> Tuple[List[str], List[float]]:
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    frame_folder = os.path.join(dataset_folder, f"dense_frames/{video_id}")
    os.makedirs(frame_folder, exist_ok=True)

    if (end_time_stamp - begin_time_stamp) < 1:
        begin_time_stamp = max(begin_time_stamp - 0.5, 0)
        end_time_stamp += 0.5

    num_frames = int((end_time_stamp - begin_time_stamp) * fps)
    time_points = [begin_time_stamp + i * (1.0 / fps) for i in range(num_frames)]

    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    for t in time_points:
        frame_idx = int(round(t * video_fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = cap.read()
        if success:
            frame_name = f"frame_{t:.3f}.jpg"
            frame_path = os.path.join(frame_folder, frame_name)
            if not os.path.exists(frame_path):
                cv2.imwrite(frame_path, frame)
    cap.release()

    candidates = [
        file
        for file in os.listdir(frame_folder)
        if file.startswith("frame_") and file.endswith((".png", ".jpg"))
    ]
    if not candidates:
        return [], []

    frame_paths = []
    timestamps = []

    for time_point in time_points:
        closest_frame = min(
            candidates,
            key=lambda file: abs(
                float(
                    file.replace("frame_", "")
                    .replace(".png", "")
                    .replace(".jpg", "")
                )
                - time_point
            ),
        )
        frame_path = os.path.join(frame_folder, closest_frame)
        if _safe_open_image(frame_path):
            frame_paths.append(frame_path)
            timestamps.append(time_point)

    if len(frame_paths) > MAX_FRAMES:
        frame_paths = _sample_list(frame_paths, MAX_FRAMES)
        timestamps = _sample_list(timestamps, MAX_FRAMES)

    return frame_paths, timestamps


def clip_number_to_clip_path(
    dataset_folder: str,
    clip_numbers: Sequence[int],
    video_path: str,
    clip_duration: int = 10,
    fps: int = 2,
) -> Tuple[List[str], List[float]]:
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    frame_folder = os.path.join(dataset_folder, "dense_frames", video_id)
    os.makedirs(frame_folder, exist_ok=True)

    frame_list: List[str] = []
    second_list: List[float] = []

    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    for clip_number in clip_numbers:
        begin_time_stamp = clip_number * clip_duration
        end_time_stamp = begin_time_stamp + clip_duration
        time_points = [
            begin_time_stamp + i * (1.0 / fps)
            for i in range(int(clip_duration * fps))
        ]

        for t in time_points:
            frame_idx = int(round(t * video_fps))
            frame_name = f"frame_{t:.3f}.jpg"
            frame_path = os.path.join(frame_folder, frame_name)

            if not os.path.exists(frame_path):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                success, frame = cap.read()
                if success:
                    cv2.imwrite(frame_path, frame)

        candidates = [
            file for file in os.listdir(frame_folder) if file.endswith((".jpg", ".png"))
        ]
        for time_point in time_points:
            try:
                frame_name = min(
                    candidates,
                    key=lambda file: abs(
                        float(
                            file.replace(".png", "")
                            .replace(".jpg", "")
                            .split("_")[1]
                        )
                        - time_point
                    ),
                )
                frame_path = os.path.join(frame_folder, frame_name)

                if _safe_open_image(frame_path):
                    frame_list.append(frame_path)
                    second_list.append(time_point)
            except Exception as exc:
                print(f"Error at time {time_point:.3f}: {exc}")
                continue

    cap.release()

    if len(frame_list) > MAX_FRAMES:
        frame_list = _sample_list(frame_list, MAX_FRAMES)
        second_list = _sample_list(second_list, MAX_FRAMES)

    if 0 < len(frame_list) < MIN_FRAMES:
        frame_list = _sample_list(frame_list, MIN_FRAMES)
        second_list = _sample_list(second_list, MIN_FRAMES)

    if not frame_list:
        raise KeyError(f"Frame list {dataset_folder} {clip_numbers} {video_path} is invalid!")

    return frame_list, second_list


def is_valid_video(path: str) -> bool:
    try:
        cap = cv2.VideoCapture(path)
    except Exception:
        return False

    if not cap.isOpened():
        cap.release()
        return False
    cap.release()

    try:
        VideoReader(path, ctx=cpu(0), num_threads=1)
        return True
    except Exception:
        return False


def parse_subtitle_time(time_str: str) -> float:
    h, m, s_ms = time_str.split(":")
    try:
        s, ms = s_ms.split(",")
    except ValueError:
        s, ms = s_ms.split(".")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def parse_caption_time(time_str: str) -> float:
    h, m, s_ms = time_str.split(":")
    s, ms = s_ms.split(".")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def extract_frames(video_path: str, num_frames: int = 64) -> List[Image.Image]:
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    fps = vr.get_avg_fps()

    if fps <= 0:
        raise ValueError("Invalid FPS for the given video.")

    video_duration = total_frames / fps
    num_frames = max(1, min(num_frames, max(1, int(video_duration))))

    frame_indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
    try:
        frames = vr.get_batch(frame_indices).asnumpy()
    except Exception:
        frames = vr.get_batch(frame_indices).numpy()

    return [Image.fromarray(frame) for frame in frames]


def load_subtitles(video_path: str) -> Dict[Tuple[float, float], str]:
    subtitle_path = video_path.replace("video", "subtitles").replace(".mp4", ".srt")

    if os.path.exists(subtitle_path):
        subtitles = {}
        with open(subtitle_path, "r", encoding="utf-8") as file:
            content = file.read().split("\n\n")
            for section in content:
                if section.strip():
                    lines = section.strip().split("\n")
                    if len(lines) >= 3:
                        time_range = lines[1].split(" --> ")
                        start_time = parse_subtitle_time(time_range[0])
                        end_time = parse_subtitle_time(time_range[1])
                        text = " ".join(line for line in lines[2:])
                        subtitles[(start_time, end_time)] = text
        return subtitles

    subtitle_path = video_path.replace("videos", "subtitles").replace(".mp4", "_en.json")
    with open(subtitle_path, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    subtitles = {}
    for item in data_list:
        start_time = parse_subtitle_time(item["start"])
        end_time = parse_subtitle_time(item["end"])
        subtitles[(start_time, end_time)] = item["line"]
    return subtitles


def load_caption(video_path: str) -> Dict[Tuple[float, float], str]:
    caption_path = video_path.replace("videos", "caption").replace(".mp4", ".srt")

    if os.path.exists(caption_path):
        captions = {}
        with open(caption_path, "r", encoding="utf-8") as file:
            content = file.read().split("\n\n")
            for section in content:
                if section.strip():
                    lines = section.strip().split("\n")
                    if len(lines) >= 3:
                        time_range = lines[1].split(" --> ")
                        start_time = parse_subtitle_time(time_range[0])
                        end_time = parse_subtitle_time(time_range[1])
                        text = " ".join(line for line in lines[2:])
                        captions[(start_time, end_time)] = text
        return captions

    caption_path = video_path.replace("videos", "caption").replace(".mp4", ".json")
    with open(caption_path, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    captions = {}
    for item in data_list:
        start_time = parse_subtitle_time(item["start"])
        end_time = parse_subtitle_time(item["end"])
        captions[(start_time, end_time)] = item["line"]
    return captions


def _strip_font_tags(text: str) -> str:
    pattern = r'<font color="white" size=".72c">(.*?)</font>'
    matched = re.findall(pattern, text)
    return matched[0] if matched else text


def extract_caption(video_path: str) -> List[Tuple[float, float, str]]:
    captions = load_caption(video_path)
    caption_frames = []
    for (start_time, end_time), text in captions.items():
        caption_frames.append((float(start_time), float(end_time), _strip_font_tags(text)))
    return caption_frames


def extract_subtitles(video_path: str) -> List[Tuple[float, float, str]]:
    subtitles = load_subtitles(video_path)
    subtitle_frames = []
    for (start_time, end_time), text in subtitles.items():
        subtitle_frames.append((float(start_time), float(end_time), _strip_font_tags(text)))
    return subtitle_frames


def is_valid_frame(frame_path: str) -> bool:
    return _safe_open_image(frame_path)


def load_image(image_path: str) -> Optional[str]:
    frame = cv2.imread(image_path)
    if frame is None:
        print("Frame", image_path, "not valid!!")
        return None

    try:
        resized_frame = _resize_keep_aspect(frame, MAX_IMAGE_RESOLUTION)
        _, buffer = cv2.imencode(".jpg", resized_frame)
        return base64.b64encode(buffer).decode("utf-8")
    except Exception:
        print("Frame", image_path, "not valid!!")
        return None


def image_paths_to_base64(image_paths: Sequence[str]) -> Union[List[str], bool]:
    base64_frames = []

    for image_path in image_paths:
        image_path = image_path.strip()
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Warning: Could not read image {image_path}")
            return False

        resized_frame = _resize_keep_aspect(frame, 768)
        _, buffer = cv2.imencode(".jpg", resized_frame)
        base64_frames.append(base64.b64encode(buffer).decode("utf-8"))

    return base64_frames


def format_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}h--{minutes:02d}min--{seconds:02d}sec"


def _seconds_to_time_str(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _probe_duration(video_path: str) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    out = subprocess.check_output(cmd).decode().strip()
    dur = float(out)
    if dur < 0:
        dur = 0.0
    return math.floor(dur * 1000) / 1000.0


def split_video_to_clips(
    video_path: str,
    clip_dir: str,
    clip_duration: int = 10,
    workers: int = 24,
    tolerate_missing: int = 2,
    fast_copy: bool = True,
    overwrite: bool = True,
    tolerance: float = 0.5,
) -> Tuple[List[str], float]:
    _ = fast_copy
    _ = tolerance

    os.makedirs(clip_dir, exist_ok=True)
    duration = _probe_duration(video_path)

    expected = math.ceil((duration - EPS) / clip_duration) if duration > EPS else 0

    existing = [
        f for f in os.listdir(clip_dir) if f.endswith(".mp4") and f.startswith("clip_")
    ]
    if len(existing) >= max(0, expected - tolerate_missing):
        def _idx(p: str) -> int:
            try:
                return int(p.split("_")[1])
            except Exception:
                return 10**9

        return [os.path.join(clip_dir, f) for f in sorted(existing, key=_idx)], duration

    def _cut_one(idx: int, start: float, end: float) -> Optional[str]:
        start_str = _seconds_to_time_str(start).replace(":", "-")
        end_str = _seconds_to_time_str(end).replace(":", "-")
        out_path = os.path.join(clip_dir, f"clip_{idx}_{start_str}_to_{end_str}.mp4")

        if end - start <= EPS:
            return None

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            str(start),
            "-i",
            video_path,
            "-t",
            str(max(0.0, end - start)),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-force_key_frames",
            f"expr:gte(t,n_forced*{clip_duration})",
        ]

        if overwrite:
            cmd.append("-y")
        cmd += ["-avoid_negative_ts", "make_zero", out_path]

        subprocess.run(cmd, check=True)
        return out_path

    tasks = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for i in range(expected):
            start = i * clip_duration
            end = min((i + 1) * clip_duration, duration)
            if end - start > EPS:
                tasks.append(executor.submit(_cut_one, i, start, end))

        results = []
        for future in as_completed(tasks):
            path = future.result()
            if path is not None:
                results.append(path)

    cleaned = []
    for path in results:
        try:
            if path and os.path.exists(path) and os.path.getsize(path) > 0:
                cleaned.append(path)
            elif path and os.path.exists(path):
                os.remove(path)
        except Exception:
            pass

    def _idx_full(path: str) -> int:
        try:
            return int(os.path.basename(path).split("_")[1])
        except Exception:
            return 10**9

    return sorted(cleaned, key=_idx_full), duration


def merge_intervals(intervals: List[Tuple[float, float]], min_len: float = 8.0):
    if not intervals:
        return []

    merged = []
    current_start, current_end = intervals[0]

    for start, end in intervals[1:]:
        if (current_end - current_start) < min_len:
            current_end = end
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end

    if (current_end - current_start) < min_len and merged:
        last_start, last_end = merged.pop()
        merged.append((last_start, current_end))
    else:
        merged.append((current_start, current_end))

    return merged


def most_common_string(answers: Sequence[str]) -> Optional[str]:
    most_common = Counter(answers).most_common(1)
    return most_common[0][0] if most_common else None


def get_subtitles_in_range(
    video_path: str,
    time_ranges: Union[Tuple[float, float], List[Tuple[float, float]]],
    time_end: float = -1,
) -> str:
    """
    Get subtitles within given time ranges.
    time_ranges can be:
        - a tuple: (begin, end)
        - a list of tuples: [(begin, end), (begin, end), ...]
    Returns formatted string.
    """
    if time_end != -1:
        time_ranges = [(float(time_ranges), float(time_end))]  # backward compatibility
    elif isinstance(time_ranges, tuple):
        time_ranges = [time_ranges]

    all_subtitle_triples = extract_subtitles(video_path)

    results = []
    for begin_timestamp, end_timestamp in time_ranges:
        cur_subtitle_triples = [
            {
                "start": float(x[0]),
                "end": float(x[1]),
                "subtitle": x[2],
            }
            for x in all_subtitle_triples
            if begin_timestamp <= x[0] <= end_timestamp
        ]

        results.extend(
            [
                f"{item['start']:.1f}-{item['end']:.1f}s: {item['subtitle']}"
                for item in cur_subtitle_triples
                if item["subtitle"]
            ]
        )

    return "\n".join(results)


def get_captions_in_range(
    video_path: str,
    begin_timestamp: int,
    end_timestamp: int,
) -> List[Dict[str, Any]]:
    all_caption_triples = extract_caption(video_path)
    cur_caption_triples = [
        {
            "start": int(x[0]),
            "end": int(x[1]),
            "caption": x[2],
        }
        for x in all_caption_triples
        if begin_timestamp <= x[0] <= end_timestamp
    ]
    return str(cur_caption_triples)


def build_prompt_subtitles(subtitles: List[Dict[str, Any]]) -> str:
    prompt_lines = []
    for item in subtitles:
        start = item["start"]
        end = item["end"]
        subtitle = item["subtitle"]
        prompt_lines.append(f'Video {start}s-{end}s subtitle: "{subtitle}"')
    return "\n".join(prompt_lines)


def build_prompt_caption(captions: List[Dict[str, Any]]) -> str:
    prompt_lines = []
    for item in captions:
        start = item["start"]
        end = item["end"]
        caption = item["caption"]
        prompt_lines.append(f'Video {start}s-{end}s caption: "{caption}"')
    return "\n".join(prompt_lines)


def parse_and_sort_file_paths(file_paths: List[Tuple[str, float]]):
    def time_to_seconds(time_str: str) -> int:
        h, m, s = map(int, time_str.split("-"))
        return h * 3600 + m * 60 + s

    time_intervals = []
    for file_path, score in file_paths:
        match = re.search(
            r"clip_\d+_(\d{2}-\d{2}-\d{2})_to_(\d{2}-\d{2}-\d{2})",
            file_path,
        )
        if match:
            start_time = match.group(1)
            end_time = match.group(2)

            start_seconds = time_to_seconds(start_time)
            end_seconds = time_to_seconds(end_time)

            time_intervals.append((start_seconds, end_seconds, file_path, score))

    time_intervals.sort(key=lambda x: x[0])
    intervals = [(s, e) for s, e, _, _ in time_intervals]
    files = [
        fp.replace("wangjiaqi-24032/lichenglin/", "lichenglin-253208540324/")
        for _, _, fp, _ in time_intervals
    ]
    return intervals, files


def process_code(text: str) -> str:
    text = text.replace("get_informative_clips", "retrieval.get_informative_clips")
    text = text.replace("get_informative_subtitles", "retrieval.get_informative_subtitles")
    text = text.replace("get_informative_captions", "retrieval.get_informative_captions")

    text = text.replace("query_native", "analysis.query_native")
    text = text.replace("query_mc", "analysis.query_mc")
    text = text.replace("query_yn", "analysis.query_yn")
    text = text.replace("query_count", "analysis.query_count")
    text = text.replace("get_subtitle_hints", "analysis.get_subtitle_hints")
    text = text.replace("trim_after", "analysis.trim_after")
    text = text.replace("trim_before", "analysis.trim_before")
    text = text.replace("trim_frames", "analysis.trim_frames")
    text = text.replace("trim_around", "analysis.trim_around")
    text = text.replace("detect_object", "analysis.detect_object")
    text = text.replace("run_ocr", "analysis.run_ocr")
    text = text.replace("crop_left(", "analysis.crop_left(")
    text = text.replace("crop_right(", "analysis.crop_right(")
    text = text.replace("crop_top(", "analysis.crop_top(")
    text = text.replace("crop_bottom(", "analysis.crop_bottom(")
    text = text.replace("crop_left_top(", "analysis.crop_left_top(")
    text = text.replace("crop_right_top(", "analysis.crop_right_top(")
    text = text.replace("crop_left_bottom(", "analysis.crop_left_bottom(")
    text = text.replace("crop_right_bottom(", "analysis.crop_right_bottom(")
    text = text.replace("crop(", "analysis.crop(")
    text = text.replace("crop (", "analysis.crop (")

    return (
        text.replace("query_count", "analysis.query_count")
        .replace("{{", "{")
        .replace("}}", "}")
    )


def normalize(s: str) -> str:
    return re.sub(r"[^a-z ]", "", s.lower())


def get_video_duration(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    if fps <= 0:
        return 0.0
    return frame_count / fps


def process_data(
    data: Dict[str, Any],
    clip_save_folder: str,
    data_dir: str,
) -> Tuple[List[str], float]:
    video_name = data["video_uid"]
    clips_dir = os.path.join(clip_save_folder, video_name)
    os.makedirs(clips_dir, exist_ok=True)

    video_path = os.path.join(data_dir, "video", f"{video_name}.mp4")
    clips, duration = split_video_to_clips(
        video_path=video_path,
        clip_dir=clips_dir,
        clip_duration=10,
        workers=24,
        tolerate_missing=2,
        fast_copy=True,
    )
    return clips, duration


@dataclass
class Segment:
    start: float
    end: float


def detect_segments(video_path: str, threshold: int = 27) -> List[Segment]:
    vm = VideoManager([video_path])
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=threshold))

    try:
        vm.start()
        base_tc = vm.get_base_timecode()
        sm.detect_scenes(frame_source=vm)
        scenes = sm.get_scene_list(base_tc)
    finally:
        vm.release()

    return [Segment(s[0].get_seconds(), s[1].get_seconds()) for s in scenes] or [
        Segment(0.0, get_video_duration(video_path))
    ]


def split_video(video_path: str, min_interval: int = 60, threshold: int = 27):
    _ = min_interval
    segments = detect_segments(video_path, threshold=threshold)
    intervals = [(int(seg.start), int(seg.end)) for seg in segments]
    return intervals


def sort_path(paths: Sequence[str]) -> List[str]:
    def clip_number(path: str) -> int:
        match = re.search(r"clip_(\d+)_", path)
        return int(match.group(1)) if match else -1

    return sorted(paths, key=clip_number)


_MARKERS = [
    r"first",
    r"second",
    r"third",
    r"next",
    r"then",
    r"after that",
    r"afterwards",
    r"subsequently",
    r"finally",
    r"last",
    r"lastly",
]
_MARKER_ALT = r"(?:\band\s+)?(?:" + "|".join(_MARKERS) + r")\b"

_SEGMENT_REGEX = re.compile(
    rf"(?is)"
    rf"(?:^|\W)"
    rf"(?P<marker>{_MARKER_ALT})"
    rf"[^\w]*"
    rf"(?P<seg>.*?)"
    rf"(?=(?:\W{_MARKER_ALT}\W)|$)"
)


def _normalize_event(text: str) -> str:
    t = text.strip()
    if len(t) > 1 and t[0] in "\"'‘“" and t[-1] in "\"'’”":
        t = t[1:-1].strip()
    t = re.sub(r"\s+", " ", t)
    t = t.strip(" \t\r\n.;,、，。")
    return t


def _events_from_sentence(sentence: str) -> List[str]:
    events = [_normalize_event(m.group("seg")) for m in _SEGMENT_REGEX.finditer(sentence)]
    events = [e for e in events if e]
    if not events:
        lone = _normalize_event(sentence)
        if lone:
            events = [lone]
    return events


def extract_unique_events(options: List[str]) -> List[str]:
    all_events = []
    for option in options:
        all_events.extend(_events_from_sentence(option))

    unique_events = list(dict.fromkeys(all_events))
    return unique_events if unique_events else options
