from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import subprocess
import sys
import textwrap
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable
import inspect


CURRENT_DIR = Path(__file__).resolve().parent
UTILS_DIR = CURRENT_DIR / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from runtime import SAFE_BUILTINS, VideoRuntimeAPI, normalize_model_result


EPS = 1.0
CLIP_PREFIX = "clip_"
CLIP_SUFFIX = ".mp4"


def seconds_to_hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def probe_duration(video_path: str) -> float:
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
    output = subprocess.check_output(cmd, text=True).strip()
    duration = max(0.0, float(output))
    return math.floor(duration * 1000) / 1000.0


def resolve_video_path(video_path: str) -> str:
    resolved = str(Path(video_path).expanduser().resolve())
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"Video file not found: {resolved}")
    return resolved


def clip_index(name: str) -> int:
    try:
        return int(name.split("_")[1])
    except (IndexError, ValueError):
        return 10**9


def list_clip_files(clip_dir: str) -> list[str]:
    path = Path(clip_dir)
    if not path.exists():
        return []
    files = []
    for clip_path in path.iterdir():
        if not clip_path.is_file():
            continue
        if not clip_path.name.startswith(CLIP_PREFIX) or clip_path.suffix != CLIP_SUFFIX:
            continue
        if is_clip_file_valid(str(clip_path)):
            files.append(str(clip_path))
        else:
            try:
                clip_path.unlink()
            except OSError:
                pass
    return sorted(files, key=lambda item: clip_index(Path(item).name))


def is_clip_file_valid(path: str) -> bool:
    file_path = Path(path)
    return file_path.exists() and file_path.stat().st_size > 0


def clean_invalid_clips(paths: list[str]) -> list[str]:
    valid = []
    for path in paths:
        try:
            if is_clip_file_valid(path):
                valid.append(path)
            elif os.path.exists(path):
                os.remove(path)
        except OSError:
            continue
    return valid


def split_video_to_clips(
    video_path: str,
    clip_dir: str,
    clip_duration: int = 10,
    workers: int = 8,
    tolerate_missing: int = 2,
    overwrite: bool = True,
) -> tuple[list[str], float]:
    Path(clip_dir).mkdir(parents=True, exist_ok=True)
    video_path = resolve_video_path(video_path)

    duration = probe_duration(video_path)
    expected_count = math.ceil((duration - EPS) / clip_duration) if duration > EPS else 0

    existing = list_clip_files(clip_dir)
    if len(existing) >= max(0, expected_count - tolerate_missing):
        return existing, duration

    def run_ffmpeg(cmd: list[str]) -> None:
        subprocess.run(cmd, check=True)

    def run_moviepy_split(start: float, end: float, output_path: str) -> str | None:
        from moviepy.video.io.VideoFileClip import VideoFileClip

        if end - start <= EPS:
            return None

        with VideoFileClip(video_path) as clip:
            subclip = clip.subclipped(start, end)
            try:
                subclip.write_videofile(
                    output_path,
                    codec="libx264",
                    audio_codec="aac",
                    logger=None,
                )
            except Exception:
                subclip.write_videofile(output_path, logger=None)
        return output_path

    def cut_one(index: int, start: float, end: float) -> str | None:
        if end - start <= EPS:
            return None

        start_tag = seconds_to_hms(start).replace(":", "-")
        end_tag = seconds_to_hms(end).replace(":", "-")
        output_path = os.path.join(
            clip_dir,
            f"clip_{index}_{start_tag}_to_{end_tag}.mp4",
        )

        overwrite_flag = ["-y"] if overwrite else ["-n"]
        duration_str = str(max(0.0, end - start))
        ffmpeg_attempts = [
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                *overwrite_flag,
                "-ss",
                str(start),
                "-i",
                video_path,
                "-t",
                duration_str,
                "-c",
                "copy",
                "-avoid_negative_ts",
                "make_zero",
                output_path,
            ],
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                *overwrite_flag,
                "-ss",
                str(start),
                "-i",
                video_path,
                "-t",
                duration_str,
                "-avoid_negative_ts",
                "make_zero",
                output_path,
            ],
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                *overwrite_flag,
                "-ss",
                str(start),
                "-i",
                video_path,
                "-t",
                duration_str,
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                "-avoid_negative_ts",
                "make_zero",
                output_path,
            ],
        ]

        last_error: Exception | None = None
        for cmd in ffmpeg_attempts:
            try:
                run_ffmpeg(cmd)
                if is_clip_file_valid(output_path):
                    return output_path
            except Exception as exc:
                last_error = exc
                if os.path.exists(output_path):
                    try:
                        os.remove(output_path)
                    except OSError:
                        pass

        try:
            return run_moviepy_split(start, end, output_path)
        except Exception as exc:
            if last_error is not None:
                raise RuntimeError(
                    f"Failed to split clip with ffmpeg and moviepy for {output_path}"
                ) from last_error
            raise exc

    futures = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index in range(expected_count):
            start = index * clip_duration
            end = min((index + 1) * clip_duration, duration)
            if end - start > EPS:
                futures.append(executor.submit(cut_one, index, start, end))

        outputs = [future.result() for future in as_completed(futures)]

    clips = clean_invalid_clips([path for path in outputs if path])
    return sorted(clips, key=lambda item: clip_index(Path(item).name)), duration


def ensure_video_clips(
    video_path: str,
    clip_save_folder: str,
    clip_duration: int = 10,
    workers: int = 8,
) -> tuple[str, list[str], float]:
    video_path = resolve_video_path(video_path)
    video_id = Path(video_path).stem
    clip_dir = os.path.join(clip_save_folder, video_id)
    Path(clip_dir).mkdir(parents=True, exist_ok=True)

    clip_paths = list_clip_files(clip_dir)
    if clip_paths:
        return clip_dir, clip_paths, probe_duration(video_path)

    clip_paths, duration = split_video_to_clips(
        video_path=video_path,
        clip_dir=clip_dir,
        clip_duration=clip_duration,
        workers=workers,
        tolerate_missing=2,
        overwrite=True,
    )
    return clip_dir, clip_paths, duration


def extract_code_block(code_string: str) -> str:
    tagged = re.search(r"<code>(.*?)</code>", code_string, flags=re.DOTALL | re.IGNORECASE)
    if tagged:
        return tagged.group(1).strip()

    fenced = re.search(r"```python(.*?)```", code_string, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()

    return code_string.strip()


def process_generated_code(code_string: str) -> str:
    raw_code = textwrap.dedent(extract_code_block(code_string)).strip()
    if not raw_code:
        raise ValueError("No executable code was found in the generated program.")

    ast.parse(raw_code)
    if "def execute_command" not in raw_code:
        raise ValueError("Generated code must define execute_command(...).")
    return raw_code


def build_runtime(
    video_path: str,
    clip_dir: str,
    clip_paths: list[str],
    clip_save_folder: str,
    clip_duration: int,
) -> tuple[VideoRuntimeAPI, dict[str, Any]]:
    # Retrieval uses the same cached 10-second clip directory prepared by the executor.
    api = VideoRuntimeAPI(
        clip_save_folder=clip_save_folder,
        clip_duration=clip_duration,
        current_video_path=video_path,
        current_clip_dir=clip_dir,
        current_clip_paths=clip_paths,
    )
    runtime_globals: dict[str, Any] = {"__builtins__": SAFE_BUILTINS}
    runtime_globals.update(api.build_globals())
    return api, runtime_globals


def compile_execute_function(
    code_string: str,
    runtime_globals: dict[str, Any],
) -> tuple[Callable[..., Any], str]:
    processed_code = process_generated_code(code_string)
    local_env: dict[str, Any] = {}
    exec(processed_code, runtime_globals, local_env)
    execute_command = local_env.get("execute_command") or runtime_globals.get("execute_command")
    if execute_command is None:
        raise ValueError("No function named execute_command found in input code.")
    return execute_command, processed_code


def run_execute_command(
    code_string: str,
    video_path: str,
    question: str,
    choices: list[str],
    duration: int | float | None,
    clip_save_folder: str,
    clip_duration: int = 10,
    workers: int = 8,
) -> dict[str, Any]:
    video_path = resolve_video_path(video_path)
    clip_dir, clip_paths, real_duration = ensure_video_clips(
        video_path=video_path,
        clip_save_folder=clip_save_folder,
        clip_duration=clip_duration,
        workers=workers,
    )

    _, runtime_globals = build_runtime(
        video_path=video_path,
        clip_dir=clip_dir,
        clip_paths=clip_paths,
        clip_save_folder=clip_save_folder,
        clip_duration=clip_duration,
    )
    normalized_duration = int(math.ceil(real_duration)) if not duration or duration <= 0 else int(duration)
    runtime_globals["question"] = question
    runtime_globals["choices"] = choices
    runtime_globals["duration"] = normalized_duration
    runtime_globals["video"] = video_path
    execute_command, processed_code = compile_execute_function(code_string, runtime_globals)

    param_count = len(inspect.signature(execute_command).parameters)
    if param_count <= 2:
        raw_result = execute_command(video_path, question)
    else:
        raw_result = execute_command(video_path, question, choices, normalized_duration)
    result = normalize_model_result(
        raw_result,
        choices=choices,
        metadata={
            "video_path": video_path,
            "question": question,
            "duration": normalized_duration,
            "clip_dir": clip_dir,
            "clip_duration": clip_duration,
            "num_clips": len(clip_paths),
        },
    )

    return {
        "success": True,
        "result": result,
        "answer": result["answer"],
        "confidence": result["confidence"],
        "raw_output": result["raw_output"],
        "processed_code": processed_code,
        "clip_dir": clip_dir,
        "clip_paths": clip_paths,
        "num_clips": len(clip_paths),
        "error": "",
        "traceback": "",
    }


def safe_run_execute_command(
    code_string: str,
    video_path: str,
    question: str,
    choices: list[str],
    duration: int | float | None,
    clip_save_folder: str,
    clip_duration: int = 10,
    workers: int = 8,
) -> dict[str, Any]:
    processed_code = ""
    try:
        try:
            processed_code = process_generated_code(code_string)
        except Exception:
            processed_code = extract_code_block(code_string).strip()

        return run_execute_command(
            code_string=code_string,
            video_path=video_path,
            question=question,
            choices=choices,
            duration=duration,
            clip_save_folder=clip_save_folder,
            clip_duration=clip_duration,
            workers=workers,
        )
    except Exception as exc:
        return {
            "success": False,
            "result": normalize_model_result(
                {},
                choices=choices,
                metadata={"video_path": video_path, "question": question},
            ),
            "answer": "",
            "confidence": 0.0,
            "raw_output": "",
            "processed_code": processed_code,
            "clip_dir": "",
            "clip_paths": [],
            "num_clips": 0,
            "stage": "execute",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute a VideoPro generated visual program against the runtime APIs.")
    parser.add_argument("--video", required=True, help="Path to the input video.")
    parser.add_argument("--question", required=True, help="Multiple-choice question about the video.")
    parser.add_argument("--choices", required=True, nargs="+", help="Answer choices.")
    parser.add_argument("--code-file", default=None, help="Path to a file containing generated code or a full model response.")
    parser.add_argument("--code", default=None, help="Inline generated code or a full model response.")
    parser.add_argument("--duration", type=int, default=0, help="Optional video duration override. Use 0 to auto-detect.")
    parser.add_argument("--clip-save-folder", default="./clips", help="Folder for cached video clips.")
    parser.add_argument("--clip-duration", type=int, default=10, help="Clip duration in seconds.")
    parser.add_argument("--workers", type=int, default=8, help="Workers used for video splitting.")
    parser.add_argument("--output", default=None, help="Optional JSON file for the execution result.")
    return parser.parse_args()


def _load_code(args: argparse.Namespace) -> str:
    if args.code:
        return args.code
    if args.code_file:
        return Path(args.code_file).read_text(encoding="utf-8")
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise ValueError("Provide generated code via --code, --code-file, or stdin.")


def main() -> dict[str, Any]:
    args = parse_args()
    code_string = _load_code(args)
    result = safe_run_execute_command(
        code_string=code_string,
        video_path=args.video,
        question=args.question,
        choices=args.choices,
        duration=args.duration,
        clip_save_folder=args.clip_save_folder,
        clip_duration=args.clip_duration,
        workers=args.workers,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
