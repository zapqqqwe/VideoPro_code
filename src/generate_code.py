from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Sequence

from openai import OpenAI


DEFAULT_MODEL = os.getenv("VIDEOPRO_MODEL", "qwen3vl")
DEFAULT_API_BASE = os.getenv("VIDEOPRO_API_BASE", "http://0.0.0.0:8007/v1")
DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")

RUNTIME_API_REFERENCE = """
Available helper APIs and variables:
- query_native(video, question, choices) -> {"answer", "confidence", "raw_output", "metadata"}
- query_mc(frames, question, choices) -> {"answer", "confidence", "raw_output", "metadata"}
- query_frames(frames, question) -> {"answer", "confidence", "raw_output", "metadata"}
- query_yn(frames, question) -> {"answer", "confidence", "raw_output", "metadata"}
- get_informative_clips(video, query, top_k=3, total_duration=None) -> (intervals, clip_paths)
- get_subtitle_hints(video, question, choices, duration) -> str
- extract_frames(video, num_frames=32) -> list[PIL.Image]
- detect_object(frame, text) -> list[[x1, y1, x2, y2]]
- trim_frames(video, start, end, num_frames=64) -> list[PIL.Image]
- trim_around(video, timestamp, intervals=30, num_frames=64) -> list[PIL.Image]
- trim_before(video, timestamp, intervals=30, num_frames=64) -> list[PIL.Image]
- trim_after(video, timestamp, intervals=30, num_frames=64) -> list[PIL.Image]
- crop(frame, box) and crop_left/right/top/bottom(...) -> PIL.Image
- make_result(answer="", confidence=0.0, raw_output="", **metadata) -> dict
- `choices` and `duration` are available in the execution environment.
""".strip()

INSTRUCTION_TEMPLATE = """
You will receive a multiple-choice question about a video. Your output must define a Python
function in the following format:
<planning>
Briefly judge whether the question can be direct solved by videoLLM and plan the main API calls
and reasoning steps you will use.
• If it can be answered, use native reasoning mode with a single query_native call.
• If it needs long-range or multi-step reasoning, use more detailed visual program mode with
other modules (retrieval, subtitles, frame analysis, etc.).
</planning>
<code>
Write a Python function in the following format.
def execute_command(video, question):
# Visual program code (no comments needed inside the code body).
...
return answer
</code>
""".strip()


def build_prompt(question: str, options: Sequence[str]) -> str:
    letters = [chr(65 + i) for i in range(len(options))]
    choices_str = "\n".join(f"{letter}. {option}" for letter, option in zip(letters, options))
    return (
        f"{INSTRUCTION_TEMPLATE}\n\n"
        f"{RUNTIME_API_REFERENCE}\n\n"
        f"Question: {question}\n"
        f"Choices:\n{choices_str}\n"
    )


def extract_tag_content(text: str, tag: str) -> str:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def extract_code(text: str) -> str:
    code = extract_tag_content(text, "code")
    if code:
        return code

    fence = re.search(r"```python(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()

    function_match = re.search(
        r"def\s+execute_command\s*\(\s*(?:video_path|video)\s*,\s*question(?:\s*,\s*choices\s*,\s*duration)?\s*\)\s*:(.*)",
        text,
        flags=re.DOTALL,
    )
    return function_match.group(0).strip() if function_match else text.strip()


def create_client(
    api_key: str = DEFAULT_API_KEY,
    api_base: str = DEFAULT_API_BASE,
) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=api_base)


def resolve_video_path(video_path: str) -> str:
    resolved = str(Path(video_path).expanduser().resolve())
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"Video file not found: {resolved}")
    return resolved


def infer_video_mcq(
    video_path: str,
    question: str,
    options: Sequence[str],
    model: str = DEFAULT_MODEL,
    max_tokens: int = 32000,
    temperature: float = 0.0,
    api_base: str = DEFAULT_API_BASE,
) -> dict:
    client = create_client(api_base=api_base)
    video_path = resolve_video_path(video_path)
    prompt = build_prompt(question, options)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )

    output_text = response.choices[0].message.content if response.choices else ""
    planning = extract_tag_content(output_text, "planning")
    code = extract_code(output_text)

    return {
        "prompt": prompt,
        "planning": planning,
        "code": code,
        "output_text": output_text,
        "model": model,
        "video_path": video_path,
        "question": question,
        "choices": list(options),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate VideoPro visual programs from a deployed VLM.")
    parser.add_argument("--video", required=True, help="Path to the input video.")
    parser.add_argument("--question", required=True, help="Multiple-choice question about the video.")
    parser.add_argument("--choices", required=True, nargs="+", help="Answer choices.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Served model name (default: {DEFAULT_MODEL}).")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE, help=f"OpenAI-compatible API base (default: {DEFAULT_API_BASE}).")
    parser.add_argument("--max-tokens", type=int, default=32000, help="Maximum response tokens.")
    parser.add_argument("--output", default=None, help="Optional JSON path for the full generation result.")
    parser.add_argument("--output-code", default=None, help="Optional path to save only the generated code.")
    return parser.parse_args()


def main() -> dict:
    args = parse_args()
    result = infer_video_mcq(
        video_path=args.video,
        question=args.question,
        options=args.choices,
        model=args.model,
        max_tokens=args.max_tokens,
        api_base=args.api_base,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.output_code:
        code_path = Path(args.output_code)
        code_path.parent.mkdir(parents=True, exist_ok=True)
        code_path.write_text(result["code"], encoding="utf-8")

    print(result["output_text"])
    return result


if __name__ == "__main__":
    main()
