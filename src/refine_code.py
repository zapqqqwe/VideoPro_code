from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from openai import OpenAI

from generate_code import DEFAULT_API_BASE, DEFAULT_API_KEY, DEFAULT_MODEL, RUNTIME_API_REFERENCE
from generate_code import resolve_video_path


def create_client(
    api_key: str = DEFAULT_API_KEY,
    api_base: str = DEFAULT_API_BASE,
) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=api_base)


def build_question_with_choices(question: str, choices: Sequence[str]) -> str:
    letters = [chr(65 + i) for i in range(len(choices))]
    choices_str = "\n".join(f"{letter}. {choice}" for letter, choice in zip(letters, choices))
    return f"Question: {question}\nChoices:\n{choices_str}"


def extract_refined_code(output_text: str) -> str:
    if not output_text:
        return ""

    tagged = re.search(r"<code>(.*?)</code>", output_text, flags=re.DOTALL | re.IGNORECASE)
    if tagged:
        return tagged.group(1).strip()

    fenced = re.search(r"```python(.*?)```", output_text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()

    function_match = re.search(
        r"def\s+execute_command\s*\(\s*(?:video_path|video)\s*,\s*question(?:\s*,\s*choices\s*,\s*duration)?\s*\)\s*:(.*)",
        output_text,
        flags=re.DOTALL,
    )
    return function_match.group(0).strip() if function_match else output_text.strip()


def _format_previous_result(previous_result: Mapping[str, Any] | None) -> str:
    if not previous_result:
        return "No previous execution result was available."
    return json.dumps(dict(previous_result), ensure_ascii=False, indent=2)


def build_refine_prompt(
    question: str,
    choices: Sequence[str],
    current_code: str,
    previous_result: Mapping[str, Any] | None = None,
    error_log: str | None = None,
    confidence_threshold: float = 0.75,
) -> tuple[str, str]:
    question_block = build_question_with_choices(question, choices)
    previous_result_block = _format_previous_result(previous_result)
    confidence = float(previous_result.get("confidence", 0.0)) if previous_result else 0.0

    if error_log:
        prompt_type = "bug_fix"
        instruction = """
You will receive a multiple-choice question about a video and a Python visual program in the
execute_command format, and a runtime error log from running this program.
{question_with_choices}
Buggy visual program: {current_code}
Runtime error log: {error_log}
Refine this visual program by fixing the bugs.
""".strip()
    elif "query_native" in current_code and confidence < confidence_threshold:
        prompt_type = "native_low_confidence"
        instruction = """
You will receive a multiple-choice question about a video and an existing visual program that only
uses the native-mode helper API query_native.
{question_with_choices}
Current native visual program:
<code>
{current_code}
</code>
Refine this visual program
""".strip()
    elif confidence < confidence_threshold:
        prompt_type = "program_low_confidence"
        instruction = """
You will receive a multiple-choice question about a video and an existing visual program.
{question_with_choices}
Current visual program: {current_code}
Refine this visual program to improve its reasoning and correctness
""".strip()
    else:
        prompt_type = "general_refine"
        instruction = """
You will receive a multiple-choice question about a video and an existing visual program.
{question_with_choices}
Current visual program: {current_code}
Refine this visual program to improve its reasoning and correctness
""".strip()

    instruction = instruction.format(
        question_with_choices=question_block,
        error_log=(error_log or "").strip(),
        current_code=current_code.strip(),
    )
    prompt_parts = [
        instruction,
        "",
        RUNTIME_API_REFERENCE,
    ]

    if prompt_type != "native_low_confidence":
        prompt_parts.extend(
            [
                "",
                "Current visual program:",
                f"<code>\n{current_code.strip()}\n</code>",
            ]
        )

    if previous_result:
        prompt_parts.extend(["", "Previous execution result:", previous_result_block])

    prompt_parts.extend(
        [
            "",
            "Constraints:",
            "- Do not import anything.",
            "- Define exactly one function named execute_command.",
            "- Prefer the signature `def execute_command(video, question):`.",
            "- Use the input question instead of hard-coding text when possible.",
            "- Prefer robust fallbacks.",
            "- Return `answer` from the function body.",
            "- `choices` and `duration` are available in the execution environment.",
            "- Return only the requested tagged/code content, with no extra prose.",
        ]
    )

    return "\n".join(prompt_parts).strip(), prompt_type


def refine_code(
    video_path: str,
    question: str,
    choices: Sequence[str],
    current_code: str,
    previous_result: Mapping[str, Any] | None = None,
    error_log: str | None = None,
    confidence_threshold: float = 0.75,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 32000,
    api_base: str = DEFAULT_API_BASE,
) -> dict:
    client = create_client(api_base=api_base)
    video_path = resolve_video_path(video_path)
    prompt, prompt_type = build_refine_prompt(
        question=question,
        choices=choices,
        current_code=current_code,
        previous_result=previous_result,
        error_log=error_log,
        confidence_threshold=confidence_threshold,
    )

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
        temperature=0,
    )

    output_text = response.choices[0].message.content if response.choices else ""
    refined_code = extract_refined_code(output_text)

    return {
        "prompt_type": prompt_type,
        "prompt": prompt,
        "output_text": output_text,
        "refined_code": refined_code,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine VideoPro generated programs after failure or low confidence.")
    parser.add_argument("--video", required=True, help="Path to the input video.")
    parser.add_argument("--question", required=True, help="Multiple-choice question about the video.")
    parser.add_argument("--choices", required=True, nargs="+", help="Answer choices.")
    parser.add_argument("--code-file", default=None, help="Path to the current execute_command code.")
    parser.add_argument("--code", default=None, help="Code text when not using --code-file.")
    parser.add_argument("--error-log", default=None, help="Optional runtime error log.")
    parser.add_argument("--result-json", default=None, help="Optional JSON file containing the previous execution result.")
    parser.add_argument("--confidence-threshold", type=float, default=0.75, help="Refinement threshold.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Served model name (default: {DEFAULT_MODEL}).")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE, help=f"OpenAI-compatible API base (default: {DEFAULT_API_BASE}).")
    parser.add_argument("--output", default=None, help="Optional JSON file for the full refinement result.")
    parser.add_argument("--output-code", default=None, help="Optional path to save only the refined code.")
    return parser.parse_args()


def _load_text(code: str | None, code_file: str | None) -> str:
    if code:
        return code
    if code_file:
        return Path(code_file).read_text(encoding="utf-8")
    raise ValueError("Either --code or --code-file must be provided.")


def main() -> dict:
    args = parse_args()
    current_code = _load_text(args.code, args.code_file)
    previous_result = None
    if args.result_json:
        previous_result = json.loads(Path(args.result_json).read_text(encoding="utf-8"))

    result = refine_code(
        video_path=args.video,
        question=args.question,
        choices=args.choices,
        current_code=current_code,
        previous_result=previous_result,
        error_log=args.error_log,
        confidence_threshold=args.confidence_threshold,
        model=args.model,
        api_base=args.api_base,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.output_code:
        code_path = Path(args.output_code)
        code_path.parent.mkdir(parents=True, exist_ok=True)
        code_path.write_text(result["refined_code"], encoding="utf-8")

    print(result["output_text"])
    return result


if __name__ == "__main__":
    main()
