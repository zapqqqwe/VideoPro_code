from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from execute_code import safe_run_execute_command
from generate_code import infer_video_mcq
from generate_code import resolve_video_path
from refine_code import refine_code


def run_pipeline(
    video_path: str,
    question: str,
    choices: list[str],
    clip_save_folder: str = "./clips",
    clip_duration: int = 10,
    workers: int = 8,
    confidence_threshold: float = 0.75,
    max_refine_rounds: int = 1,
    model: str = "qwen3vl",
    initial_code: str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    video_path = resolve_video_path(video_path)

    def log(message: str) -> None:
        if verbose:
            print(message)

    generation_result: dict[str, Any] | None = None
    if initial_code:
        current_code = initial_code
        log("[1/3] Using provided visual program.")
    else:
        log("[1/3] Generating visual program ...")
        generation_result = infer_video_mcq(
            video_path=video_path,
            question=question,
            options=choices,
            model=model,
        )
        current_code = generation_result["code"] or generation_result["output_text"]
        log(f"    generated program length: {len(current_code)} chars")

    refinement_history: list[dict[str, Any]] = []
    execution_result: dict[str, Any] = {}

    for round_index in range(max_refine_rounds + 1):
        if round_index == 0:
            log("[2/3] Executing visual program ...")
        else:
            log(f"[3/3] Re-executing refined program (round {round_index}) ...")

        execution_result = safe_run_execute_command(
            code_string=current_code,
            video_path=video_path,
            question=question,
            choices=choices,
            duration=0,
            clip_save_folder=clip_save_folder,
            clip_duration=clip_duration,
            workers=workers,
        )

        success = execution_result.get("success", False)
        answer = execution_result.get("answer", "")
        confidence = float(execution_result.get("confidence", 0.0))
        error = execution_result.get("error", "")
        processed_code = execution_result.get("processed_code", "")

        log(f"    success={success} answer={answer!r} confidence={confidence:.3f}")
        if not success and error:
            log(f"    error={error}")
        if not success and processed_code:
            log("    processed_code:")
            print(processed_code)

        needs_refine = (not success) or (not answer) or (confidence < confidence_threshold)
        if not needs_refine:
            break

        if round_index >= max_refine_rounds:
            log("[3/3] Reached refinement limit.")
            break

        log("[3/3] Refining visual program ...")
        refine_result = refine_code(
            video_path=video_path,
            question=question,
            choices=choices,
            current_code=execution_result.get("processed_code") or current_code,
            previous_result=execution_result.get("result"),
            error_log=execution_result.get("traceback") if not success else None,
            confidence_threshold=confidence_threshold,
            model=model,
        )
        refinement_history.append(refine_result)
        current_code = refine_result["refined_code"] or refine_result["output_text"]
        log(f"    refinement type: {refine_result['prompt_type']}")

    final_result = execution_result.get("result", {})
    if not execution_result.get("success") and verbose and execution_result.get("traceback"):
        print(execution_result["traceback"])
    return {
        "success": execution_result.get("success", False),
        "answer": execution_result.get("answer", ""),
        "confidence": execution_result.get("confidence", 0.0),
        "raw_output": execution_result.get("raw_output", ""),
        "result": final_result,
        "code": execution_result.get("processed_code", current_code),
        "generation": generation_result,
        "refinements": refinement_history,
        "refined": bool(refinement_history),
        "refinement_rounds": len(refinement_history),
        "confidence_threshold": confidence_threshold,
        "execution": execution_result,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VideoPro end-to-end pipeline: generate -> execute -> refine.")
    parser.add_argument("--video", required=True, help="Path to the input video.")
    parser.add_argument("--question", required=True, help="Multiple-choice question about the video.")
    parser.add_argument("--choices", required=True, nargs="+", help="Answer choices.")
    parser.add_argument("--clip-save-folder", default="./clips", help="Folder for cached video clips.")
    parser.add_argument("--clip-duration", type=int, default=10, help="Video clip duration in seconds.")
    parser.add_argument("--workers", type=int, default=8, help="Workers used for video splitting.")
    parser.add_argument("--confidence-threshold", type=float, default=0.75, help="Refinement threshold.")
    parser.add_argument("--max-refine-rounds", type=int, default=1, help="Maximum refinement retries.")
    parser.add_argument("--model", default="qwen3vl", help="Served model name.")
    parser.add_argument("--code-file", default=None, help="Optional existing code file. When set, skip generation.")
    parser.add_argument("--output", default=None, help="Optional JSON file for the final pipeline result.")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress logs.")
    return parser.parse_args()


def main() -> dict[str, Any]:
    args = parse_args()
    initial_code = None
    if args.code_file:
        initial_code = Path(args.code_file).read_text(encoding="utf-8")

    result = run_pipeline(
        video_path=args.video,
        question=args.question,
        choices=args.choices,
        clip_save_folder=args.clip_save_folder,
        clip_duration=args.clip_duration,
        workers=args.workers,
        confidence_threshold=args.confidence_threshold,
        max_refine_rounds=args.max_refine_rounds,
        model=args.model,
        initial_code=initial_code,
        verbose=not args.quiet,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nAnswer: {result['answer']} (confidence={result['confidence']:.3f})")
    return result


if __name__ == "__main__":
    main()
