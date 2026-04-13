"""
VideoPro End-to-End Inference Pipeline
=======================================
Usage:
    python src/run.py \
        --video /path/to/video.mp4 \
        --question "What is the person doing?" \
        --choices "A. Cooking" "B. Playing guitar" "C. Riding a bicycle" "D. Swimming" \
        --clip_save_folder ./clips \
        --clip_duration 10

The pipeline:
    1. Generate a visual program (or native answer) via the deployed VLM.
    2. Execute the generated program within the video module runtime.
    3. If execution fails or confidence is low, trigger self-refinement.
    4. Return the final answer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from repo root or from src/
sys.path.insert(0, str(Path(__file__).parent))

from generate_code import infer_video_mcq
from execute_code import safe_run_execute_command
from refine_code import refine_code


# ---------------------------------------------------------------------------
# Confidence threshold: if the VLM returns a score below this value we
# trigger self-refinement even when execution succeeds.
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD = 0.75


def run_pipeline(
    video_path: str,
    question: str,
    choices: list[str],
    clip_save_folder: str = "./clips",
    clip_duration: int = 10,
    workers: int = 8,
    verbose: bool = True,
) -> dict:
    """
    Full VideoPro inference pipeline.

    Returns a dict with keys:
        answer          : str   – final predicted answer letter (e.g. "A")
        refined         : bool  – whether self-refinement was triggered
        prompt_type     : str   – "native" | "program" | refinement mode
        code            : str   – the executed program code
        error           : str   – execution error message (empty on success)
    """

    def log(msg: str):
        if verbose:
            print(msg)

    # ------------------------------------------------------------------
    # Step 1: Generate visual program
    # ------------------------------------------------------------------
    log("\n[1/3] Generating visual program ...")
    gen_result = infer_video_mcq(
        video_path=video_path,
        question=question,
        options=choices,
    )
    generated_text = gen_result["output_text"]
    log(f"    → generated output length: {len(generated_text)} chars")

    # ------------------------------------------------------------------
    # Step 2: Execute the generated program
    # ------------------------------------------------------------------
    log("\n[2/3] Executing visual program ...")
    exec_result = safe_run_execute_command(
        code_string=generated_text,
        video_path=video_path,
        question=question,
        choices=choices,
        duration=0,               # auto-detected inside
        clip_save_folder=clip_save_folder,
        clip_duration=clip_duration,
        workers=workers,
    )

    answer = exec_result.get("result", "")
    error  = exec_result.get("error", "")
    code   = exec_result.get("processed_code", "")
    success = exec_result.get("success", False)

    log(f"    → success={success}  answer={answer!r}  error={error!r}")

    # ------------------------------------------------------------------
    # Step 3: Self-refinement
    #   Triggered when:
    #     (a) execution failed (success=False), OR
    #     (b) native mode was used (low confidence by design)
    # ------------------------------------------------------------------
    refined = False
    prompt_type = "program"

    needs_refine = (not success) or ("query_native" in code)
    if needs_refine:
        log("\n[3/3] Self-refinement triggered ...")
        refine_result = refine_code(
            video_path=video_path,
            question=question,
            choices=choices,
            current_code=code,
            error_log=error if not success else None,
        )
        prompt_type = refine_result["prompt_type"]
        refined_code = refine_result["refined_code"]
        log(f"    → refinement mode: {prompt_type}")

        # Re-execute the refined program
        exec_result2 = safe_run_execute_command(
            code_string=f"<code>\n{refined_code}\n</code>",
            video_path=video_path,
            question=question,
            choices=choices,
            duration=0,
            clip_save_folder=clip_save_folder,
            clip_duration=clip_duration,
            workers=workers,
        )
        if exec_result2.get("success"):
            answer = exec_result2.get("result", answer)
            code   = exec_result2.get("processed_code", refined_code)
            error  = ""
        else:
            # keep original answer if refinement also failed
            error = exec_result2.get("error", error)

        refined = True
    else:
        log("\n[3/3] No refinement needed.")

    log(f"\n✓ Final answer: {answer!r}  (refined={refined}, mode={prompt_type})")

    return {
        "answer":      answer,
        "refined":     refined,
        "prompt_type": prompt_type,
        "code":        code,
        "error":       error,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VideoPro: adaptive program reasoning for long video understanding."
    )
    parser.add_argument("--video",    required=True, help="Path to the input video file.")
    parser.add_argument("--question", required=True, help="The question about the video.")
    parser.add_argument(
        "--choices",
        required=True,
        nargs="+",
        help='Answer choices, e.g. "A. Cooking" "B. Playing guitar" ...',
    )
    parser.add_argument(
        "--clip_save_folder",
        default="./clips",
        help="Directory where 10-second clips will be cached (default: ./clips).",
    )
    parser.add_argument(
        "--clip_duration",
        type=int,
        default=10,
        help="Duration of each video clip in seconds (default: 10).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers for video splitting (default: 8).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save the result as JSON.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose logging.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    result = run_pipeline(
        video_path=args.video,
        question=args.question,
        choices=args.choices,
        clip_save_folder=args.clip_save_folder,
        clip_duration=args.clip_duration,
        workers=args.workers,
        verbose=not args.quiet,
    )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Result saved to {out_path}")

    print(f"\nAnswer: {result['answer']}")
    return result


if __name__ == "__main__":
    main()
