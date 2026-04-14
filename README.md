# VideoPro

VideoPro is a long-video reasoning framework that combines program generation, executable video APIs, and self-refinement. After the model is deployed, the repository can run the full inference loop:

1. `src/generate_code.py` generates a visual program.
2. `src/execute_code.py` executes the generated program against the packaged video APIs in `src/utils`.
3. `src/refine_code.py` rewrites the program when execution fails or confidence is below a threshold.
4. `src/run.py` connects the full pipeline end to end.

The implementation in this repository is aligned with the paper idea of adaptive routing plus refinement: simple questions can stay in native VideoLLM mode, while harder questions are routed into multi-step visual programs that use retrieval, frame extraction, detection, subtitle hints, and temporal trimming.

For local testing, if you have a file like `test/video.mp4`, you can use it directly with every CLI example below by replacing `/path/to/video.mp4`.

## Repository Layout

```text
VideoPro/
├── src/
│   ├── run.py
│   ├── generate_code.py
│   ├── execute_code.py
│   ├── refine_code.py
│   └── utils/
│       ├── runtime.py
│       ├── analysis.py
│       ├── retriever.py
│       └── video_utils.py
├── scripts/
│   ├── deploy.sh
│   └── train.sh
├── dataset/
├── docs/
├── models/
├── requirements.txt
└── README.md
```

## Environment

```bash
conda create -n videopro python=3.10 -y
conda activate videopro
pip install -r requirements.txt
pip install flash_attn==2.8.3 --no-build-isolation
```

The project assumes:

- `ffmpeg` and `ffprobe` are available in `PATH`
- the deployed VLM is exposed through an OpenAI-compatible API
- the local model assets used by `src/utils/analysis.py` and `src/utils/retriever.py` are present under `models/`

## Model Assets

Download the checkpoints you need under `./models`. The repo currently expects at least:

- the served VideoPro/Qwen3-VL checkpoint for inference
- `models/LanguageBind_Video_FT`
- `models/LanguageBind_Image`
- `models/bge-m3`
- `models/grounding-dino-base`

If you keep the paths unchanged, the provided scripts will work without further edits.

## Deploy the Model

Use the provided deployment script:

```bash
bash scripts/deploy.sh
```

The default deployment serves the model at `http://0.0.0.0:8007/v1` with served model name `qwen3vl`.

If your endpoint differs, set:

```bash
export VIDEOPRO_API_BASE="http://your-host:port/v1"
export VIDEOPRO_MODEL="qwen3vl"
```

## Runtime API Surface

Generated programs are executed with a stable API layer from [`src/utils/runtime.py`](src/utils/runtime.py). The main callable functions available inside `execute_command(...)` are:

- `query_native(video_path, question, choices)`
- `query_mc(frames, question, choices)`
- `query_frames(frames, question)`
- `query_yn(frames, question)`
- `get_informative_clips(video_path, query, top_k=3, total_duration=None)`
- `get_subtitle_hints(video_path, question, choices, duration)`
- `extract_frames(video_path, num_frames=32)`
- `detect_object(frame, text)`
- `trim_frames(video_path, start, end, num_frames=64)`
- `trim_around(video_path, timestamp, intervals=30, num_frames=64)`
- `trim_before(video_path, timestamp, intervals=30, num_frames=64)`
- `trim_after(video_path, timestamp, intervals=30, num_frames=64)`
- `crop(...)`, `crop_left(...)`, `crop_right(...)`, `crop_top(...)`, `crop_bottom(...)`
- `make_result(answer="", confidence=0.0, raw_output="", **metadata)`

All answer-producing APIs now return the same structure:

```json
{
  "answer": "B",
  "confidence": 0.83,
  "raw_output": "B",
  "metadata": {
    "mode": "mc"
  }
}
```

This is the key contract used by `execute_code.py` and `run.py` to decide whether refinement is needed.

## Generate a Visual Program

```bash
python src/generate_code.py \
  --video /path/to/video.mp4 \
  --question "What is the person doing in the video?" \
  --choices "Cooking in the kitchen" "Playing guitar" "Riding a bicycle" "Swimming in a pool" \
  --output generation.json \
  --output-code generation.py
```

The model is asked to return:

- a `<planning>` block describing routing and reasoning
- a `<code>` block containing a single `execute_command(video_path, question, choices, duration)` function

## Execute Generated Code

`src/execute_code.py` executes either:

- a full model response containing `<code>...</code>`
- a plain Python file containing `execute_command(...)`
- inline code passed through `--code`

Example:

```bash
python src/execute_code.py \
  --video /path/to/video.mp4 \
  --question "What is the person doing in the video?" \
  --choices "Cooking in the kitchen" "Playing guitar" "Riding a bicycle" "Swimming in a pool" \
  --code-file generation.py \
  --clip-save-folder ./clips \
  --clip-duration 10 \
  --output execute_result.json
```

Execution always starts by preparing clip cache under `clip_save_folder/<video_id>/`:

1. the input video is split into 10-second `clip_*.mp4` segments
2. the retrieval stack reuses that clip cache when building LanguageBind embeddings under `dataset/embeddings/`
3. retrieval APIs such as `get_informative_clips(...)` work on those cached segments

The execution result includes:

- `success`
- `answer`
- `confidence`
- `raw_output`
- `processed_code`
- `traceback` when execution fails

## Refine Generated Code

When the program crashes or confidence is too low, run:

```bash
python src/refine_code.py \
  --video /path/to/video.mp4 \
  --question "What is the person doing in the video?" \
  --choices "Cooking in the kitchen" "Playing guitar" "Riding a bicycle" "Swimming in a pool" \
  --code-file generation.py \
  --result-json execute_result.json \
  --confidence-threshold 0.75 \
  --output refinement.json \
  --output-code refined.py
```

`refine_code.py` supports three cases:

- execution bug fix using traceback
- native-mode low-confidence refinement
- program low-confidence refinement

## End-to-End Pipeline

Use `src/run.py` for the full loop:

```bash
python src/run.py \
  --video /path/to/video.mp4 \
  --question "What is the person doing in the video?" \
  --choices "Cooking in the kitchen" "Playing guitar" "Riding a bicycle" "Swimming in a pool" \
  --clip-save-folder ./clips \
  --clip-duration 10 \
  --confidence-threshold 0.75 \
  --max-refine-rounds 1 \
  --output final_result.json
```

Pipeline behavior:

1. Generate the initial program.
2. Execute it with the runtime APIs.
   The executor first cuts the input video into 10-second clips and exposes those clips to the retrieval runtime for LanguageBind embedding.
3. If execution fails, answer is empty, or confidence is below the threshold, refine the code.
4. Re-execute the refined code and return the final result.

If you already have a saved program and want to skip generation:

```bash
python src/run.py \
  --video /path/to/video.mp4 \
  --question "What is the person doing in the video?" \
  --choices "Cooking in the kitchen" "Playing guitar" "Riding a bicycle" "Swimming in a pool" \
  --code-file generation.py
```

## Generated Program Example

After deployment, `src/generate_code.py` may produce code like:

```python
def execute_command(video_path, question, choices, duration):
    try:
        intervals, clip_paths = get_informative_clips(
            video_path,
            "person doing an activity",
            top_k=2,
            total_duration=duration,
        )
    except Exception:
        intervals, clip_paths = [], []

    frames = []
    for clip_path in clip_paths:
        try:
            frames.extend(extract_frames(clip_path, num_frames=16))
        except Exception:
            continue

    if not frames:
        frames = extract_frames(video_path, num_frames=32)

    person_frames = []
    for frame in frames:
        boxes = detect_object(frame, "person")
        if boxes:
            person_frames.append(frame)

    return query_mc(person_frames or frames, question, choices)
```

This style now runs directly because `query_mc(...)` returns a structured result that the executor can normalize and score.

## Open-Sourcing Notes

Before publishing the repository, check the following:

- remove hard-coded absolute local paths if your release environment differs
- document which external checkpoints are required and how to download them
- document GPU requirements for deployment, retrieval, and detection
- confirm the license compatibility of bundled model weights and third-party code
- add a small demo asset or minimal benchmark example if you want one-command reproducibility

## Verification Status

The repository code has been refactored so that:

- the runtime API surface is explicit and consistent
- `generate_code.py`, `execute_code.py`, `refine_code.py`, and `run.py` share the same result contract
- low-confidence execution now triggers refinement based on structured confidence

Actual end-to-end execution still depends on your local model deployment, weights, and GPU environment being available.
