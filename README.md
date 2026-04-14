<div align="center">
    <h1 align="center">VideoPro: Adaptive Program Reasoning for Long Video Understanding
    </h1>

<a href="https://arxiv.org/abs/2509.17743">
<img src='https://img.shields.io/badge/arXiv-2509.17743-blue' alt='Paper PDF'></a>

[![Project Website](https://img.shields.io/badge/Website-VideoPro-blue)](https://zapqqqwe.github.io/VideoPro_/)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-VideoPro_Model-yellow)](https://huggingface.co/zapqqqwe/videopro_grpo)

</div>


## 🔥 News
- [2026/04/12] 🔥🔥 We release the code and model of **VideoPro**!


## Introduction

We propose <b>VideoPro</b>, a unified framework for long-video understanding with <b>adaptive reasoning</b> and <b>self-refinement</b>. Long-video understanding is difficult because query-relevant evidence is often sparse and distributed across distant temporal segments. VideoPro addresses this with a coarse-to-fine analysis pipeline that dynamically routes each query to either native VideoLLM reasoning or multi-step visual program reasoning, and performs self-refinement when execution fails or prediction confidence is low.

<img alt="image" src="docs/static/images/teaser.png" />

### ✨ Highlights:

- **Adaptive Query-level Routing**: Dynamically routes each query to either native VideoLLM reasoning for simple or high-confidence questions, or multi-step visual program reasoning for complex long-range queries.

- **Executable Visual Programs**: The model generates and executes structured Python programs using a rich video module library, enabling precise temporal grounding and fine-grained visual analysis across long videos.

- **Three-Mode Self-Refinement**:
  - *Native refinement*: refines low-confidence direct answers from native VideoLLM
  - *Bug fix*: fixes failed programs using runtime error logs
  - *Program refinement*: improves low-confidence program reasoning outputs

- **General Video Module Library**: A rich set of callable APIs available within visual programs, including multimodal retrieval (`get_informative_clips`), temporal localization (`trim_frames`, `trim_around`, `trim_before`, `trim_after`), object detection (`detect_object`), frame extraction (`extract_frames`), subtitle-based retrieval (`get_subtitle_hints`), and multi-choice QA (`query_mc`, `query_native`, `query_frames`).

- **Two-Stage Training**: Stage 1 — Supervised Fine-Tuning (SFT) on 7,489 program reasoning samples; Stage 2 — Group Relative Policy Optimization (GRPO) on 6,009 self-refinement samples covering all three refinement modes.

<img alt="image" src="docs/static/images/pipeline.png" />

<img alt="image" src="docs/static/images/results.png" />


## 📁 Project Structure

```text
VideoPro/
├── src/
│   ├── run.py                  # End-to-end inference pipeline (entry point)
│   ├── generate_code.py        # Step 1: Generate visual program via VLM
│   ├── execute_code.py         # Step 2: Execute the generated program
│   ├── refine_code.py          # Step 3: Self-refine on failure or low confidence
│   └── utils/
│       ├── runtime.py          # Stable runtime API layer for generated programs
│       ├── analysis.py         # AnalysisManager: QA, detection, temporal trim APIs
│       ├── retriever.py        # RetrievalManager: LanguageBind-based clip retrieval
│       └── video_utils.py      # Video splitting, frame extraction, subtitle parsing
├── scripts/
│   ├── train.sh                # Training commands (SFT + GRPO)
│   └── deploy.sh               # Model deployment command
├── dataset/
│   ├── train_sft.jsonl         # SFT training data
│   └── train_grpo.jsonl        # GRPO training data
├── models/                     # Model checkpoints (downloaded here)
├── clips/                      # Video clip cache (auto-created at inference)
├── docs/                       # Project page (GitHub Pages)
│   ├── index.html
│   └── static/images/
├── requirements.txt
└── README.md
```


## 🔥 Set Up Environment

```bash
conda create -n videopro python=3.10 -y
conda activate videopro
pip install -r requirements.txt
pip install flash_attn==2.8.3 --no-build-isolation
```

The inference pipeline also assumes:

- `ffmpeg` and `ffprobe` are available in `PATH`
- the served VLM is exposed through an OpenAI-compatible endpoint
- local model assets used by `LanguageBind`, `BGE-M3`, and `Grounding DINO` are available under `models/`


## 🔧 Model Preparation

Download the model from Hugging Face:

```bash
huggingface-cli download --resume-download zapqqqwe/videopro_grpo \
    --local-dir ./models/videopro
```

The model is based on **Qwen3-VL**.


## 🚀 Inference

### Step 1: Deploy the Model

```bash
bash scripts/deploy.sh
```

By default the deployment script serves the model at:

- `VIDEOPRO_API_BASE=http://0.0.0.0:8007/v1`
- `VIDEOPRO_MODEL=qwen3vl`

If needed, you can override them before running inference:

```bash
export VIDEOPRO_API_BASE="http://0.0.0.0:8007/v1"
export VIDEOPRO_MODEL="qwen3vl"
```

### Step 2: End-to-End Pipeline with `run.py`

`src/run.py` is the main entry point of the project. It is responsible for:

1. calling `generate_code.py` to generate the initial visual program
2. calling `execute_code.py` to execute the generated code with the packaged video APIs
3. checking whether execution failed or confidence is below the threshold
4. calling `refine_code.py` when refinement is needed
5. rerunning the refined program and returning the final answer

Use the end-to-end script `src/run.py` for the full pipeline `generate -> execute -> refine`:

```bash
python src/run.py \
    --video /path/to/video.mp4 \
    --question "What is the person doing in the video?" \
    --choices "Cooking in the kitchen" "Playing guitar" \
              "Riding a bicycle" "Swimming in a pool" \
    --clip-save-folder ./clips \
    --clip-duration 10 \
    --confidence-threshold 0.75 \
    --max-refine-rounds 1
```

If your test file is already stored locally, for example `test/video.mp4`, you can use it directly:

```bash
python src/run.py \
    --video test/video.mp4 \
    --question "What is the person doing in the video?" \
    --choices "Cooking in the kitchen" "Playing guitar" \
              "Riding a bicycle" "Swimming in a pool"
```

Output:

```text
[1/3] Generating visual program ...
[2/3] Executing visual program ...
[3/3] No refinement needed.

Answer: B (confidence=0.83)
```

Optionally save the result to JSON:

```bash
python src/run.py --video ... --question ... --choices ... --output result.json
```

### `run.py` Arguments

```bash
python src/run.py \
    --video /path/to/video.mp4 \
    --question "your question" \
    --choices "choice 1" "choice 2" "choice 3" "choice 4" \
    --clip-save-folder ./clips \
    --clip-duration 10 \
    --workers 8 \
    --confidence-threshold 0.75 \
    --max-refine-rounds 1 \
    --model qwen3vl \
    --output result.json
```

Main arguments:

- `--video`: input video path
- `--question`: multiple-choice question
- `--choices`: answer options
- `--clip-save-folder`: where 10-second clips are cached
- `--clip-duration`: clip length for splitting and retrieval, default `10`
- `--workers`: parallel workers for clip preparation
- `--confidence-threshold`: if final confidence is below this threshold, refinement is triggered
- `--max-refine-rounds`: maximum number of refinement retries
- `--model`: served VLM name
- `--output`: optional JSON output path
- `--quiet`: suppress verbose logs


## How It Works Internally

```text
Video + Question
      │
      ▼
generate_code.py         ← VLM generates <planning> + <code>
      │
      │  code_string
      ▼
execute_code.py          ← Video split into 10s clips → LanguageBind embeddings
      │                     → execute_command() runs with full API runtime
      │
      ├─ execution failed?      ──► refine_code.py (bug fix)
      ├─ low confidence native? ──► refine_code.py (native refinement)
      ├─ low confidence program? ─► refine_code.py (program refinement)
      └─ success                 ──► return answer
```

**Video processing flow inside `execute_code.py`:**

1. The input video is first split into 10-second clips and cached under `clip_save_folder/<video_id>/`.
2. `RetrievalManager` reuses those cached clips to build **LanguageBind** embeddings and text/video retrieval indices.
3. Top-k clips are retrieved by semantic similarity and passed to the generated visual program.
4. The program calls runtime APIs from `AnalysisManager` and `RetrievalManager` to reason about the video.
5. If execution fails, or the returned confidence is below the threshold, `refine_code.py` rewrites the program and the executor reruns it.


## 📑 Video Module Library

The following APIs are injected into the runtime environment and can be called directly inside the generated `execute_command` function:

| API | Module | Description |
|---|---|---|
| `query_native(video_path, question, choices)` | Analysis | Direct VideoLLM answer with confidence score |
| `query_mc(frames, question, choices)` | Analysis | Multi-choice QA over sampled frames |
| `query_frames(frames, question)` | Analysis | Open-ended QA over sampled frames |
| `query_yn(frames, question)` | Analysis | Yes/No QA over sampled frames |
| `get_informative_clips(video_path, query, top_k=3, total_duration=None)` | Retrieval | Retrieve top-k semantically relevant clips via LanguageBind |
| `extract_frames(video_path, num_frames)` | Video | Uniformly sample frames from a video or clip |
| `trim_frames(video_path, start, end)` | Analysis | Extract frames from a time range `[start, end]` |
| `trim_around(video_path, timestamp, intervals)` | Analysis | Extract frames centered around a timestamp |
| `trim_before(video_path, timestamp, intervals)` | Analysis | Extract frames in the window before a timestamp |
| `trim_after(video_path, timestamp, intervals)` | Analysis | Extract frames in the window after a timestamp |
| `detect_object(frame, text)` | Analysis | Zero-shot object detection with Grounding DINO |
| `get_subtitle_hints(video_path, question, choices, duration)` | Analysis | Retrieve and summarize relevant subtitles via BGE-M3 |
| `crop(frame, box)` | Analysis | Crop a frame to a given bounding box |
| `crop_left/right/top/bottom(frame)` | Analysis | Spatial half-crop of a frame |
| `make_result(answer, confidence, raw_output, **metadata)` | Runtime | Return a structured result for the executor |

All answer-producing APIs return a unified structure:

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


## 🧪 Useful Commands

### Generate only

```bash
python src/generate_code.py \
    --video /path/to/video.mp4 \
    --question "What is the person doing in the video?" \
    --choices "Cooking in the kitchen" "Playing guitar" \
              "Riding a bicycle" "Swimming in a pool" \
    --output generation.json \
    --output-code generation.py
```

### Execute only

```bash
python src/execute_code.py \
    --video /path/to/video.mp4 \
    --question "What is the person doing in the video?" \
    --choices "Cooking in the kitchen" "Playing guitar" \
              "Riding a bicycle" "Swimming in a pool" \
    --code-file generation.py \
    --clip-save-folder ./clips \
    --clip-duration 10 \
    --output execute_result.json
```

### Refine only

```bash
python src/refine_code.py \
    --video /path/to/video.mp4 \
    --question "What is the person doing in the video?" \
    --choices "Cooking in the kitchen" "Playing guitar" \
              "Riding a bicycle" "Swimming in a pool" \
    --code-file generation.py \
    --result-json execute_result.json \
    --confidence-threshold 0.75 \
    --output refinement.json \
    --output-code refined.py
```


## 💻 Training

### Stage 1: Supervised Fine-Tuning (SFT)

```bash
FPS_MAX_FRAMES=64 VIDEO_MAX_PIXELS=50176 \
export NPROC_PER_NODE=8
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift sft \
    --model ./models/videopro \
    --train_type lora \
    --dataset dataset/train_sft.jsonl \
    --load_from_cache_file true \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 64 \
    --lora_alpha 16 \
    --freeze_vit True \
    --target_modules all-linear \
    --gradient_accumulation_steps 1 \
    --eval_steps 50 \
    --save_steps 900 \
    --save_total_limit 5 \
    --logging_steps 5 \
    --output_dir ./models/sft \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --use_chat_template False \
    --max_length 200000
```

Training data (`dataset/train_sft.jsonl`): 7,489 program reasoning samples covering native mode and visual program mode.

### Stage 2: Group Relative Policy Optimization (GRPO)

```bash
FPS_MAX_FRAMES=64 \
VIDEO_MAX_PIXELS=50176 \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
NPROC_PER_NODE=4 \
swift rlhf \
    --rlhf_type grpo \
    --model ./models/sft/checkpoint-merged \
    --train_type lora \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.75 \
    --vllm_tensor_parallel_size 4 \
    --dataset dataset/train_grpo.jsonl \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --eval_steps 1000 \
    --save_steps 500 \
    --learning_rate 1e-6 \
    --save_total_limit 5 \
    --logging_steps 5 \
    --output_dir ./models/grpo \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --max_completion_length 4096 \
    --reward_funcs coderm \
    --external_plugins plugin.py \
    --num_generations 8 \
    --temperature 1.0 \
    --log_completions true \
    --async_generate false \
    --move_model_batches 16 \
    --offload_optimizer true \
    --offload_model true \
    --sleep_level 0
```




## 📧 Contact

If you have any comments or questions, please open a new issue or feel free to contact [Chenglin Li](https://scholar.google.com/citations?user=7LlS58IAAAAJ&hl=zh-CN).


## ⭐ Citation

```bibtex
@article{li2025videopro,
  title={VideoPro: Adaptive Program Reasoning for Long Video Understanding},
  author={Li, Chenglin and Han, Feng and Wang, Yikun and Li, Ruilin and Dong, Shuai and Hou, Haowen and Li, Haitao and Chen, Qianglong and Tao, Feng and Tong, Jingqi and others},
  journal={arXiv preprint arXiv:2509.17743},
  year={2025}
}
```
