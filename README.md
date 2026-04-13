<div align="center">
    <h1 align="center">VideoPro: Adaptive Program Reasoning for Long Video Understanding
    </h1>

<a href="https://arxiv.org/abs/2509.17743">
<img src='https://img.shields.io/badge/arXiv-2509.17743-blue' alt='Paper PDF'></a>

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-VideoPro_Model-yellow)](https://huggingface.co/zapqqqwe/videopro_grpo)

</div>


## 🔥 News
- [2026/04/12] 🔥🔥 We release the code and model of **VideoPro**!


## Introduction

We propose <b>VideoPro</b>, a unified framework for long-video understanding with <b>adaptive reasoning</b> and <b>self-refinement</b>. Long-video understanding is difficult because query-relevant evidence is often sparse and distributed across distant temporal segments. VideoPro addresses this with a coarse-to-fine analysis pipeline that dynamically routes each query to either native VideoLLM reasoning or multi-step visual program reasoning, and performs self-refinement when execution fails or prediction confidence is low.

<img alt="image" src="docs/static/images/teaser.png" />

### ✨ Highlights:

- **Adaptive Query-level Routing**: Dynamically routes each query to either native VideoLLM reasoning (for simple or high-confidence questions) or multi-step visual program reasoning (for complex long-range queries).

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

```
VideoPro/
├── src/
│   ├── run.py                  # End-to-end inference pipeline (entry point)
│   ├── generate_code.py        # Step 1: Generate visual program via VLM
│   ├── execute_code.py         # Step 2: Execute the generated program
│   ├── refine_code.py          # Step 3: Self-refine on failure or low confidence
│   └── utils/
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


## 🔧 Model Preparation

Download the model from Hugging Face:

```bash
huggingface-cli download --resume-download zapqqqwe/videopro_grpo \
    --local-dir ./models/videopro
```

The model is based on **Qwen3-VL** (architecture: `Qwen3VLForConditionalGeneration`, hidden size 4096, 36 layers).


## 🚀 Inference

### Step 1: Deploy the Model

```bash
TORCH_SYMM_MEM_DISABLE_MULTICAST=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
FPS_MAX_FRAMES=64 \
VIDEO_MAX_PIXELS=50176 \
swift deploy \
    --model ./models/videopro \
    --infer_backend vllm \
    --torch_dtype bfloat16 \
    --port 8007 \
    --vllm_tensor_parallel_size 4 \
    --served_model_name "qwen3vl"
```

### Step 2: Run the Pipeline

Use the end-to-end script `src/run.py` for the full pipeline (generate → execute → refine):

```bash
python src/run.py \
    --video /path/to/video.mp4 \
    --question "What is the person doing in the video?" \
    --choices "A. Cooking in the kitchen" "B. Playing guitar" \
              "C. Riding a bicycle" "D. Swimming in a pool" \
    --clip_save_folder ./clips \
    --clip_duration 10
```

Output:
```
[1/3] Generating visual program ...
[2/3] Executing visual program ...
[3/3] No refinement needed.

✓ Final answer: 'B'
```

Optionally save the result to JSON:
```bash
python src/run.py --video ... --question ... --choices ... --output result.json
```

### How It Works Internally

```
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
      ├─ success & native? ──► refine_code.py (native refinement)
      ├─ execution failed?  ──► refine_code.py (bug fix)
      └─ success & program  ──► return answer
```

**Video processing flow inside `execute_code.py`:**
1. Video is split into 10-second clips using `ffmpeg` (parallel workers)
2. `RetrievalManager` encodes each clip with **LanguageBind** (video encoder) and the query with **BGE-M3** (text encoder)
3. Top-k clips are retrieved by cosine similarity and passed to the generated visual program
4. The program calls APIs from `AnalysisManager` and `RetrievalManager` to reason about the video


## 📑 Video Module Library

The following APIs are injected into the runtime environment and can be called directly inside the generated `execute_command` function:

| API | Module | Description |
|---|---|---|
| `query_native(video_path, question, choices)` | Analysis | Direct VideoLLM answer with confidence score |
| `query_mc(frames, question, choices)` | Analysis | Multi-choice QA over sampled frames |
| `query_frames(frames, question)` | Analysis | Open-ended QA over sampled frames |
| `query_yn(frames, question)` | Analysis | Yes/No QA over sampled frames |
| `get_informative_clips(query, video_path, top_k)` | Retrieval | Retrieve top-k semantically relevant clips via LanguageBind |
| `extract_frames(video_path, num_frames)` | Video | Uniformly sample frames from a video or clip |
| `trim_frames(video_path, start, end)` | Analysis | Extract frames from a time range [start, end] |
| `trim_around(video_path, timestamp, intervals)` | Analysis | Extract frames centered around a timestamp |
| `trim_before(video_path, timestamp, intervals)` | Analysis | Extract frames in the window before a timestamp |
| `trim_after(video_path, timestamp, intervals)` | Analysis | Extract frames in the window after a timestamp |
| `detect_object(frame, text)` | Analysis | Zero-shot object detection with Grounding DINO |
| `get_subtitle_hints(video_path, question, choices, duration)` | Analysis | Retrieve and summarize relevant subtitles via BGE-M3 |
| `crop(frame, box)` | Analysis | Crop a frame to a given bounding box |
| `crop_left/right/top/bottom(frame)` | Analysis | Spatial half-crop of a frame |


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

Training data (`dataset/train_grpo.jsonl`): 6,009 self-refinement samples:
- 4,554 low-confidence native answer refinement samples
- 580 buggy program fix samples
- 875 low-confidence program refinement samples


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
