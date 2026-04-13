import math
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from decord import VideoReader, cpu
from FlagEmbedding import BGEM3FlagModel
from languagebind import (
    LanguageBind,
    LanguageBindVideoTokenizer,
    to_device,
    transform_dict,
)
from moviepy import VideoFileClip

from video_utils import (
    extract_caption,
    extract_subtitles,
    parse_and_sort_file_paths,
    split_video_to_clips,
)


class RetrievalManager:
    def __init__(
        self,
        args=None,
        batch_size: int = 1,
        clip_save_folder: Optional[str] = None,
        clip_duration: int = 10,
        dataset_folder: str = "dataset/CG-Bench",
        retrievl_device: str = "cuda:0",
    ):
        self.args = args
        self.batch_size = batch_size
        self.clip_save_folder = clip_save_folder
        self.clip_duration = clip_duration
        self.dataset_folder = dataset_folder
        self.retriever_type = "large"
        self.device = retrievl_device if retrievl_device else "cuda:0"

        video_model_path = "models/LanguageBind_Video_FT"
        clip_type = {
            "video": "models/LanguageBind_Video_FT",
            "image": "models/LanguageBind_Image",
        }

        self.model = LanguageBind(clip_type=clip_type, device="cuda").to("cuda")
        self.model.eval()

        self.text_retriever = BGEM3FlagModel(
            "models/bge-m3",
            use_fp16=True,
            devices=["cuda"],
        )

        self.tokenizer = LanguageBindVideoTokenizer.from_pretrained(video_model_path)
        self.modality_transform = {
            key: transform_dict[key](self.model.modality_config[key])
            for key in clip_type.keys()
        }

        self.clip_embs_cache: Dict[str, Tuple[List[str], torch.Tensor]] = {}
        self.frame_embs_cache: Dict[str, Any] = {}

    def load_model_to_device(self, device):
        self.model.to(device)

        def recursive_to(module):
            for name, attr in module.__dict__.items():
                if isinstance(attr, torch.nn.Module):
                    attr.to(device)
                    recursive_to(attr)
                elif isinstance(attr, torch.Tensor):
                    setattr(module, name, attr.to(device))
                elif isinstance(attr, (list, tuple)):
                    new_attrs = []
                    for item in attr:
                        if isinstance(item, torch.nn.Module):
                            item.to(device)
                            recursive_to(item)
                        elif isinstance(item, torch.Tensor):
                            item = item.to(device)
                        new_attrs.append(item)
                    setattr(module, name, type(attr)(new_attrs))

        recursive_to(self.model)

    def load_model_to_cpu(self):
        self.device = torch.device("cpu")
        self.load_model_to_device(torch.device("cpu"))

    def load_model_to_gpu(self, gpu_id: int = 0):
        _ = gpu_id
        self.device = "cuda"
        self.model.to("cuda")

    def format_time(self, seconds: float) -> str:
        mins, secs = divmod(seconds, 60)
        hours, mins = divmod(mins, 60)
        return f"{int(hours):02d}-{int(mins):02d}-{int(secs):02d}"

    def parse_time(self, time_str: str) -> int:
        hours, mins, secs = map(int, time_str.split("-"))
        return hours * 3600 + mins * 60 + secs

    def cut_video(
        self,
        video_path: str,
        clip_save_folder: Optional[str] = None,
        total_duration: int = -1,
    ) -> List[str]:
        _ = total_duration

        if clip_save_folder is None:
            raise ValueError("clip_save_folder must not be None.")

        valid_clip_paths = set()
        os.makedirs(clip_save_folder, exist_ok=True)

        with VideoFileClip(video_path) as clip:
            duration = clip.duration

        chunk_number = math.ceil(duration / self.clip_duration)

        total_video_clip_paths = []
        for i in range(chunk_number):
            start_time = self.clip_duration * i
            end_time = start_time + self.clip_duration
            output_filename = (
                f"clip_{i}_{self.format_time(start_time)}_to_{self.format_time(end_time)}.mp4"
            )
            total_video_clip_paths.append(os.path.join(clip_save_folder, output_filename))

        if os.path.exists(clip_save_folder):
            path_list = os.listdir(clip_save_folder)
            for clip_name in path_list:
                clip_path = os.path.join(clip_save_folder, clip_name)
                try:
                    VideoReader(clip_path, ctx=cpu(0), num_threads=1)
                    valid_clip_paths.add(clip_path)
                    if clip_path in total_video_clip_paths:
                        total_video_clip_paths.remove(clip_path)
                except Exception:
                    try:
                        os.remove(clip_path)
                    except OSError:
                        pass

            return sorted(
                valid_clip_paths,
                key=lambda x: int(os.path.basename(x).split("_")[1]),
            )

        return sorted(
            valid_clip_paths,
            key=lambda x: int(os.path.basename(x).split("_")[1]),
        )

    def save_clip(
        self,
        clip: List[np.ndarray],
        clip_save_folder: str,
        clip_index: int,
        start_time: float,
        end_time: float,
        fps: float,
    ) -> str:
        start_time_str = self.format_time(start_time)
        end_time_str = self.format_time(end_time)
        os.makedirs(clip_save_folder, exist_ok=True)

        clip_path = os.path.join(
            clip_save_folder,
            f"clip_{clip_index}_{start_time_str}_to_{end_time_str}.mp4",
        )

        height, width, _ = clip[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))

        for frame in clip:
            out.write(frame)
        out.release()

        return clip_path

    def _embedding_dir(self) -> str:
        path = os.path.join(
            self.dataset_folder,
            "embeddings",
            str(self.clip_duration),
            self.retriever_type,
        )
        os.makedirs(path, exist_ok=True)
        return path

    def _subtitle_embedding_dir(self) -> str:
        path = os.path.join(
            self.dataset_folder,
            "embeddings",
            "subtitle",
            self.retriever_type,
        )
        os.makedirs(path, exist_ok=True)
        return path

    def _caption_embedding_dir(self) -> str:
        path = os.path.join(self.dataset_folder, "embeddings", "caption")
        os.makedirs(path, exist_ok=True)
        return path

    def _video_name(self, video_path: str) -> str:
        return Path(video_path).stem

    def _clip_cache_folder(self, video_path: str) -> str:
        if self.clip_save_folder is None:
            raise ValueError("clip_save_folder must not be None.")
        return os.path.join(self.clip_save_folder, self._video_name(video_path))

    def _load_pickle(self, path: str):
        with open(path, "rb") as f:
            return pickle.load(f)

    def _save_pickle(self, obj: Any, path: str):
        with open(path, "wb") as f:
            pickle.dump(obj, f)

    @torch.no_grad()
    def calculate_video_clip_embedding(
        self,
        video_path: str,
        folder_path: Optional[str] = None,
        total_duration: Optional[int] = None,
    ) -> Tuple[List[str], torch.Tensor]:
        _ = folder_path

        video_name = self._video_name(video_path)
        embedding_dir = self._embedding_dir()

        embedding_path = os.path.join(embedding_dir, f"{video_name}.pkl")
        clip_path = os.path.join(embedding_dir, f"{video_name}_clip_paths.pkl")

        if os.path.exists(embedding_path) and os.path.exists(clip_path):
            video_paths = self._load_pickle(clip_path)
            total_embeddings = self._load_pickle(embedding_path).cpu()

            if total_duration is None or total_duration < 0:
                return video_paths, total_embeddings

            min_expected = max(0, total_duration // self.clip_duration - 3)
            if len(video_paths) > min_expected:
                print("existing embedding")
                return video_paths, total_embeddings
            else:
                print(
                    "embedding exists but clip count seems insufficient:",
                    video_path,
                    len(video_paths),
                    total_duration,
                )

        clip_dir = self._clip_cache_folder(video_path)
        video_paths = self.cut_video(video_path, clip_dir, total_duration or -1)

        try:
            video_paths = split_video_to_clips(
                video_path,
                clip_dir,
                clip_duration=self.clip_duration,
            )[0]
            print("split videos")
        except Exception:
            print("error, ffmpeg:", video_path)
            video_paths = [video_path]

        if not video_paths:
            raise ValueError(f"No valid clips found for video: {video_path}")

        total_embeddings = []
        valid_video_paths = []

        for clip_path_item in video_paths:
            try:
                inputs = {
                    "video": to_device(
                        self.modality_transform["video"](clip_path_item),
                        self.device,
                    )
                }
                with torch.no_grad():
                    embeddings = self.model(inputs)

                valid_video_paths.append(clip_path_item)
                total_embeddings.append(embeddings["video"])
            except Exception as exc:
                print(exc)

            torch.cuda.empty_cache()

        if not total_embeddings:
            raise ValueError(f"Failed to compute clip embeddings for {video_path}")

        total_embeddings_tensor = torch.cat(total_embeddings, dim=0)
        self._save_pickle(total_embeddings_tensor, embedding_path)
        self._save_pickle(valid_video_paths, clip_path)

        return valid_video_paths, total_embeddings_tensor

    @torch.no_grad()
    def calculate_text_embedding(
        self,
        text: str,
        video_path: Optional[str] = None,
        flag_save_embedding: bool = True,
    ) -> torch.Tensor:
        embedding_path = None

        if flag_save_embedding:
            if video_path is None:
                raise ValueError("video_path is required when flag_save_embedding=True")
            video_name = self._video_name(video_path)
            embedding_path = os.path.join(
                self._subtitle_embedding_dir(),
                f"{video_name}_subtitle.pkl",
            )

            try:
                embeddings = self._load_pickle(embedding_path)
                return embeddings
            except Exception:
                pass

        inputs = {
            "language": to_device(
                self.tokenizer(
                    text,
                    max_length=77,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ),
                self.device,
            )
        }

        with torch.no_grad():
            embeddings = self.model(inputs)

        language_embedding = embeddings["language"]

        if flag_save_embedding and embedding_path is not None:
            self._save_pickle(language_embedding, embedding_path)

        torch.cuda.empty_cache()
        return language_embedding

    def _get_cached_clip_embeddings(
        self,
        video_path: str,
        total_duration: int = -1,
    ) -> Tuple[List[str], torch.Tensor]:
        if video_path not in self.clip_embs_cache:
            if len(self.clip_embs_cache) > 1:
                self.clip_embs_cache = {}

            video_clip_paths, clip_embs = self.calculate_video_clip_embedding(
                video_path=video_path,
                folder_path=self._embedding_dir(),
                total_duration=total_duration,
            )

            if isinstance(clip_embs, dict):
                clip_embs = clip_embs["video"]

            clip_embs = clip_embs.cpu()
            self.clip_embs_cache[video_path] = (video_clip_paths, clip_embs)

        return self.clip_embs_cache[video_path]

    @staticmethod
    def _normalize_embedding(embedding: torch.Tensor) -> torch.Tensor:
        return embedding / embedding.norm(p=2, dim=1, keepdim=True)

    @staticmethod
    def _validate_retrieval_mode(
        top_k: int,
        similarity_threshold: float,
        topk_similarity: float,
    ):
        is_topk = top_k != 0 and similarity_threshold == -100 and topk_similarity == 0
        is_threshold = top_k == 0 and similarity_threshold != -100 and topk_similarity == 0
        is_topk_similarity = top_k == 0 and similarity_threshold == -100 and topk_similarity != 0

        if not (is_topk or is_threshold or is_topk_similarity):
            raise ValueError(
                "Only one of top_k, similarity_threshold, or topk_similarity should be assigned."
            )

    def get_informative_clips(
        self,
        query: str,
        video_path: str,
        top_k: int = 0,
        similarity_threshold: float = -100,
        topk_similarity: float = 0,
        total_duration: int = -1,
        return_score: bool = False,
    ):
        _ = return_score

        torch.cuda.empty_cache()

        if ".mp4" not in video_path:
            video_path, query = query, video_path

        self._validate_retrieval_mode(top_k, similarity_threshold, topk_similarity)

        effective_top_k = top_k
        if similarity_threshold != -100 or topk_similarity != 0:
            effective_top_k = 100

        q_emb = self.calculate_text_embedding(query, flag_save_embedding=False).cpu()
        q_emb = self._normalize_embedding(q_emb)

        video_clip_paths, clip_embs = self._get_cached_clip_embeddings(
            video_path,
            total_duration,
        )
        clip_embs = self._normalize_embedding(clip_embs)

        similarities = torch.matmul(q_emb, clip_embs.T)
        top_k_indices = similarities[0].argsort(descending=True)[:effective_top_k].tolist()

        results = []
        for idx in top_k_indices:
            sim_score = similarities[0][idx].item()
            if sim_score > similarity_threshold:
                results.append((video_clip_paths[idx], sim_score))

        torch.cuda.empty_cache()

        if top_k == 0:
            results = results[:10]

        return parse_and_sort_file_paths(results)

    @torch.no_grad()
    def get_informative_clips_with_video_query(
        self,
        query: str,
        query_video_path: str,
        video_path: str,
        top_k: int = 0,
        similarity_threshold: float = -100,
        topk_similarity: float = 0,
        total_duration: int = -1,
        return_score: bool = False,
    ):
        _ = return_score

        torch.cuda.empty_cache()
        self._validate_retrieval_mode(top_k, similarity_threshold, topk_similarity)

        effective_top_k = top_k
        if similarity_threshold != -100 or topk_similarity != 0:
            effective_top_k = 100

        text_emb = self.calculate_text_embedding(query, flag_save_embedding=False).cpu()
        text_emb = self._normalize_embedding(text_emb)

        inputs = {
            "video": to_device(
                self.modality_transform["video"](query_video_path),
                self.device,
            )
        }
        with torch.no_grad():
            q_emb = self.model(inputs)["video"].cpu()
        q_emb = self._normalize_embedding(q_emb)
        q_emb = q_emb + text_emb

        video_clip_paths, clip_embs = self._get_cached_clip_embeddings(
            video_path,
            total_duration,
        )
        clip_embs = self._normalize_embedding(clip_embs)

        similarities = torch.matmul(q_emb, clip_embs.T)
        top_k_indices = similarities[0].argsort(descending=True)[:effective_top_k].tolist()

        results = []
        for idx in top_k_indices:
            sim_score = similarities[0][idx].item()
            if sim_score > similarity_threshold:
                results.append((video_clip_paths[idx], sim_score))

        torch.cuda.empty_cache()

        if top_k == 0:
            results = results[:10]

        return results

    def get_clips_by_threshold(
        self,
        query: str,
        video_path: str,
        similarity_threshold: float = 0.5,
        max_candidates: int = 100,
        total_duration: int = -1,
    ):
        torch.cuda.empty_cache()

        q_emb = self.calculate_text_embedding(query, flag_save_embedding=False).cpu()
        q_emb = self._normalize_embedding(q_emb)

        video_clip_paths, clip_embs = self._get_cached_clip_embeddings(
            video_path,
            total_duration,
        )
        clip_embs = self._normalize_embedding(clip_embs)

        similarities = torch.matmul(q_emb, clip_embs.T)[0]
        top_idx = similarities.argsort(descending=True)[:max_candidates].tolist()

        results = []
        threshold = float(similarity_threshold)
        for idx in top_idx:
            sim_score = float(similarities[idx].item())
            if sim_score >= threshold:
                results.append((video_clip_paths[idx], sim_score))

        torch.cuda.empty_cache()
        return parse_and_sort_file_paths(results)

    @torch.no_grad()
    def get_informative_captions(
        self,
        query: str,
        video_path: str,
        top_k: int = 1,
        total_duration: int = -1,
        return_embeddings: bool = False,
        merge_sentence: bool = False,
        flag_save_embedding: int = 1,
    ):
        _ = total_duration
        _ = return_embeddings
        _ = merge_sentence

        q_emb = self.text_retriever.encode(
            query,
            batch_size=12,
            max_length=256,
        )["dense_vecs"]

        captions_with_time = extract_caption(video_path)
        captions = [x[2] for x in captions_with_time]

        caption_embeddings = None
        if flag_save_embedding:
            video_name = self._video_name(video_path)
            embedding_path = os.path.join(
                self._caption_embedding_dir(),
                f"{video_name}_caption.pkl",
            )
            try:
                caption_embeddings = self._load_pickle(embedding_path)
                if hasattr(caption_embeddings, "cpu"):
                    caption_embeddings = caption_embeddings.cpu()
            except Exception as exc:
                print(exc)

                caption_embeddings = self.text_retriever.encode(
                    captions,
                    batch_size=12,
                    max_length=256,
                )["dense_vecs"]
                self._save_pickle(caption_embeddings, embedding_path)
        else:
            caption_embeddings = self.text_retriever.encode(
                captions,
                batch_size=12,
                max_length=256,
            )["dense_vecs"]

        similarities = np.dot(q_emb, caption_embeddings.T).flatten()
        top_k_indices = np.argsort(similarities)[-top_k:][::-1].tolist()
        return [captions_with_time[i] for i in top_k_indices]

    @torch.no_grad()
    def get_informative_subtitles(
        self,
        video_path: str,
        query: str,
        top_k: int = 1,
        total_duration: int = -1,
        return_embeddings: bool = False,
        merge_sentence: bool = False,
        flag_save_embedding: int = 1,
    ):
        _ = total_duration
        _ = return_embeddings
        _ = merge_sentence

        subtitle_srt_1 = video_path.replace("videos", "subtitles").replace(".mp4", ".srt")
        subtitle_json = video_path.replace("videos", "subtitles").replace(".mp4", "_en.json")
        subtitle_srt_2 = video_path.replace("video", "subtitles").replace(".mp4", ".srt")

        if (
            not os.path.exists(subtitle_srt_1)
            and not os.path.exists(subtitle_json)
            and not os.path.exists(subtitle_srt_2)
        ):
            return ""

        q_emb = self.text_retriever.encode(
            query,
            batch_size=128,
            max_length=256,
        )["dense_vecs"]

        subtitles_with_time = extract_subtitles(video_path)
        subtitles = [x[2] for x in subtitles_with_time]

        subtitle_embeddings = None
        if flag_save_embedding:
            video_name = self._video_name(video_path)
            embedding_path = os.path.join(
                self._subtitle_embedding_dir(),
                f"{video_name}_subtitle.pkl",
            )
            try:
                subtitle_embeddings = self._load_pickle(embedding_path)
            except Exception as exc:
                print(exc)
                subtitle_embeddings = self.text_retriever.encode(
                    subtitles,
                    batch_size=12,
                    max_length=256,
                )["dense_vecs"]
                self._save_pickle(subtitle_embeddings, embedding_path)
        else:
            subtitle_embeddings = self.text_retriever.encode(
                subtitles,
                batch_size=12,
                max_length=256,
            )["dense_vecs"]

        similarities = np.dot(q_emb, subtitle_embeddings.T).flatten()
        top_k_indices = np.argsort(similarities)[-top_k:][::-1].tolist()

        result = [subtitles_with_time[i] for i in top_k_indices]
        intervals = [(row[0], row[1]) for row in result]
        return intervals


# Alias for backward compatibility
Retrieval_Manager = RetrievalManager
