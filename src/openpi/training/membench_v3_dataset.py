import json
import os
import pathlib
from typing import SupportsIndex

import numpy as np
import polars as pl

from lerobot.common.datasets.video_utils import decode_video_frames


class MemBenchV3Dataset:
    """Local LeRobot v3 dataset reader for MemBench data.

    OpenPI currently pins an older LeRobot loader that expects v2.1 JSONL metadata.
    This reader handles the v3 parquet metadata layout directly.
    """

    def __init__(
        self,
        repo_id: str,
        *,
        action_horizon: int,
        root: str | pathlib.Path | None = None,
        tolerance_s: float = 1e-4,
        video_backend: str | None = None,
    ):
        self.repo_id = repo_id
        self.root = pathlib.Path(root) if root is not None else _default_lerobot_root() / repo_id
        self.action_horizon = action_horizon
        self.tolerance_s = tolerance_s
        self.video_backend = video_backend or os.getenv("OPENPI_MEMBENCH_VIDEO_BACKEND", "torchcodec")

        info_path = self.root / "meta" / "info.json"
        with info_path.open("r", encoding="utf-8") as f:
            self.info = json.load(f)

        self.fps = int(self.info["fps"])
        self.video_path = self.info["video_path"]
        self.camera_keys = [
            key for key, feature in self.info["features"].items() if feature["dtype"] in ("video", "image")
        ]
        self._image_shapes = {
            key: tuple(self.info["features"][key]["shape"])
            for key in self.camera_keys
            if "shape" in self.info["features"][key]
        }

        self.tasks = _load_tasks(self.root / "meta" / "tasks.parquet")
        self._load_episode_metadata()
        self._load_frame_data()

    def _load_episode_metadata(self) -> None:
        episode_files = sorted((self.root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
        if not episode_files:
            raise FileNotFoundError(f"No v3 episode metadata found under {self.root / 'meta' / 'episodes'}")

        episodes = pl.read_parquet([str(path) for path in episode_files])
        self._episode_from: dict[int, int] = {}
        self._episode_length: dict[int, int] = {}
        self._video_meta: dict[str, dict[int, tuple[int, int, float]]] = {key: {} for key in self.camera_keys}

        for row in episodes.iter_rows(named=True):
            episode_index = int(row["episode_index"])
            self._episode_from[episode_index] = int(row["dataset_from_index"])
            self._episode_length[episode_index] = int(row["length"])

            for key in self.camera_keys:
                prefix = f"videos/{key}"
                self._video_meta[key][episode_index] = (
                    int(row[f"{prefix}/chunk_index"]),
                    int(row[f"{prefix}/file_index"]),
                    float(row[f"{prefix}/from_timestamp"]),
                )

    def _load_frame_data(self) -> None:
        data_files = sorted((self.root / "data").glob("chunk-*/file-*.parquet"))
        if not data_files:
            raise FileNotFoundError(f"No v3 frame data found under {self.root / 'data'}")

        frame_data = pl.read_parquet(
            [str(path) for path in data_files],
            columns=["observation.state", "action", "timestamp", "episode_index", "task_index", "index"],
        ).sort("index")

        self._states = frame_data["observation.state"].to_numpy()
        self._actions = frame_data["action"].to_numpy()
        self._timestamps = frame_data["timestamp"].to_numpy()
        self._episode_indices = frame_data["episode_index"].to_numpy()
        self._task_indices = frame_data["task_index"].to_numpy()
        self._indices = frame_data["index"].to_numpy()

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: SupportsIndex) -> dict:
        idx = int(index.__index__())
        episode_index = int(self._episode_indices[idx])
        timestamp = float(self._timestamps[idx])
        action_indices = self._action_indices(idx, episode_index)

        item = {
            "observation.state": self._states[idx],
            "action": self._actions[action_indices],
            "timestamp": timestamp,
            "episode_index": episode_index,
            "task_index": int(self._task_indices[idx]),
            "index": int(self._indices[idx]),
        }

        for key in self.camera_keys:
            item[key] = self._get_image(key, episode_index, timestamp)

        return item

    def _action_indices(self, idx: int, episode_index: int) -> np.ndarray:
        episode_start = self._episode_from[episode_index]
        episode_length = self._episode_length[episode_index]
        offset = idx - episode_start
        offsets = np.minimum(offset + np.arange(self.action_horizon), episode_length - 1)
        return episode_start + offsets

    def _get_image(self, key: str, episode_index: int, timestamp: float) -> np.ndarray:
        if os.getenv("OPENPI_SKIP_IMAGE_DECODE") == "1":
            return np.zeros((3, 1, 1), dtype=np.float32)

        chunk_index, file_index, from_timestamp = self._video_meta[key][episode_index]
        video_path = self.root / self.video_path.format(
            video_key=key,
            chunk_index=chunk_index,
            file_index=file_index,
        )
        frames = decode_video_frames(
            video_path,
            [from_timestamp + timestamp],
            self.tolerance_s,
            self.video_backend,
        )
        return frames.squeeze(0).numpy()


def is_membench_v3_dataset(repo_id: str) -> bool:
    root = _default_lerobot_root() / repo_id
    return (root / "meta" / "tasks.parquet").is_file() and (root / "meta" / "episodes").is_dir()


def _default_lerobot_root() -> pathlib.Path:
    return pathlib.Path(os.getenv("HF_LEROBOT_HOME", "~/.cache/huggingface/lerobot")).expanduser()


def _load_tasks(tasks_path: pathlib.Path) -> dict[int, str]:
    tasks = pl.read_parquet(tasks_path)
    return {int(row["task_index"]): str(row["task"]) for row in tasks.iter_rows(named=True)}
