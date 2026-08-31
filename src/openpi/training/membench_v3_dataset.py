import bisect
from collections.abc import Sequence
import csv
import json
import os
import pathlib
import re
from typing import Literal, SupportsIndex

from lerobot.common.datasets.video_utils import decode_video_frames
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

# Columns needed from the per-frame data parquet files. Other columns (e.g. camera
# intrinsics/extrinsics) carry HuggingFace Arrow extension types that current Polars
# builds cannot parse, so we read these files with pyarrow (which falls back to the
# storage type) and only project the columns we need.
_FRAME_DATA_COLUMNS = (
    "observation.state",
    "action",
    "timestamp",
    "frame_index",
    "episode_index",
    "task_index",
    "index",
)

_SUBTASK_SIDECAR_COLUMNS = ("episode_index", "frame_index", "index", "subtask_index")
_SUBTASK_SIDECAR_SCHEMA = b"membench.pi05_subtask_sidecar/v1"

# A full audit of the 6,000 training-camera videos found four converted files
# whose encoded streams are one or two frames shorter than their frame tables.
_MAX_VIDEO_TAIL_FRAME_DEFICIT = 2


def _fixed_list_to_ndarray(column: pa.ChunkedArray) -> np.ndarray:
    """Convert a fixed_size_list Arrow column into a 2-D numpy array of shape (rows, k)."""
    return np.stack(column.to_numpy(zero_copy_only=False))


def _torchcodec_fallback_frame_index(error: RuntimeError, frames_from_episode_end: int) -> int | None:
    match = re.search(r"Invalid frame index=(\d+).*numFrames=(\d+)", str(error))
    if match is None:
        return None
    requested_index, num_frames = (int(value) for value in match.groups())
    if requested_index < num_frames:
        return None
    deficit = requested_index - num_frames + 1 + frames_from_episode_end
    if not 1 <= deficit <= _MAX_VIDEO_TAIL_FRAME_DEFICIT:
        return None
    return num_frames - 1


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
        # Optional allowlist: only decode the cameras the consuming policy actually uses
        # (e.g. OPENPI_MEMBENCH_CAMERA_KEYS="observation.images.robot0_agentview_right,observation.images.robot0_eye_in_hand").
        # Unused depth/left cameras are otherwise decoded for every frame, which is wasteful.
        if allowed := os.getenv("OPENPI_MEMBENCH_CAMERA_KEYS"):
            allowed_keys = {key.strip() for key in allowed.split(",") if key.strip()}
            self.camera_keys = [key for key in self.camera_keys if key in allowed_keys]
        self._image_shapes = {
            key: tuple(self.info["features"][key]["shape"])
            for key in self.camera_keys
            if "shape" in self.info["features"][key]
        }

        self.tasks = _load_tasks(self.root / "meta" / "tasks.parquet")
        self.subtasks = _load_subtask_metadata(self.root / "meta")
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

        # Read with pyarrow (not polars): the frame files contain HuggingFace Arrow
        # extension-type columns (camera intrinsics/extrinsics) that current Polars
        # builds reject even when projected out.
        data_schemas = [pq.read_schema(str(path)) for path in data_files]
        inline_subtask_flags = ["subtask_index" in schema.names for schema in data_schemas]
        if any(inline_subtask_flags) and not all(inline_subtask_flags):
            raise ValueError(f"Only some frame parquet files contain subtask_index under {self.root / 'data'}")

        has_inline_subtasks = all(inline_subtask_flags)
        sidecar_paths = None if has_inline_subtasks else _find_subtask_sidecars(self.root, data_files)
        frame_columns = [*_FRAME_DATA_COLUMNS, *(("subtask_index",) if has_inline_subtasks else ())]
        tables = []
        for data_path in data_files:
            table = pq.read_table(str(data_path), columns=frame_columns)
            if sidecar_paths is not None:
                subtask_indices = _load_subtask_sidecar(self.root, data_path, table, sidecar_paths[data_path])
                table = table.append_column("subtask_index", subtask_indices)
            tables.append(table)
        frame_data = pa.concat_tables(tables).sort_by("index")

        self._states = _fixed_list_to_ndarray(frame_data.column("observation.state"))
        self._actions = _fixed_list_to_ndarray(frame_data.column("action"))
        self._timestamps = frame_data.column("timestamp").to_numpy()
        self._episode_indices = frame_data.column("episode_index").to_numpy()
        self._task_indices = frame_data.column("task_index").to_numpy()
        self._subtask_indices = (
            frame_data.column("subtask_index").to_numpy() if "subtask_index" in frame_data.column_names else None
        )
        self._indices = frame_data.column("index").to_numpy()
        if self._subtask_indices is not None and self.subtasks is not None:
            unknown_indices = sorted(set(map(int, np.unique(self._subtask_indices))) - self.subtasks.keys())
            if unknown_indices:
                raise ValueError(
                    f"Subtask indices {unknown_indices} from {self.root} are missing from its subtask metadata"
                )

    def __len__(self) -> int:
        return len(self._indices)

    @property
    def has_subtask_indices(self) -> bool:
        return self._subtask_indices is not None

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
        if self._subtask_indices is not None:
            item["subtask_index"] = int(self._subtask_indices[idx])

        episode_last_index = self._episode_from[episode_index] + self._episode_length[episode_index] - 1
        frames_from_episode_end = episode_last_index - idx
        for key in self.camera_keys:
            item[key] = self._get_image(key, episode_index, timestamp, frames_from_episode_end)

        return item

    def _action_indices(self, idx: int, episode_index: int) -> np.ndarray:
        episode_start = self._episode_from[episode_index]
        episode_length = self._episode_length[episode_index]
        offset = idx - episode_start
        offsets = np.minimum(offset + np.arange(self.action_horizon), episode_length - 1)
        return episode_start + offsets

    def _get_image(self, key: str, episode_index: int, timestamp: float, frames_from_episode_end: int) -> np.ndarray:
        if os.getenv("OPENPI_SKIP_IMAGE_DECODE") == "1":
            return np.zeros(self._image_shapes.get(key, (224, 224, 3)), dtype=np.uint8)

        chunk_index, file_index, from_timestamp = self._video_meta[key][episode_index]
        video_path = self.root / self.video_path.format(
            video_key=key,
            chunk_index=chunk_index,
            file_index=file_index,
        )
        query_timestamp = from_timestamp + timestamp
        try:
            frames = decode_video_frames(
                video_path,
                [query_timestamp],
                self.tolerance_s,
                self.video_backend,
            )
        except AssertionError as error:
            # Some converted datasets contain one or two fewer video frames
            # than the parquet metadata at an episode boundary. Permit only
            # tail samples to reuse the nearest preceding video frame; all
            # interior synchronization errors remain fatal.
            if "violate the tolerance" not in str(error):
                raise
            max_fallback_frames = _MAX_VIDEO_TAIL_FRAME_DEFICIT - frames_from_episode_end
            if max_fallback_frames < 1:
                raise
            for fallback_frames in range(1, max_fallback_frames + 1):
                try:
                    frames = decode_video_frames(
                        video_path,
                        [query_timestamp - fallback_frames / self.fps],
                        self.tolerance_s,
                        self.video_backend,
                    )
                    break
                except AssertionError:
                    if fallback_frames == max_fallback_frames:
                        raise
        except RuntimeError as error:
            fallback_frame_index = _torchcodec_fallback_frame_index(error, frames_from_episode_end)
            if fallback_frame_index is None:
                raise

            # TorchCodec reports the encoded frame count in its bounds error.
            # Decode the actual final frame directly so a two-frame deficit
            # also succeeds for both affected logical samples.
            frames = decode_video_frames(
                video_path,
                [fallback_frame_index / self.fps],
                self.tolerance_s,
                self.video_backend,
            )
        return frames.squeeze(0).numpy()


class ConcatMemBenchV3Dataset:
    """Frame-balanced concatenation of local MemBench v3 datasets.

    Each source keeps its own episode/video indexing. Prompt ids are remapped to
    a global namespace so one task or subtask lookup table can be used after
    concatenation.
    """

    def __init__(
        self,
        repo_ids: Sequence[str],
        *,
        action_horizon: int,
        prompt_source: Literal["task", "subtask"] = "subtask",
    ):
        if not repo_ids:
            raise ValueError("repo_ids must not be empty")
        if prompt_source not in ("task", "subtask"):
            raise ValueError(f"Unsupported prompt source: {prompt_source}")
        self.repo_ids = tuple(repo_ids)
        self.prompt_source = prompt_source
        self.datasets = tuple(MemBenchV3Dataset(repo_id, action_horizon=action_horizon) for repo_id in self.repo_ids)
        self._ends: list[int] = []
        self._prompt_index_maps: list[dict[int, int]] = []
        total = 0
        global_prompts: dict[int, str] = {}
        global_prompt_index = 0
        for dataset in self.datasets:
            total += len(dataset)
            self._ends.append(total)
            if prompt_source == "task":
                source_prompts = dataset.tasks
            else:
                if dataset.subtasks is None or not dataset.has_subtask_indices:
                    raise ValueError(
                        f'MemBench dataset "{dataset.repo_id}" must contain both '
                        "subtask metadata and per-frame subtask annotations"
                    )
                source_prompts = dataset.subtasks

            prompt_index_map: dict[int, int] = {}
            for local_index, prompt in sorted(source_prompts.items()):
                prompt_index_map[local_index] = global_prompt_index
                global_prompts[global_prompt_index] = prompt
                global_prompt_index += 1
            self._prompt_index_maps.append(prompt_index_map)

        self.tasks = global_prompts if prompt_source == "task" else {}
        self.subtasks = global_prompts if prompt_source == "subtask" else None

    @property
    def sampling_weights(self) -> np.ndarray:
        """Per-frame weights that give every source task equal probability."""
        weights = np.empty(len(self), dtype=np.float64)
        start = 0
        for dataset in self.datasets:
            end = start + len(dataset)
            weights[start:end] = 1.0 / len(dataset)
            start = end
        return weights

    def __len__(self) -> int:
        return self._ends[-1]

    def __getitem__(self, index: SupportsIndex) -> dict:
        global_index = int(index.__index__())
        if global_index < 0:
            global_index += len(self)
        if global_index < 0 or global_index >= len(self):
            raise IndexError(global_index)
        source_index = bisect.bisect_right(self._ends, global_index)
        source_start = 0 if source_index == 0 else self._ends[source_index - 1]
        item = self.datasets[source_index][global_index - source_start]
        index_key = "task_index" if self.prompt_source == "task" else "subtask_index"
        local_prompt_index = int(item[index_key])
        try:
            item[index_key] = self._prompt_index_maps[source_index][local_prompt_index]
        except KeyError as error:
            raise ValueError(
                f"{index_key} {local_prompt_index} from dataset {self.repo_ids[source_index]} "
                f"is missing from its {self.prompt_source} metadata"
            ) from error
        return item


def is_membench_v3_dataset(repo_id: str) -> bool:
    root = _default_lerobot_root() / repo_id
    return (root / "meta" / "tasks.parquet").is_file() and (root / "meta" / "episodes").is_dir()


def _default_lerobot_root() -> pathlib.Path:
    return pathlib.Path(os.getenv("HF_LEROBOT_HOME", "~/.cache/huggingface/lerobot")).expanduser()


def _load_tasks(tasks_path: pathlib.Path) -> dict[int, str]:
    tasks = pl.read_parquet(tasks_path)
    return {int(row["task_index"]): str(row["task"]) for row in tasks.iter_rows(named=True)}


def _load_subtasks(subtasks_path: pathlib.Path) -> dict[int, str]:
    subtasks = pl.read_parquet(subtasks_path)
    return {int(row["subtask_index"]): str(row["subtask"]) for row in subtasks.iter_rows(named=True)}


def _load_subtask_metadata(meta_root: pathlib.Path) -> dict[int, str] | None:
    parquet_path = meta_root / "subtasks.parquet"
    tsv_path = meta_root / "subtasks.tsv"
    parquet_subtasks = _load_subtasks(parquet_path) if parquet_path.is_file() else None
    tsv_subtasks = _load_subtasks_tsv(tsv_path) if tsv_path.is_file() else None
    if parquet_subtasks is not None and tsv_subtasks is not None and parquet_subtasks != tsv_subtasks:
        raise ValueError(f"Subtask metadata differs between {parquet_path} and {tsv_path}")
    return parquet_subtasks if parquet_subtasks is not None else tsv_subtasks


def _load_subtasks_tsv(subtasks_path: pathlib.Path) -> dict[int, str]:
    subtasks: dict[int, str] = {}
    with subtasks_path.open("r", encoding="utf-8", newline="") as file:
        for line_number, row in enumerate(csv.reader(file, delimiter="\t"), start=1):
            if not row or all(not value.strip() for value in row):
                continue
            if len(row) != 2:
                raise ValueError(f"Expected two TSV columns at {subtasks_path}:{line_number}, got {len(row)}")
            try:
                subtask_index = int(row[0])
            except ValueError as error:
                raise ValueError(f"Invalid subtask index at {subtasks_path}:{line_number}: {row[0]!r}") from error
            subtask = row[1].strip()
            if not subtask:
                raise ValueError(f"Empty subtask label at {subtasks_path}:{line_number}")
            if subtask_index in subtasks:
                raise ValueError(f"Duplicate subtask index {subtask_index} at {subtasks_path}:{line_number}")
            subtasks[subtask_index] = subtask
    if not subtasks:
        raise ValueError(f"No subtasks found in {subtasks_path}")
    return subtasks


def _find_subtask_sidecars(
    dataset_root: pathlib.Path, data_files: Sequence[pathlib.Path]
) -> dict[pathlib.Path, pathlib.Path] | None:
    data_root = dataset_root / "data"
    sidecar_root = dataset_root / "subtask_indices"
    if not sidecar_root.is_dir():
        return None

    expected = {data_path.relative_to(data_root) for data_path in data_files}
    actual = {path.relative_to(sidecar_root) for path in sidecar_root.glob("chunk-*/file-*.parquet")}
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise ValueError(
            f"Subtask sidecars do not match frame parquet files under {dataset_root}: "
            f"missing={list(map(str, missing))}, unexpected={list(map(str, unexpected))}"
        )
    return {data_path: sidecar_root / data_path.relative_to(data_root) for data_path in data_files}


def _load_subtask_sidecar(
    dataset_root: pathlib.Path,
    data_path: pathlib.Path,
    frame_table: pa.Table,
    sidecar_path: pathlib.Path,
) -> pa.ChunkedArray:
    schema = pq.read_schema(str(sidecar_path))
    missing_columns = [column for column in _SUBTASK_SIDECAR_COLUMNS if column not in schema.names]
    if missing_columns:
        raise ValueError(f"Subtask sidecar {sidecar_path} is missing columns: {missing_columns}")
    if not pa.types.is_integer(schema.field("subtask_index").type):
        raise ValueError(
            f"subtask_index must be an integer in {sidecar_path}, got {schema.field('subtask_index').type}"
        )

    metadata = schema.metadata or {}
    if metadata.get(b"membench.schema") != _SUBTASK_SIDECAR_SCHEMA:
        raise ValueError(f"Unsupported or missing membench.schema in {sidecar_path}")
    expected_source = data_path.relative_to(dataset_root).as_posix()
    actual_source = metadata.get(b"membench.source_data", b"").decode("utf-8")
    if actual_source != expected_source:
        raise ValueError(f"Subtask sidecar {sidecar_path} references {actual_source!r}, expected {expected_source!r}")

    sidecar = pq.read_table(str(sidecar_path), columns=list(_SUBTASK_SIDECAR_COLUMNS))
    if sidecar.num_rows != frame_table.num_rows:
        raise ValueError(
            f"Subtask sidecar row count {sidecar.num_rows} != frame row count {frame_table.num_rows}: {sidecar_path}"
        )
    for column in ("episode_index", "frame_index", "index"):
        frame_values = frame_table.column(column).to_numpy(zero_copy_only=False)
        sidecar_values = sidecar.column(column).to_numpy(zero_copy_only=False)
        if not np.array_equal(frame_values, sidecar_values):
            raise ValueError(f"Subtask sidecar {sidecar_path} is not aligned on {column}")
    return sidecar.column("subtask_index")
