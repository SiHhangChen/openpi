import dataclasses
import pathlib
from typing import ClassVar

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from openpi.models import pi0_config
from openpi.training import config as training_config
from openpi.training import data_loader
from openpi.training import membench_v3_dataset


class _FakeMemBenchDataset:
    _DATA: ClassVar[dict] = {
        "task_a": {
            "tasks": {0: "task a zero", 1: "task a one"},
            "subtasks": {0: "subtask a zero", 1: "subtask a one"},
            "items": [
                {"task_index": 0, "subtask_index": 0},
                {"task_index": 1, "subtask_index": 1},
            ],
        },
        "task_b": {
            "tasks": {0: "task b zero"},
            "subtasks": {0: "subtask b zero"},
            "items": [{"task_index": 0, "subtask_index": 0}],
        },
    }

    def __init__(self, repo_id: str, *, action_horizon: int):
        del action_horizon
        data = self._DATA[repo_id]
        self.repo_id = repo_id
        self.tasks = data["tasks"]
        self.subtasks = data["subtasks"]
        self.has_subtask_indices = self.subtasks is not None
        self._items = data["items"]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> dict:
        return dict(self._items[index])


class _BrokenSubtaskDataset:
    repo_id = "broken"
    tasks: ClassVar = {0: "task"}
    subtasks = None
    has_subtask_indices = False

    def __len__(self) -> int:
        return 1


_REAL_MEMBENCH_DATASET = membench_v3_dataset.MemBenchV3Dataset


@pytest.fixture(autouse=True)
def _fake_source_dataset(monkeypatch):
    monkeypatch.setattr(membench_v3_dataset, "MemBenchV3Dataset", _FakeMemBenchDataset)


def test_concat_dataset_remaps_task_indices() -> None:
    dataset = membench_v3_dataset.ConcatMemBenchV3Dataset(("task_a", "task_b"), action_horizon=50, prompt_source="task")

    assert dataset.tasks == {0: "task a zero", 1: "task a one", 2: "task b zero"}
    assert dataset.subtasks is None
    assert [dataset[index]["task_index"] for index in range(len(dataset))] == [0, 1, 2]


def test_concat_dataset_remaps_subtask_indices() -> None:
    dataset = membench_v3_dataset.ConcatMemBenchV3Dataset(
        ("task_a", "task_b"), action_horizon=50, prompt_source="subtask"
    )

    assert dataset.tasks == {}
    assert dataset.subtasks == {0: "subtask a zero", 1: "subtask a one", 2: "subtask b zero"}
    assert [dataset[index]["subtask_index"] for index in range(len(dataset))] == [0, 1, 2]


def test_concat_dataset_sampling_weights_balance_sources() -> None:
    dataset = membench_v3_dataset.ConcatMemBenchV3Dataset(("task_a", "task_b"), action_horizon=50, prompt_source="task")

    np.testing.assert_allclose(dataset.sampling_weights, np.array([0.5, 0.5, 1.0]))
    np.testing.assert_allclose(dataset.sampling_weights[:2].sum(), dataset.sampling_weights[2:].sum())


def test_concat_dataset_rejects_missing_subtask_metadata(monkeypatch) -> None:
    monkeypatch.setattr(
        membench_v3_dataset,
        "MemBenchV3Dataset",
        lambda *args, **kwargs: _BrokenSubtaskDataset(),
    )

    with pytest.raises(ValueError, match="subtask metadata and per-frame subtask annotations"):
        membench_v3_dataset.ConcatMemBenchV3Dataset(("broken",), action_horizon=50, prompt_source="subtask")


def test_loads_subtask_metadata_from_tsv(tmp_path: pathlib.Path) -> None:
    meta_root = tmp_path / "meta"
    meta_root.mkdir()
    (meta_root / "subtasks.tsv").write_text("0\topen the drawer\n1\tclose the drawer\n", encoding="utf-8")

    assert membench_v3_dataset._load_subtask_metadata(meta_root) == {  # noqa: SLF001
        0: "open the drawer",
        1: "close the drawer",
    }


def test_loads_aligned_subtask_sidecar(tmp_path: pathlib.Path) -> None:
    data_path, sidecar_path, frame_table = _write_sidecar_fixture(tmp_path)

    result = membench_v3_dataset._load_subtask_sidecar(  # noqa: SLF001
        tmp_path, data_path, frame_table, sidecar_path
    )

    assert result.to_pylist() == [0, 0, 1]


def test_rejects_misaligned_subtask_sidecar(tmp_path: pathlib.Path) -> None:
    data_path, sidecar_path, frame_table = _write_sidecar_fixture(tmp_path, sidecar_indices=[0, 2, 1])

    with pytest.raises(ValueError, match="not aligned on index"):
        membench_v3_dataset._load_subtask_sidecar(  # noqa: SLF001
            tmp_path, data_path, frame_table, sidecar_path
        )


def _write_sidecar_fixture(
    root: pathlib.Path, *, sidecar_indices: list[int] | None = None
) -> tuple[pathlib.Path, pathlib.Path, pa.Table]:
    relative = pathlib.Path("chunk-000/file-000.parquet")
    data_path = root / "data" / relative
    sidecar_path = root / "subtask_indices" / relative
    data_path.parent.mkdir(parents=True)
    sidecar_path.parent.mkdir(parents=True)
    frame_table = pa.table(
        {
            "episode_index": [0, 0, 0],
            "frame_index": [0, 1, 2],
            "index": [0, 1, 2],
        }
    )
    sidecar = pa.table(
        {
            "episode_index": [0, 0, 0],
            "frame_index": [0, 1, 2],
            "index": sidecar_indices or [0, 1, 2],
            "subtask_index": pa.array([0, 0, 1], type=pa.int16()),
        }
    ).replace_schema_metadata(
        {
            b"membench.schema": b"membench.pi05_subtask_sidecar/v1",
            b"membench.source_data": b"data/chunk-000/file-000.parquet",
        }
    )
    pq.write_table(sidecar, sidecar_path)
    return data_path, sidecar_path, frame_table


@pytest.mark.parametrize(
    ("message", "frames_from_episode_end", "expected"),
    [
        ("Invalid frame index=2084 for streamIndex=0 numFrames=2084", 0, 2083),
        ("Invalid frame index=3261 for streamIndex=0 numFrames=3261", 1, 3260),
        ("Invalid frame index=3262 for streamIndex=0 numFrames=3261", 0, 3260),
        ("Invalid frame index=3261 for streamIndex=0 numFrames=3261", 2, None),
        ("video decoder failed", 0, None),
    ],
)
def test_torchcodec_fallback_frame_index(message: str, frames_from_episode_end: int, expected: int | None) -> None:
    assert (
        membench_v3_dataset._torchcodec_fallback_frame_index(  # noqa: SLF001
            RuntimeError(message), frames_from_episode_end
        )
        == expected
    )


def test_pyav_tail_fallback_handles_two_missing_frames(monkeypatch, tmp_path: pathlib.Path) -> None:
    calls: list[float] = []

    class _DecodedFrame:
        def squeeze(self, _: int) -> "_DecodedFrame":
            return self

        def numpy(self) -> np.ndarray:
            return np.zeros((3, 2, 2), dtype=np.float32)

    def _decode(_, timestamps, __, ___):
        calls.append(timestamps[0])
        if len(calls) < 3:
            raise AssertionError("query timestamps unexpectedly violate the tolerance")
        return _DecodedFrame()

    monkeypatch.setattr(membench_v3_dataset, "decode_video_frames", _decode)
    dataset = object.__new__(_REAL_MEMBENCH_DATASET)
    dataset.root = tmp_path
    dataset.video_path = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    dataset.fps = 20
    dataset.tolerance_s = 1e-4
    dataset.video_backend = "pyav"
    dataset._video_meta = {"camera": {0: (0, 0, 0.0)}}  # noqa: SLF001

    result = dataset._get_image("camera", 0, 1.0, 0)  # noqa: SLF001

    assert result.shape == (3, 2, 2)
    np.testing.assert_allclose(calls, [1.0, 0.95, 0.9])


def test_data_loader_applies_global_task_prompt(monkeypatch) -> None:
    monkeypatch.setattr(membench_v3_dataset, "is_membench_v3_dataset", lambda repo_id: True)
    config = training_config.DataConfig(
        repo_id="task_a",
        repo_ids=("task_a", "task_b"),
        prompt_from_task=True,
    )

    dataset = data_loader.create_torch_dataset(
        config,
        action_horizon=50,
        model_config=pi0_config.Pi0Config(pi05=True),
    )

    assert dataset[0]["prompt"] == "task a zero"
    assert dataset[2]["prompt"] == "task b zero"


def test_standalone_multi_config_uses_task_prompts(tmp_path) -> None:
    config = training_config.get_config("pi05_membench_multi_15_lora")
    config = dataclasses.replace(config, assets_base_dir=str(tmp_path))
    resolved = config.data.create(config.assets_dirs, config.model)

    assert len(resolved.repo_ids) == 15
    assert resolved.prompt_from_task
    assert not resolved.prompt_from_subtask
    assert resolved.asset_id == "membench_all_15_task"
    assert config.model.max_token_len == 200
    assert config.num_train_steps == 100_000


def test_prompt_v2_config_warm_starts_from_previous_params() -> None:
    config = training_config.get_config("memer_multi_15_prompt_v2")

    assert config.weight_loader.params_path == (
        "checkpoints/memer_multi_15/memer_multi_15_bs96_100000steps_torchcodec/70000/params"
    )
    assert config.data.assets.assets_dir == "./assets/memer_multi_15"
    assert config.data.assets.asset_id == "membench_multi_15"
    assert config.data.prompt_source == "subtask"
    assert config.lr_schedule.warmup_steps == 1_000
    assert config.lr_schedule.peak_lr == 1e-5
    assert config.lr_schedule.decay_steps == 30_000
    assert config.lr_schedule.decay_lr == 1e-6
    assert config.num_train_steps == 30_000
    assert config.ema_decay is None
