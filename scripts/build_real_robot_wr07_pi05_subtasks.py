#!/usr/bin/env python3
"""Build a WR07 LeRobot v3 view with three episode-level Pi0.5 prompts.

Source frame data and videos are linked, not copied or modified. The only new
training data are subtask metadata and frame-aligned Parquet sidecars.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


COLORS = ("brown", "white", "pink")
SIDECAR_SCHEMA = b"membench.pi05_subtask_sidecar/v1"


def link(source: Path, target: Path) -> None:
    target.symlink_to(os.path.relpath(source, target.parent), target_is_directory=source.is_dir())


def build(source: Path, annotations: Path, output: Path) -> dict:
    source = source.resolve(strict=True)
    annotations = annotations.resolve(strict=True)
    output = output.absolute()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if source == output or source in output.parents:
        raise ValueError("Output must be separate from the source dataset")

    info = json.loads((source / "meta/info.json").read_text())
    if info["codebase_version"] != "v3.0" or info["fps"] != 30:
        raise ValueError("Expected the WR07 real-robot LeRobot v3 recording at 30 FPS")
    episode_rows = {
        int(row["episode_index"]): row
        for path in (source / "meta/episodes").glob("chunk-*/file-*.parquet")
        for row in pq.read_table(path, columns=["episode_index", "length"]).to_pylist()
    }
    if len(episode_rows) != info["total_episodes"]:
        raise ValueError("Duplicate or missing episode metadata")

    paths = sorted((annotations / "teacher_raw").glob("episode_*/annotation.json"))
    labels_by_episode: dict[int, str] = {}
    colors_by_episode: dict[int, str] = {}
    annotation_hashes: dict[int, str] = {}
    for path in paths:
        raw = path.read_bytes()
        annotation = json.loads(raw)
        episode = annotation["episode_index"]
        if (type(episode) is not int or episode not in episode_rows or episode in labels_by_episode
                or path.parent.name != f"episode_{episode:06d}"
                or annotation.get("task_id") != "WR-07"):
            raise ValueError(f"Invalid annotation identity: {path}")
        stages = annotation.get("stages", [])
        if len(stages) != 1:
            raise ValueError(f"Expected exactly one color prompt in episode {episode}")
        stage = stages[0]
        if stage["start_frame"] != 0 or stage["end_frame"] != episode_rows[episode]["length"] - 1:
            raise ValueError(f"Annotation does not cover episode {episode}")
        color = stage["entity_memory"]["movable_item_1"]["source_container_color"]
        label = stage["active_subgoal"]
        if color not in COLORS or not isinstance(label, str) or not label.strip():
            raise ValueError(f"Invalid color prompt in episode {episode}")
        if label.count(color) != 2:
            raise ValueError(f"Prompt does not identify the source and target color: {episode}")
        if len(annotation.get("evidence", [])) != 1 or annotation["evidence"][0]["completion_frame"] != stage["end_frame"]:
            raise ValueError(f"Completion evidence disagrees with episode {episode}")
        labels_by_episode[episode] = label
        colors_by_episode[episode] = color
        annotation_hashes[episode] = hashlib.sha256(raw).hexdigest()
    if set(labels_by_episode) != set(episode_rows):
        raise ValueError(f"Missing annotations: {sorted(set(episode_rows) - set(labels_by_episode))}")

    label_by_color = {}
    for episode, label in labels_by_episode.items():
        color = colors_by_episode[episode]
        if color in label_by_color and label_by_color[color] != label:
            raise ValueError(f"Multiple prompt texts for {color}")
        label_by_color[color] = label
    if set(label_by_color) != set(COLORS):
        raise ValueError("All three WR07 colors must be present")
    index_by_color = {color: index for index, color in enumerate(COLORS)}

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temporary:
        staging = Path(temporary)
        for name in ("data", "videos", "images"):
            if (source / name).exists():
                link(source / name, staging / name)
        (staging / "meta").mkdir()
        for source_meta in (source / "meta").iterdir():
            if source_meta.name in ("subtasks.parquet", "subtasks.tsv"):
                continue
            link(source_meta, staging / "meta" / source_meta.name)
        pq.write_table(
            pa.table({"subtask_index": range(len(COLORS)),
                      "subtask": [label_by_color[color] for color in COLORS]}),
            staging / "meta/subtasks.parquet",
        )

        frame_counts = Counter()
        data_root = source / "data"
        for data_path in sorted(data_root.glob("chunk-*/file-*.parquet")):
            relative = data_path.relative_to(data_root)
            frame_table = pq.read_table(data_path, columns=["episode_index", "frame_index", "index"])
            episode_indices = frame_table["episode_index"].to_numpy()
            unique = np.unique(episode_indices)
            if len(unique) != 1:
                raise ValueError(f"Expected one episode per frame file: {data_path}")
            episode = int(unique[0])
            frames = frame_table["frame_index"].to_numpy()
            if not np.array_equal(frames, np.arange(episode_rows[episode]["length"])):
                raise ValueError(f"Noncontiguous frames in episode {episode}")
            frame_counts[episode] += frame_table.num_rows
            sidecar = frame_table.append_column(
                "subtask_index",
                pa.array(np.full(frame_table.num_rows, index_by_color[colors_by_episode[episode]], dtype=np.int64)),
            ).replace_schema_metadata({
                b"membench.schema": SIDECAR_SCHEMA,
                b"membench.source_data": f"data/{relative.as_posix()}".encode(),
            })
            sidecar_path = staging / "subtask_indices" / relative
            sidecar_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(sidecar, sidecar_path, compression="zstd")
        if frame_counts != Counter({ep: row["length"] for ep, row in episode_rows.items()}):
            raise ValueError("Sidecars do not cover every source frame exactly once")

        manifest = {
            "source": str(source), "annotation_root": str(annotations),
            "episodes": len(episode_rows), "frames": sum(frame_counts.values()),
            "sample_stride": 10,
            "sample_anchors": sum((row["length"] + 9) // 10 for row in episode_rows.values()),
            "prompt_by_color": label_by_color,
            "episodes_by_color": dict(Counter(colors_by_episode.values())),
            "annotation_sha256_by_episode": annotation_hashes,
        }
        (staging / "subtask_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        staging.rename(output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.annotations, args.output), indent=2))


if __name__ == "__main__":
    main()
