import dataclasses
from typing import ClassVar

import numpy as np

from openpi import transforms


@dataclasses.dataclass(frozen=True)
class MemBenchInputs(transforms.DataTransformFn):
    """Inputs for MobileMemBench-style LeRobot datasets.

    Expected inputs after repacking:
    - images: dict with "agentview_right" and "eye_in_hand", plus optional
      "agentview_left"
    - state: proprioceptive state, any dimension <= model action dim
    - actions: action chunks, any dimension <= model action dim
    - prompt: language instruction
    """

    REQUIRED_CAMERAS: ClassVar[tuple[str, ...]] = ("agentview_left", "agentview_right", "eye_in_hand")

    def __call__(self, data: dict) -> dict:
        in_images = data["images"]
        expected_cameras = self.REQUIRED_CAMERAS
        if unexpected := set(in_images) - set(expected_cameras):
            raise ValueError(f"Unexpected cameras {tuple(sorted(unexpected))}; expected a subset of {expected_cameras}")
        if missing := set(self.REQUIRED_CAMERAS) - set(in_images):
            raise ValueError(f"Missing required cameras {tuple(sorted(missing))}; got {tuple(in_images)}")

        inputs = {
            "image": {
                "base_0_rgb": _to_hwc_uint8(in_images["agentview_left"]),
                "left_wrist_0_rgb": _to_hwc_uint8(in_images["agentview_right"]),
                "right_wrist_0_rgb": _to_hwc_uint8(in_images["eye_in_hand"]),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
            "state": np.asarray(data["state"], dtype=np.float32),
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class MemBenchOutputs(transforms.DataTransformFn):
    """Outputs for MobileMemBench-style policies."""

    # Number of dataset action dimensions to expose to the runtime.  Historical
    # MemBench datasets contain a 13-dimensional control vector, while the
    # real-robot WR07 recording contains 14 arm joints plus 3 base velocities.
    # The model still predicts its fixed 32-dimensional Pi0.5 action vector;
    # this transform removes the zero-padded tail on the way back out.
    action_dim: int = 13

    def __call__(self, data: dict) -> dict:
        if "actions" not in data:
            return data
        return {"actions": np.asarray(data["actions"])[:, : self.action_dim]}


@dataclasses.dataclass(frozen=True)
class SliceState(transforms.DataTransformFn):
    """Truncate the state to its first ``keep_dim`` dimensions (drops trailing dims, e.g. eef pose).

    This runs before normalization and padding, so norm stats are computed on the
    reduced state and ``PadStatesAndActions`` later pads it back up to the model
    action dim.
    """

    keep_dim: int

    def __call__(self, data: dict) -> dict:
        if "state" in data:
            state = np.asarray(data["state"])
            data = {**data, "state": state[..., : self.keep_dim]}
        return data


def _to_hwc_uint8(image) -> np.ndarray:
    image = np.asarray(image)

    if image.ndim != 3:
        raise ValueError(f"Expected image with 3 dimensions, got shape {image.shape}")

    if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.transpose(image, (1, 2, 0))

    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0) * 255.0

    return image.astype(np.uint8)
