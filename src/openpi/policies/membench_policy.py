import dataclasses
from typing import ClassVar

import numpy as np

from openpi import transforms


@dataclasses.dataclass(frozen=True)
class MemBenchInputs(transforms.DataTransformFn):
    """Inputs for MobileMemBench-style LeRobot datasets.

    Expected inputs after repacking:
    - images: dict with "agentview_right" and "eye_in_hand"
    - state: proprioceptive state, any dimension <= model action dim
    - actions: action chunks, any dimension <= model action dim
    - prompt: language instruction
    """

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("agentview_right", "eye_in_hand")

    def __call__(self, data: dict) -> dict:
        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        base_image = _to_hwc_uint8(in_images["agentview_right"])
        wrist_image = _to_hwc_uint8(in_images["eye_in_hand"])

        inputs = {
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
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

    action_dim: int

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
