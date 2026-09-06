import numpy as np

from openpi.policies import membench_policy


def _image(value: int) -> np.ndarray:
    return np.full((8, 8, 3), value, dtype=np.uint8)


def _data(images: dict[str, np.ndarray]) -> dict:
    return {
        "images": images,
        "state": np.zeros(30, dtype=np.float32),
        "actions": np.zeros((50, 13), dtype=np.float32),
        "prompt": "test",
    }


def test_two_camera_input_keeps_third_slot_masked() -> None:
    result = membench_policy.MemBenchInputs()(_data({"agentview_right": _image(1), "eye_in_hand": _image(2)}))

    np.testing.assert_array_equal(result["image"]["base_0_rgb"], _image(1))
    np.testing.assert_array_equal(result["image"]["left_wrist_0_rgb"], _image(2))
    np.testing.assert_array_equal(result["image"]["right_wrist_0_rgb"], _image(0))
    assert not result["image_mask"]["right_wrist_0_rgb"]


def test_three_camera_input_uses_third_slot() -> None:
    result = membench_policy.MemBenchInputs()(
        _data(
            {
                "agentview_right": _image(1),
                "eye_in_hand": _image(2),
                "agentview_left": _image(3),
            }
        )
    )

    np.testing.assert_array_equal(result["image"]["base_0_rgb"], _image(1))
    np.testing.assert_array_equal(result["image"]["left_wrist_0_rgb"], _image(2))
    np.testing.assert_array_equal(result["image"]["right_wrist_0_rgb"], _image(3))
    assert result["image_mask"]["right_wrist_0_rgb"]
