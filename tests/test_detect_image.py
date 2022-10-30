import numpy as np
from detect_image import find_bibs_on_image
import pytest


def test_bib_detection():
    result_array, _ = find_bibs_on_image(
        weights=r"runs/train/yolov7-bib-detection-v1/weights/best.pt",
        source=r"inference\images\snapshot_smaller.jpeg",
        nosave=True,
    )

    expected_result = np.array(
        [
            [383, 181, 426, 202, 0.38566, 0],
            [736, 259, 803, 289, 0.38563, 0],
            [544, 131, 589, 153, 0.32539, 0],
        ]
    )

    np.testing.assert_array_almost_equal(expected_result, result_array, 0.001)

if __name__=="__main__":
    pytest.main()