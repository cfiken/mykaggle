from pytest import approx
import numpy as np
import torch

from mykaggle.lib.torch_util import label_smoothing


class TestTorchUtil:

    def test_label_smoothing(self):
        targets = np.array([
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1]
        ])
        targets = torch.tensor(targets)
        smoothing = 0.1
        expected = [
            [0.02, 0.92, 0.02, 0.02, 0.02],
            [0.02, 0.02, 0.02, 0.02, 0.92],
        ]
        actual = label_smoothing(targets, 5, smoothing).numpy()
        assert expected[0] == approx(actual.tolist()[0])
        assert expected[1] == approx(actual.tolist()[1])
