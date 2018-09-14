import pytest
import numpy as np
from srf.test import TestCase
from srf.preprocess.function.on_tor_lors import map_process, direction, SliceByAxis
from srf.preprocess.function.on_tor_lors import *


class TestToRLoRs(TestCase):
    def setUp(self):
        super().setUp()

    def get_dummy_lors(self):

        lors = np.array([
            [-5.0, -5.0, -5.0, 5.0, 5.0, 5.0],
            [-5.0, -5.0, 0.0, 5.0, 5.0, 0.0],
            [0.0, 5.0, 0.0, 0.0, -5.0, 0.0],
            [-5.0, 0.0, 0.0, 5.0, 0.0, 0.0],
            [0.0, -3.0, 5.0, 0.0, 3.0, -5.0],
            [102., 37., -15., 109., -5., -15.]
        ], dtype=np.float32)
        return lors.reshape((-1, 6))

    def test_direction(self):
        lors = self.get_dummy_lors()
        result = direction(lors[0, :].reshape((-1, 6)))
        expected = np.array([[10.0, 10.0, 10.0], ])
        self.assertFloatArrayEqual(result, expected)

    def test_slice_by_axis(self):
        lors = self.get_dummy_lors()
        result = SliceByAxis(Axis.x)(lors[:, 0:3])
        expected = np.array(lors[:, 0])
        self.assertFloatArrayEqual(result, expected)

    def get_dummy_directions(self):
        return np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0], ], dtype=np.float32
        )

    def test_dominentBy(self):
        directions = self.get_dummy_directions()
        # directions = [directions[:, 0], directions[:, 1], directions[:, 2]]
        result = dominentBy(
            Axis.z, directions[:, 0], directions[:, 1], directions[:, 2])
        expected = [[2], ]
        self.assertFloatArrayEqual(result, expected)

    def test_partition(self):
        lors = self.get_dummy_lors()
        result = Partition(Axis.x)(lors)
        expected = np.array([
            [-5.0, -5.0, -5.0, 5.0, 5.0, 5.0],
            [-5.0, -5.0, 0.0, 5.0, 5.0, 0.0],
            [-5.0, 0.0, 0.0, 5.0, 0.0, 0.0],
        ], dtype=np.float32)
        self.assertFloatArrayEqual(result, expected)

    @pytest.mark.skip(reason='NIY')
    def test_sigma2_factor(self):
        pass

    def test_map_process(self):
        lors = self.get_dummy_lors()
        result = map_process(lors)
        expected = {
            Axis.x: np.array([
                [-5.0, -5.0, -5.0, 5.0, 5.0, 5.0],
                [-5.0, 0.0, -5.0, 5.0, 0.0, 5.0],
                [0.0, 0.0, -5.0, 0.0, 0.0, 5.0],
            ]),
            Axis.y: np.array([
                [0.0, 0.0, -5.0, 0.0, 0.0, 5.0],
                [109., -15., -5., 102.,  -15., 37.]
            ]),
            Axis.z: np.array([
                [0.0, 3.0, -5.0, 0.0, -3.0, 5.0],
            ]),
        }
        self.assertFloatArrayEqual(result[Axis.x][:, 0:6], expected[Axis.x])
        self.assertFloatArrayEqual(result[Axis.y][:, 0:6], expected[Axis.y])
        self.assertFloatArrayEqual(result[Axis.z][:, 0:6], expected[Axis.z])
