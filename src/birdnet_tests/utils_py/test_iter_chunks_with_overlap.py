
from itertools import islice

import numpy as np
import numpy.testing as npt

from birdnet.utils import iter_chunks_with_overlap


def test_2_0__returns_02_24():
  result = list(islice(iter_chunks_with_overlap(2, 0), 2))
  assert result == [
    (0, 2),
    (2, 4),
  ]

  assert isinstance(result[0][0], float)
  assert isinstance(result[0][1], float)
  assert isinstance(result[1][0], float)
  assert isinstance(result[1][1], float)


def test_2_1__returns_02_13():
  result = list(islice(iter_chunks_with_overlap(2, 1), 2))
  assert result == [
    (0, 2),
    (1, 3),
  ]


def test_2_2_1p9__returns_02__0p1_2p1():
  result = np.array(list(islice(iter_chunks_with_overlap(2, 1.9), 2)))
  npt.assert_array_almost_equal(result, [
    (0, 2),
    (0.1, 2.1),
  ])


def test_2_2_1p9__returns_02__1p5_3p5():
  result = np.array(list(islice(iter_chunks_with_overlap(2, 0.5), 2)))
  npt.assert_array_almost_equal(result, [
    (0, 2),
    (1.5, 3.5),
  ])
