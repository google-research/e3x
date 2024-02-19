# Copyright 2024 The e3x Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pathlib
from typing import Any, Dict
import e3x
from e3x.so3._spherical_harmonics_lut import _generate_spherical_harmonics_lookup_table
import numpy as np
import pytest


_EXPECTED_LUT: Dict[str, Any] = dict(
    max_degree=2,
    cm=np.asarray(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, -1 / 2, 0, -np.sqrt(3) / 2],
            [0, 0, 0, 0, 0, 0, -1 / 2, 0, np.sqrt(3) / 2],
            [0, 0, 0, 0, 0, np.sqrt(3), 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, np.sqrt(3), 0],
            [0, 0, 0, 0, np.sqrt(3), 0, 0, 0, 0],
        ],
        dtype=np.float32,
    ),
    ls=np.asarray(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 2],
            [0, 2, 0],
            [2, 0, 0],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ],
        dtype=np.int32,
    ),
)


def assert_lut_is_correct(lut: Dict[str, Any], max_degree: int = 2) -> None:
  assert lut["max_degree"] == max_degree
  assert np.allclose(lut["cm"], _EXPECTED_LUT["cm"])
  assert np.array_equal(lut["ls"], _EXPECTED_LUT["ls"])


@pytest.fixture(autouse=True)
def create_dummy_lut(tmp_path: pathlib.Path):
  path = tmp_path / "lut.npz"
  with path.open("wb") as f:
    np.savez_compressed(f, max_degree=-1)
  e3x.Config.set_spherical_harmonics_cache(path)
  yield  # Cleanup code comes after yield.
  e3x.Config.set_spherical_harmonics_cache()  # Reset state.


def test_generate_lookup_table(max_degree: int = 2) -> None:
  lut = _generate_spherical_harmonics_lookup_table(max_degree)
  assert_lut_is_correct(lut)


def test_incremental_generation(max_degree: int = 2) -> None:
  lut = None
  for l in range(max_degree + 1):
    lut = _generate_spherical_harmonics_lookup_table(l)
  assert_lut_is_correct(lut)
