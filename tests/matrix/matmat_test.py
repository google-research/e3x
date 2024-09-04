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

"""Tests for matmat.
"""

import itertools
import math
from typing import Any, Dict, List, Sequence, Union

import e3x
import jax
from jax import numpy as jnp
import jaxtyping
import numpy as np
import pytest
import scipy.spatial.transform

Rotation = scipy.spatial.transform.Rotation

Array = jaxtyping.Array
Bool = jaxtyping.Bool
Float = jaxtyping.Float
Shaped = jaxtyping.Shaped
UInt32 = jaxtyping.UInt32
Dtype = Any
Shape = Sequence[Union[int, Any]]
PRNGKey = UInt32[Array, '2']


@pytest.mark.parametrize('ls', [[1], [2, 4]])
def test_compute_side_lengths(ls):
  result = e3x.matrix.matmat.compute_side_lengths(ls)
  assert len(result) == len(ls)
  assert isinstance(result, tuple)
  for i in range(len(ls)):
    assert result[i] == 2 * ls[i] + 1


@pytest.mark.parametrize('ls', [[0], [2, 4], [0, 1, 2, 3]])
def test_make_dict_irreps_mult(ls):
  result = e3x.matrix.matmat.make_dict_irreps_mult(ls)
  assert isinstance(result, dict)
  assert len(result) == 2 * max(ls) + 1
  # Compare the dimension of the big matrix created from these l's:
  # - as sum of dimension of irreducibles occurring in it.
  dim = 0
  for i in range(2 * max(ls)+1):
    assert i in result
    assert result[i] >= 1
    dim += result[i] * (2 * i + 1)
  # - as the square of the total side length of the matrix.
  total_length = 2 * sum(ls) + len(ls)
  assert dim == total_length ** 2
  # Check the number of irreducibles.
  num = 0
  for a, b in itertools.product(ls, ls):
    num += 2 * min(a, b) + 1
  assert sum(result.values()) == num


@pytest.mark.parametrize('ls', [[0], [2, 4], [0, 1, 2, 3]])
@pytest.mark.parametrize('max_l', [0, 1, 2])
def test_make_dict_irreps_max_degree_limit(ls, max_l):
  result = e3x.matrix.matmat.make_dict_irreps_mult(ls, max_l)
  assert isinstance(result, dict)
  assert len(result) == min(2 * max(ls), max_l) + 1


@pytest.mark.parametrize('n_shells', [1, 3])
@pytest.mark.parametrize('mult', [1, 3])
@pytest.mark.parametrize('stddev', [1.0, 0.1])
@pytest.mark.parametrize('ls', [[1], [0, 2]])
def test_init_matrix_irreps_weights(
    n_shells: int, ls: Sequence[int], mult: int, stddev: float
):
  key: PRNGKey = jax.random.PRNGKey(42)
  dict_irreps: Dict[int, int] = e3x.matrix.matmat.make_dict_irreps_mult(ls)
  result = e3x.matrix.matmat.init_matrix_irreps_weights(
      key, n_shells, dict_irreps, mult, stddev
  )
  max_l = 2 * max(ls)  # Largest L of any subrepresentation of matrices.
  assert len(result) == max_l + 1
  all_weights = []
  for l in range(max_l+1):
    assert result[l].shape == (n_shells, dict_irreps[l]* mult*mult)
    all_weights.append(result[l] * math.sqrt(2 * l + 1))
  all_weights = jnp.concatenate(all_weights, axis=1).flatten()
  actual_stddev = jnp.std(all_weights).item()
  error = abs(actual_stddev - stddev)
  assert error < 2 * stddev / math.sqrt(len(all_weights))


def test_get_traces():
  mat = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                   [1.0, 2.1, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                   [1.0, 2.0, 0.1, 4.0, 5.0, 6.0, 7.0, 8.0],
                   [1.0, 2.0, 3.0, 0.2, 5.0, 6.0, 7.0, 8.0],
                   [1.0, 2.0, 3.0, 4.0, 0.3, 6.0, 7.0, 8.0],
                   [1.0, 2.0, 3.0, 4.0, 5.0, 0.2, 7.0, 8.0],
                   [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.4, 8.0],
                   [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.6]])
  ls = [0, 1]
  mult = 2
  # shift_by_id=False: Do not subtract identity matrix before computing traces.
  traces = e3x.matrix.matmat.get_traces(mat, ls, mult, False)
  expected = jnp.array([1.0, 2.0, 1.0, 2.1,    # 1x1 submatrices
                        0.6, 21.0, 12.0, 1.2]  # 3x3 submatrices
                      )
  assert jnp.allclose(traces, expected)

  # shift_by_id=True: Subtract identity matrix before computing traces.
  traces = e3x.matrix.matmat.get_traces(mat, ls, mult, True)
  expected = jnp.array([0.0, 2.0, 1.0, 1.1,    # 1x1 submatrices
                        -2.4, 21.0, 12.0, -1.8]  # 3x3 submatrices
                      )
  assert jnp.allclose(traces, expected)


# Test combine_irreps, make_rects, make_big_square together
# by testing invariance under rotations.
@pytest.mark.parametrize('n_shells', [1, 3])
@pytest.mark.parametrize('ls', [[1], [0, 1, 2], [2, 3]])
@pytest.mark.parametrize('l_max', [3, 6])
@pytest.mark.parametrize('mult', [1, 3])
@pytest.mark.parametrize('shift_id', [True, False])
def test_rotational_invariance(ls: List[int], l_max: int, mult: int,
                               n_shells: int, shift_id: bool):
  n_points = 20
  xyz = np.random.normal(size=(n_shells, n_points, 3))
  r = Rotation.random().as_matrix()
  xyz_r = np.einsum('spd,de->spe', xyz, r)
  sh = e3x.so3.irreps.spherical_harmonics(xyz, max_degree=l_max)
  sum_sh = jnp.sum(sh, axis=1)  # [shells, (l_max+1)^2]
  sh_r = e3x.so3.irreps.spherical_harmonics(xyz_r, max_degree=l_max)
  sum_sh_r = jnp.sum(sh_r, axis=1)  # [shells, (l_max+1)^2]

  key = jax.random.PRNGKey(42)
  dict_irreps = e3x.matrix.matmat.make_dict_irreps_mult(ls, l_max)
  stddev = 0.05
  params = e3x.matrix.matmat.init_matrix_irreps_weights(
      key, n_shells, dict_irreps, mult, stddev
  )

  # Construct the square matrix from the irreps.
  sum_sh_t = jnp.transpose(sum_sh)  # [(l_max+1)^2, shells]
  raw_irreps = e3x.matrix.matmat.combine_irreps(sum_sh_t, params, 'high')

  # Check also that the identity matrix is added if requested.
  square_raw = e3x.matrix.matmat.make_square_matrix(
      raw_irreps, ls, mult, l_max, False, 'high'
  )
  square_shifted = e3x.matrix.matmat.make_square_matrix(
      raw_irreps, ls, mult, l_max, True, 'high'
  )
  side_length = (2 * sum(ls) + len(ls)) * mult
  id_mat = np.eye(side_length)
  assert id_mat.shape == square_raw.shape == square_shifted.shape
  assert np.allclose(square_shifted - square_raw, id_mat, atol=1e-6, rtol=1e-5)

  square = square_shifted if shift_id else square_raw

  # Construct the square matrix corresponding to the rotated xyz.
  sum_sh_t_r = jnp.transpose(sum_sh_r)  # [(l_max+1)^2, shells]
  raw_irreps_r = e3x.matrix.matmat.combine_irreps(sum_sh_t_r, params, 'high')
  square_r = e3x.matrix.matmat.make_square_matrix(
      raw_irreps_r, ls, mult, l_max, shift_id, 'high'
  )
  assert square.shape == square_r.shape

  # Check that the square matrices themselves are not equal...
  prod_traces = e3x.matrix.matmat.get_traces(square, ls, mult, shift_id)
  prod_traces_r = e3x.matrix.matmat.get_traces(square_r, ls, mult, shift_id)
  assert not np.allclose(
      square[-2:, -2:], square_r[-2:, -2:], atol=1e-6, rtol=1e-5
  )
  # ...but the traces of the square submatrices are.
  assert np.allclose(prod_traces, prod_traces_r, atol=1e-6, rtol=1e-5)

  # Check powers of the matrix.
  mat2 = jnp.matmul(square, square)
  mat3 = jnp.matmul(mat2, square)
  prod_traces2 = e3x.matrix.matmat.get_traces(mat2, ls, mult, shift_id)
  prod_traces3 = e3x.matrix.matmat.get_traces(mat3, ls, mult, shift_id)
  mat2_r = jnp.matmul(square_r, square_r)
  mat3_r = jnp.matmul(mat2_r, square_r)
  prod_traces2_r = e3x.matrix.matmat.get_traces(mat2_r, ls, mult, shift_id)
  prod_traces3_r = e3x.matrix.matmat.get_traces(mat3_r, ls, mult, shift_id)
  assert np.allclose(prod_traces2, prod_traces2_r, atol=1e-5, rtol=1e-5)
  assert np.allclose(prod_traces3, prod_traces3_r, atol=1e-5, rtol=1e-5)
