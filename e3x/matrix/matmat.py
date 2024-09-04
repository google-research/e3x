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

r"""Helper functions for matrices of matrices.

The "matrix of matrices" is a square matrix made up of smaller
rectangular matrices. The smaller matrices are of the sizes :math:`a \times b`
where :math:`a` and :math:`b` are of the form :math:`2\ell+1` with :math:`\ell`
in a list ``degrees`` of ints. Each :math:`\ell` in ``degrees`` is used
``mult`` times, so the big square matrix consists of
``(mult*len(degrees))**2`` rectangular submatrices.

Each :math:`a \times b` matrix is a representation of :math:`\mathrm{SO}(3)`
with the decomposition

.. math::
  H(\ell_a) \otimes H(\ell_b) = H(\lvert\ell_a-\ell_b\rvert) \oplus \dots \oplus
  H(\ell_a+\ell_b)

where :math:`a = 2 \cdot \ell_a + 1` and :math:`b = 2 \cdot \ell_b + 1`,
and the :math:`H(\ell)` are the irreps of :math:`\mathrm{SO}(3)`.
"""

import collections
import itertools
import math
from typing import Any, Dict, List, Sequence, Tuple, Optional, Union

import e3x
import jax
from jax import numpy as jnp
import jaxtyping


Array = jaxtyping.Array
Bool = jaxtyping.Bool
Float = jaxtyping.Float
Shaped = jaxtyping.Shaped
UInt32 = jaxtyping.UInt32
Dtype = Any
Shape = Sequence[Union[int, Any]]
PRNGKey = UInt32[Array, '2']
PrecisionLike = jax.lax.PrecisionLike


def compute_side_lengths(degrees: Sequence[int]) -> Tuple[int, ...]:
  """Compute the side lengths from the degrees (L's).

  Args:
    degrees:  array of degrees for parts.

  Returns:
    Tuple of side lengths of rectangles from ls
  """
  return tuple([2 * l + 1 for l in degrees])


def make_dict_irreps_mult(
    degrees: Sequence[int], max_degree: Optional[int] = None
) -> Dict[int, int]:
  r"""Compute multiplicities of irreducibles in matrix space.

  For :math:`V = (\ell=ls[0]) \oplus ... \oplus (\ell=ls[-1])` compute the
  number of irreducible components of :math:`V \otimes V` with a given
  :math:`\ell`.
  Make dictionary :math:`\ell\to` Number irreps with this :math:`\ell`,
  up to :math:`\ell\leq` ``max_degree``

  Args:
    degrees: e.g. [0,1,1,2]  for 16 matrices of size {1,3,3,5} :math:`\times`
      {1,3,3,5}.
    max_degree:  Maximal :math:`l` to use in these matrices

  Returns:
    dictionary :math:`\ell\to` Number irreps with this :math:`\ell`
  """
  result = collections.defaultdict(int)
  for la, lb in itertools.product(degrees, degrees):
    max_l = la + lb if max_degree is None else min(la + lb, max_degree)
    for l in range(abs(la - lb), max_l + 1):
      result[l] += 1
  return result


def init_matrix_irreps_weights(key: PRNGKey,
                               n_shells: int,
                               dict_irreps: Dict[int, int],
                               mult: int,
                               stddev: float) -> List[Float[Array, 'S M']]:
  """Initializes weights for irreducibles needed for big matrix.

  Args:
    key: The PRNGKey used as the random key.
    n_shells: Number of shells / colors / channels / radial basis functions.
    dict_irreps: dictionary l -> Number irreps with this l
    mult: multiplicity of each l
    stddev: Standard deviation of weights
  Returns:
    [Float[n_shells, mult[l]] for l in range(max_l+1) ]
  """
  weights = []
  max_l = max(dict_irreps.keys())
  keys = jax.random.split(key, max_l + 1)
  for l in range(max_l+1):
    weights.append(jax.random.normal(keys[l],
                                     (n_shells, dict_irreps[l]* mult*mult))
                   * stddev / math.sqrt(2*l+1))
  return weights


def combine_irreps(inputs: Float[Array, '... Q F'],
                   weights: List[Float[Array, 'F M']],
                   precision: PrecisionLike
                   ) -> List[Float[Array, '... D M']]:
  """Combines n_shells input irreps into mult[l] output irreps for each l.

  Args:
    inputs: Float[... (L+1)**2, n_shells]
            The input has irreps of dim 1,3,5,...,2L+1.
    weights: [ Float[n_shells, mult[l]]  for l in range(l_max+1) ]
    precision: Precision for einsum, e.g. 'high'.
  Returns:
    [Float[..., 2l+1, mult[l]] for l in range(l_max+1) ]
  """
  l_max = len(weights) - 1
  irreps = []
  for l in range(l_max + 1):
    assert weights[l].shape[0] == inputs.shape[-1]  # inputs: n_shells
    irreps.append(jnp.einsum('io,...vi->...vo',
                             weights[l],
                             inputs[..., l**2 : (l + 1) ** 2, :],
                             precision=precision)
                 )
  return irreps


def make_rectangular_matrix(irreps: List[Float[Array, 'D M']],
                            idx: List[int],
                            a: int,
                            b: int,
                            cg: Float[Array, 'L1 L2 L3'],
                            mult: int,
                            precision: PrecisionLike) -> Float[Array, 'AM BM']:
  """Make a rectangular (mult*a) x (mult*b) matrix out of irreps.

  a and b must be odd.
  Args:
    irreps: List of Float[2l+1, mult(l)]
    idx:    Next irreps to use, 0 <= idx[l] < mult(l)
    a:      resulting shape is (a,b)
    b:      resulting shape is (a,b)
    cg:     Clebsch-Gordan coefficients.
    mult:   Number of rectangles in each direction.
    precision: Precision for einsum, e.g. 'high'.

  Returns:
    Float[a*mult, b*mult]
  """
  for i in range(len(irreps)):
    if len(irreps[i].shape) != 2:
      raise ValueError(f'{len(irreps[i].shape)=} for {i=} (should be 2)')
    if irreps[0].shape[0] % 2 == 0:
      raise ValueError(
          f'{irreps[i].shape[0]=} for {i=} (should be odd)'
      )
  if a % 2 == 0:
    raise ValueError(f'{a=} (should be odd)')
  if b % 2 == 0:
    raise ValueError(f'{b=} (should be odd)')
  l_min = irreps[0].shape[0] // 2
  l_max = irreps[-1].shape[0] // 2

  la = a // 2
  lb = b // 2
  # H(|la-lb|)⊕...⊕H(la+lb) ≃ H(la) ⊗ H(lb)

  lower = max(abs(la - lb), l_min)
  upper = min(la + lb, l_max) + 1
  if lower >= upper:
    # No irreducibles that would fit to this size of rectangle.
    return jnp.zeros((a * mult, b * mult), dtype=jnp.float32)

  # Initialize constants.
  if cg.shape[0] < (la + 1) ** 2:
    raise ValueError(
        f'cg.shape[0] < (la + 1) ** 2: {cg.shape[0]} < {(la + 1) ** 2}'
    )
  if cg.shape[1] < (lb + 1) ** 2:
    raise ValueError(
        f'cg.shape[1] < (lb + 1) ** 2: {cg.shape[1]} < {(lb + 1) ** 2}'
    )
  if cg.shape[2] < upper**2:
    raise ValueError(
        f'cg.shape[2] < upper**2: {cg.shape[2]} < {(upper**2)}'
    )
  cg_ab = cg[la**2 : (la + 1) ** 2, lb**2 : (lb + 1) ** 2, : upper**2]

  m2: int = mult * mult
  concat = [irreps[l][:, idx[l] : idx[l] + m2] for l in range(lower, upper)]
  v = jnp.concatenate(concat, axis=0)  # [upperˆ2-lowerˆ2, mult**2]
  for l in range(lower, upper):
    idx[l] += m2
    if idx[l] > irreps[l].shape[1]:
      raise ValueError(
          f'idx[l] > irreps[l].shape[1]: {idx[l]} > {irreps[l].shape[1]}'
      )
  result = jnp.einsum(
      'mf,ijm->ijf',
      v,
      cg_ab[:, :, lower**2 : upper**2],
      precision=precision,
  )
  # Result is now [a, b, mult**2], rearranged into big matrix.
  expanded = result.reshape((a, b, mult, mult))
  rearranged = expanded.transpose((2, 0, 3, 1))  # [mult, a, mult, b]
  return rearranged.reshape((a * mult, b * mult))


def make_square_matrix(
    irreps: List[Float[Array, 'D M']],
    degrees: Sequence[int],
    mult: int,
    max_degree: int,
    shift_by_id: bool,
    precision: PrecisionLike
) -> Float[Array, 'S S']:
  """Make a big square matrix out of irreps, add identity matrix.

  Args:
    irreps:   List of Float[2l+1, mult(l)]
    degrees:       e.g. [0,1,1,2]
    mult:     multiplicity of each l in ls.
    max_degree:    maximal l to use for matrix.
    shift_by_id:   If True, add identity matrix to the result.
    precision: Precision for einsum, e.g. 'high'.

  Returns:
    Square matrix of side length sum(ns)*mult with ns = 2*ls+1
  """
  if max_degree < max(degrees):
    raise ValueError(f'max_degree < max(ls): {max_degree} < {max(degrees)}')
  for i in range(len(irreps)):
    if irreps[i].shape[0] % 2 == 0:
      raise ValueError(
          f'{irreps[i].shape[0]=} for {i=} (should be odd)'
      )
  with jax.ensure_compile_time_eval():
    cg = e3x.so3.clebsch_gordan(
        max(degrees), max(degrees), max_degree, cartesian_order=True
    )
  ns = compute_side_lengths(degrees)  # side lengths of small (a x b) - matrices
  side = sum(ns) * mult  # total side length of large matrix
  result = jnp.zeros((side, side))
  idx = [0] * len(irreps)  # pointers to next irreps to use

  # Compute all rectangles and put them into the square matrix.
  start_a = 0
  for a in ns:
    end_a = start_a + a * mult
    start_b = 0
    for b in ns:
      end_b = start_b + b * mult
      new_rect = make_rectangular_matrix(irreps, idx, a, b, cg, mult, precision)
      if shift_by_id and (start_a == start_b):
        new_rect += jnp.eye(a * mult)
      result = result.at[start_a:end_a, start_b:end_b].set(new_rect)
      start_b = end_b
    start_a = end_a
  return result


def get_traces(mat: Float[Array, 'S S'],
               ls: Sequence[int],
               mult: int,
               shift_by_id: bool) -> Float[Array, 'N']:
  """Get array of traces of all square submatrices.

  Args:
    mat:  Big square matrix assembled from (a x b) - submatrices.
    ls:   List of L's for the submatrices (a = 2*l+1)
    mult: Number of times each each L is used.
    shift_by_id: If True, subtract identity matrix before computing traces.
  Returns:
    Array of traces of all square submatrices.
  """
  result = []
  start = 0
  for l in ls:
    a = 2 * l + 1
    for i, j in itertools.product(range(mult), range(mult)):
      if shift_by_id and (i == j):
        result.append(mat[start + i*a : start + (i+1)*a,
                          start + j*a : start + (j+1)*a].trace() - a)
      else:
        result.append(mat[start + i*a : start + (i+1)*a,
                          start + j*a : start + (j+1)*a].trace())
    start += a * mult
  return jnp.array(result)
