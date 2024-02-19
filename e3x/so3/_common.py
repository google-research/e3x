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

"""Common utility functions used in the so3 submodule."""

import math
from typing import Iterator
import jax.numpy as jnp
import jaxtyping
import more_itertools

Array = jaxtyping.Array
Float = jaxtyping.Float
Integer = jaxtyping.Integer


def _check_degree_is_positive_or_zero(degree: int) -> None:
  """Checks whether the input degree is positive or zero."""
  if degree < 0:
    raise ValueError(f'degree must be positive or zero, received {degree}')


def _cartesian_permutation_for_degree(l: int) -> Integer[Array, '2*l+1']:
  """Generates a permutation to Cartesian order for degree l."""
  _check_degree_is_positive_or_zero(l)
  p = jnp.empty(shape=2 * l + 1, dtype=int)
  i = 0
  for m in range(l):
    p = p.at[i].set(2 * l + 1 - (m + 1))
    i += 1
    p = p.at[i].set(m)
    i += 1
  p = p.at[i].set((2 * l + 1) // 2)
  return p


def _cartesian_permutation(
    max_degree: int,
) -> Integer[Array, '(max_degree+1)**2']:
  """Generates a permutation to Cartesian order for all l=0..max_degree."""
  _check_degree_is_positive_or_zero(max_degree)
  p = jnp.empty(shape=(max_degree + 1) ** 2, dtype=int)
  for l in range(max_degree + 1):
    p = p.at[l**2 : (l + 1) ** 2].set(
        _cartesian_permutation_for_degree(l) + l**2
    )
  return p


def _cartesian_permutation_wigner_d_entries(
    max_degree: int,
) -> Integer[Array, 'num_wigner_d_entries']:
  """Generates a permutation to Cartesian order for Wigner-D matrix entries."""
  _check_degree_is_positive_or_zero(max_degree)
  permutations = []
  offset = 0
  for l in range(max_degree + 1):
    pvec = _cartesian_permutation_for_degree(l)
    num = pvec.size
    pmat = jnp.arange(num * num, dtype=pvec.dtype).reshape(num, num)
    pmat = jnp.reshape(pmat[pvec, :][:, pvec], -1)
    permutations.append(pmat + offset)
    offset += pmat.size
  return jnp.concatenate(permutations)


def _number_of_cartesian_monomials_of_degree(degree: int) -> int:
  """Calculates number of Cartesian monomials of a given degree."""
  return ((degree + 1) * (degree + 2)) // 2


def _number_of_spherical_harmonics_of_degree(degree: int) -> int:
  """Calculates number of spherical harmonics of a given degree."""
  return 2 * degree + 1


def _number_of_rotation_matrix_monomials_of_degree(degree: int) -> int:
  """Calculates number of monomials of 9 variables of a given degree."""
  return math.comb(degree + 8, degree)


def _number_of_wigner_d_entries_of_degree(degree: int) -> int:
  """Calculates number of Wigner-D matrix entries of a given degree."""
  num = 2 * degree + 1
  return num * num


def _total_number_of_cartesian_monomials(max_degree: int) -> int:
  """Calculates total number of Cartesian monomials."""
  return ((max_degree + 1) * (max_degree + 2) * (max_degree + 3)) // 6


def _total_number_of_spherical_harmonics(max_degree: int) -> int:
  """Calculates total number of spherical harmonics."""
  max_degree_plus_one = max_degree + 1
  return max_degree_plus_one * max_degree_plus_one


def _total_number_of_rotation_matrix_monomials(max_degree: int) -> int:
  """Calculates total number of monomials of 9 variables up to max_degree."""
  return ((max_degree + 1) * math.comb(max_degree + 9, max_degree + 1)) // 9


def _total_number_of_wigner_d_entries(max_degree: int) -> int:
  """Calculates total number of Wigner-D matrix entries."""
  return ((max_degree + 1) * (2 * max_degree + 1) * (2 * max_degree + 3)) // 3


def _partitions(n: int, k: int, l: int = 0) -> Iterator[tuple[int, ...]]:
  """Yields all k-tuples of integers >= l that sum to n."""
  if k < 1:
    return
  if k == 1:
    if n >= l:
      yield (n,)
    return
  for i in range(l, n + 1):
    for partitions in _partitions(n=n - i, k=k - 1, l=i):
      yield (i,) + partitions


def _multicombinations(n: int, k: int) -> Iterator[tuple[int, ...]]:
  """Yields all multicombinations of k elements chosen from n variables."""
  for partition in _partitions(n=n, k=k):
    for permutation in more_itertools.distinct_permutations(partition):
      yield permutation


def _monomial_powers_of_degree(degree: int) -> Iterator[tuple[int, int, int]]:
  """Yields all possible (a,b,c) for monomials xᵃyᵇzᶜ with a given degree."""
  for multicombination in _multicombinations(n=degree, k=3):
    yield multicombination


def _rotation_matrix_powers_of_degree(
    degree: int,
) -> Iterator[tuple[int, int, int, int, int, int, int, int, int]]:
  """Yields all power combinations of 9 variable monomials with given degree."""
  for multicombination in _multicombinations(n=degree, k=9):
    yield multicombination


def _integer_powers(
    x: Float[Array, '... 1 d'], max_degree: int
) -> Float[Array, '... max_degree+1 d']:
  """Calculates all integer powers up to max_degree of x along axis -2."""
  return jnp.cumprod(
      jnp.concatenate(
          (jnp.ones_like(x), jnp.repeat(x, max_degree, axis=-2)),
          axis=-2,
      ),
      axis=-2,
  )
