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

from typing import Set, Tuple
import e3x
import jax.numpy as jnp
import jaxtyping
import pytest

Array = jaxtyping.Array
Integer = jaxtyping.Integer


def test__check_degree_is_positive_or_zero() -> None:
  with pytest.raises(ValueError, match='degree must be positive or zero'):
    e3x.so3._common._check_degree_is_positive_or_zero(-1)


@pytest.mark.parametrize(
    'degree, expected',
    [
        (0, jnp.asarray([0])),
        (1, jnp.asarray([2, 0, 1])),
        (2, jnp.asarray([4, 0, 3, 1, 2])),
        (3, jnp.asarray([6, 0, 5, 1, 4, 2, 3])),
    ],
)
def test__cartesian_permutation_for_degree(
    degree: int, expected: Integer[Array, '...']
) -> None:
  assert jnp.array_equal(
      e3x.so3._common._cartesian_permutation_for_degree(degree), expected
  )


@pytest.mark.parametrize(
    'max_degree, expected',
    [
        (0, jnp.asarray([0])),
        (1, jnp.asarray([0, 3, 1, 2])),
        (2, jnp.asarray([0, 3, 1, 2, 8, 4, 7, 5, 6])),
        (
            3,
            jnp.asarray([0, 3, 1, 2, 8, 4, 7, 5, 6, 15, 9, 14, 10, 13, 11, 12]),
        ),
    ],
)
def test__cartesian_permutation(
    max_degree: int, expected: Integer[Array, '...']
) -> None:
  assert jnp.array_equal(
      e3x.so3._common._cartesian_permutation(max_degree), expected
  )


@pytest.mark.parametrize(
    'max_degree, expected',
    [
        (0, jnp.asarray([0])),
        (1, jnp.asarray([0, 9, 7, 8, 3, 1, 2, 6, 4, 5])),
        (
            2,
            # pyformat: disable
            jnp.asarray([
                0, 9, 7, 8, 3, 1, 2, 6, 4, 5, 34, 30, 33, 31, 32, 14, 10, 13,
                11, 12, 29, 25, 28, 26, 27, 19, 15, 18, 16, 17, 24, 20, 23, 21,
                22
            ]),
            # pyformat: enable
        ),
        (
            3,
            # pyformat: disable
            jnp.asarray([
                0, 9, 7, 8, 3, 1, 2, 6, 4, 5, 34, 30, 33, 31, 32, 14, 10, 13,
                11, 12, 29, 25, 28, 26, 27, 19, 15, 18, 16, 17, 24, 20, 23, 21,
                22, 83, 77, 82, 78, 81, 79, 80, 41, 35, 40, 36, 39, 37, 38, 76,
                70, 75, 71, 74, 72, 73, 48, 42, 47, 43, 46, 44, 45, 69, 63, 68,
                64, 67, 65, 66, 55, 49, 54, 50, 53, 51, 52, 62, 56, 61, 57, 60,
                58, 59
            ]),
            # pyformat: enable
        ),
    ],
)
def test__cartesian_permuation_wigner_d_entries(
    max_degree: int, expected: Integer[Array, '...']
) -> None:
  print(
      repr(e3x.so3._common._cartesian_permutation_wigner_d_entries(max_degree))
  )
  assert jnp.array_equal(
      e3x.so3._common._cartesian_permutation_wigner_d_entries(max_degree),
      expected,
  )


@pytest.mark.parametrize(
    'degree, expected',
    [(0, 1), (1, 3), (2, 6), (3, 10)],
)
def test__number_of_cartesian_monomials_of_degree(
    degree: int, expected: int
) -> None:
  assert (
      e3x.so3._common._number_of_cartesian_monomials_of_degree(degree)
      == expected
  )


@pytest.mark.parametrize(
    'degree, expected',
    [(0, 1), (1, 3), (2, 5), (3, 7)],
)
def test__number_of_spherical_harmonics_of_degree(
    degree: int, expected: int
) -> None:
  assert (
      e3x.so3._common._number_of_spherical_harmonics_of_degree(degree)
      == expected
  )


@pytest.mark.parametrize(
    'degree, expected',
    [(0, 1), (1, 9), (2, 45), (3, 165)],
)
def test__number_of_rotation_matrix_monomials_of_degree(
    degree: int, expected: int
) -> None:
  assert (
      e3x.so3._common._number_of_rotation_matrix_monomials_of_degree(degree)
      == expected
  )


@pytest.mark.parametrize(
    'degree, expected',
    [(0, 1), (1, 9), (2, 25), (3, 49)],
)
def test__number_of_wigner_d_entries_of_degree(
    degree: int, expected: int
) -> None:
  assert (
      e3x.so3._common._number_of_wigner_d_entries_of_degree(degree) == expected
  )


@pytest.mark.parametrize(
    'max_degree, expected',
    [(0, 1), (1, 4), (2, 10), (3, 20)],
)
def test__total_number_of_cartesian_monomials(
    max_degree: int, expected: int
) -> None:
  assert (
      e3x.so3._common._total_number_of_cartesian_monomials(max_degree)
      == expected
  )


@pytest.mark.parametrize(
    'max_degree, expected',
    [(0, 1), (1, 4), (2, 9), (3, 16)],
)
def test__total_number_of_spherical_harmonics(
    max_degree: int, expected: int
) -> None:
  assert (
      e3x.so3._common._total_number_of_spherical_harmonics(max_degree)
      == expected
  )


@pytest.mark.parametrize(
    'max_degree, expected',
    [(0, 1), (1, 10), (2, 55), (3, 220)],
)
def test__total_number_of_rotation_matrix_monomials(
    max_degree: int, expected: int
) -> None:
  assert (
      e3x.so3._common._total_number_of_rotation_matrix_monomials(max_degree)
      == expected
  )


@pytest.mark.parametrize(
    'max_degree, expected',
    [(0, 1), (1, 10), (2, 35), (3, 84)],
)
def test__total_number_of_wigner_d_entries(
    max_degree: int, expected: int
) -> None:
  assert (
      e3x.so3._common._total_number_of_wigner_d_entries(max_degree) == expected
  )


@pytest.mark.parametrize(
    'n, k, l, expected',
    [
        (1, 1, 0, {(1,)}),
        (1, 1, 1, {(1,)}),
        (2, 2, 1, {(1, 1)}),
        (2, 2, 0, {(1, 1), (0, 2)}),
        (3, 3, 0, {(1, 1, 1), (0, 1, 2), (0, 0, 3)}),
    ],
)
def test__partitions(
    n: int, k: int, l: int, expected: Set[Tuple[int, ...]]
) -> None:
  partitions = set(e3x.so3._common._partitions(n, k, l))
  assert partitions == expected


@pytest.mark.parametrize(
    'n, k, expected',
    [
        (1, 1, {(1,)}),
        (2, 2, {(1, 1), (0, 2), (2, 0)}),
        (
            3,
            3,
            {
                (1, 1, 1),
                (0, 1, 2),
                (0, 2, 1),
                (1, 0, 2),
                (1, 2, 0),
                (2, 0, 1),
                (2, 1, 0),
                (0, 0, 3),
                (0, 3, 0),
                (3, 0, 0),
            },
        ),
    ],
)
def test__multicombinations(
    n: int, k: int, expected: Set[Tuple[int, ...]]
) -> None:
  multicombinations = set(e3x.so3._common._multicombinations(n, k))
  assert multicombinations == expected


@pytest.mark.parametrize(
    'degree, expected',
    [
        (0, {(0, 0, 0)}),
        (1, {(0, 0, 1), (0, 1, 0), (1, 0, 0)}),
        (2, {(0, 1, 1), (1, 0, 1), (1, 1, 0), (0, 0, 2), (0, 2, 0), (2, 0, 0)}),
    ],
)
def test__monomial_powers_of_degree(
    degree: int, expected: Set[Tuple[int, ...]]
) -> None:
  powers = set(e3x.so3._common._monomial_powers_of_degree(degree))
  assert powers == expected


@pytest.mark.parametrize(
    'degree, expected',
    [
        (0, {(0, 0, 0, 0, 0, 0, 0, 0, 0)}),
        (
            1,
            {
                (0, 0, 0, 0, 0, 0, 0, 0, 1),
                (0, 0, 0, 0, 0, 0, 0, 1, 0),
                (0, 0, 0, 0, 0, 0, 1, 0, 0),
                (0, 0, 0, 0, 0, 1, 0, 0, 0),
                (0, 0, 0, 0, 1, 0, 0, 0, 0),
                (0, 0, 0, 1, 0, 0, 0, 0, 0),
                (0, 0, 1, 0, 0, 0, 0, 0, 0),
                (0, 1, 0, 0, 0, 0, 0, 0, 0),
                (1, 0, 0, 0, 0, 0, 0, 0, 0),
            },
        ),
    ],
)
def test__rotation_matrix_powers_of_degree(
    degree: int, expected: Set[Tuple[int, ...]]
) -> None:
  powers = set(e3x.so3._common._rotation_matrix_powers_of_degree(degree))
  assert powers == expected


def test__integer_powers() -> None:
  x = jnp.asarray([[0.0, -1.0, 2.0, -1.5]])
  max_degree = 2
  assert jnp.allclose(
      e3x.so3._common._integer_powers(x, max_degree),
      jnp.asarray([
          [1.0, 1.0, 1.0, 1.0],
          [0.0, -1.0, 2.0, -1.5],
          [0.0, 1.0, 4.0, 2.25],
      ]),
  )
