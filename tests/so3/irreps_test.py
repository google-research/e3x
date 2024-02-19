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

import re
import e3x
from ..testing import subtests
import jax
import jax.numpy as jnp
import jaxtyping
import pytest

Array = jaxtyping.Array
Float = jaxtyping.Float
Integer = jaxtyping.Integer


@pytest.mark.parametrize('x', [-1.5, 0.0, 1.0])
@pytest.mark.parametrize('y', [-1.5, 0.0, 1.0])
@pytest.mark.parametrize('z', [-1.5, 0.0, 1.0])
def test_spherical_harmonics(x: float, y: float, z: float) -> None:
  r = jnp.asarray([x, y, z])
  output = e3x.so3.spherical_harmonics(
      r,
      max_degree=2,
      r_is_normalized=True,
      cartesian_order=False,
      normalization='racah',
  )
  expected = jnp.asarray([
      1,
      y,
      z,
      x,
      jnp.sqrt(3) * x * y,
      jnp.sqrt(3) * y * z,
      (z**2 - 0.5 * (x**2 + y**2)),
      jnp.sqrt(3) * x * z,
      jnp.sqrt(3) / 2 * (x**2 - y**2),
  ])
  assert jnp.allclose(output, expected, atol=1e-5)


@pytest.mark.parametrize('max_degree', [0, 1, 2, 3, 4, 5])
def test_spherical_harmonics_max_degree(max_degree: int) -> None:
  assert e3x.so3.spherical_harmonics(
      jnp.zeros(3), max_degree=max_degree
  ).shape == ((max_degree + 1) ** 2,)


@pytest.mark.parametrize('r_is_normalized', [True, False])
def test_spherical_harmonics_r_is_normalized(r_is_normalized: bool) -> None:
  r = jnp.asarray([1.0, 1.0, 1.0])
  expected = jnp.asarray([1.0, 1.0, 1.0, 1.0])
  if not r_is_normalized:
    expected = expected.at[1:].divide(jnp.sqrt(3))
  assert jnp.allclose(
      e3x.so3.spherical_harmonics(
          r,
          max_degree=1,
          r_is_normalized=r_is_normalized,
          cartesian_order=True,
          normalization='racah',
      ),
      expected,
      atol=1e-5,
  )


@pytest.mark.parametrize('cartesian_order', [True, False])
def test_spherical_harmonics_cartesian_order(cartesian_order: bool) -> None:
  r = jnp.asarray([0.0, 2.0, 3.0])
  output = e3x.so3.spherical_harmonics(
      r,
      max_degree=1,
      r_is_normalized=True,
      cartesian_order=cartesian_order,
      normalization='racah',
  )
  if cartesian_order:
    expected = jnp.asarray([1.0, 0.0, 2.0, 3.0])
  else:
    expected = jnp.asarray([1.0, 2.0, 3.0, 0.0])
  assert jnp.allclose(output, expected, atol=1e-5)


@pytest.mark.parametrize(
    'normalization', ['4pi', 'orthonormal', 'racah', 'schmidt']
)
def test_spherical_harmonics_normalization(normalization: str) -> None:
  r = jnp.asarray([0.0, 2.0, 3.0])
  output = e3x.so3.spherical_harmonics(
      r,
      max_degree=1,
      r_is_normalized=True,
      cartesian_order=True,
      normalization=normalization,
  )
  expected = jnp.asarray([1.0, 0.0, 2.0, 3.0])
  if normalization == '4pi':
    expected *= jnp.sqrt(2 * jnp.asarray([0, 1, 1, 1]) + 1)
  elif normalization == 'orthonormal':
    expected *= jnp.sqrt((2 * jnp.asarray([0, 1, 1, 1]) + 1) / (4 * jnp.pi))
  assert jnp.allclose(output, expected, atol=1e-5)


def test_spherical_harmonics_has_nan_safe_derivatives(
    max_degree: int = 2,
) -> None:
  finfo = jnp.finfo(jnp.float32)
  x = jnp.asarray([
      [0.0, 0.0, 0.0],
      [finfo.tiny, 0.0, 0.0],
      [finfo.eps, 0.0, 0.0],
  ])
  func = lambda x: e3x.so3.spherical_harmonics(x, max_degree=max_degree)
  for y in e3x.ops.evaluate_derivatives(func, x, max_order=4):
    assert jnp.all(jnp.isfinite(y))


@pytest.mark.parametrize('x', [-1.5, 0.0, 1.0])
@pytest.mark.parametrize('y', [-1.5, 0.0, 1.0])
@pytest.mark.parametrize('z', [-1.5, 0.0, 1.0])
def test_solid_harmonics(x: float, y: float, z: float):
  r = jnp.asarray([x, y, z])
  output = e3x.so3.solid_harmonics(r, max_degree=2, cartesian_order=True)
  expected = jnp.asarray([
      1,
      x,
      y,
      z,
      jnp.sqrt(3) / 2 * (x**2 - y**2),
      jnp.sqrt(3) * x * y,
      jnp.sqrt(3) * x * z,
      jnp.sqrt(3) * y * z,
      (z**2 - 0.5 * (x**2 + y**2)),
  ])
  assert jnp.allclose(output, expected, atol=1e-5)


@pytest.fixture(name='expected_cg')
def fixture_expected_cg() -> Float[Array, '9 9 9']:
  """Clebsch-Gordant for max_degree1 = max_degree2 = max_degree3 = 2."""
  # pyformat: disable
  return jnp.asarray([
      [
          [1, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 1, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 1, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 1, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 1, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 1, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 1, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 1, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 1]
      ],
      [
          [0, 1, 0, 0, 0, 0, 0, 0, 0],
          [jnp.sqrt(3)/3, 0, 0, 0, 0, 0, -jnp.sqrt(6)/6, 0, -jnp.sqrt(2)/2],
          [0, 0, 0, jnp.sqrt(2)/2, 0, jnp.sqrt(2)/2, 0, 0, 0],
          [0, 0, -jnp.sqrt(2)/2, 0, jnp.sqrt(2)/2, 0, 0, 0, 0],
          [0, 0, 0, jnp.sqrt(30)/10, 0, -jnp.sqrt(6)/6, 0, 0, 0],
          [0, 0, jnp.sqrt(30)/10, 0, jnp.sqrt(6)/6, 0, 0, 0, 0],
          [0, -jnp.sqrt(10)/10, 0, 0, 0, 0, 0, jnp.sqrt(2)/2, 0],
          [0, 0, 0, 0, 0, 0, -jnp.sqrt(2)/2, 0, jnp.sqrt(6)/6],
          [0, -jnp.sqrt(30)/10, 0, 0, 0, 0, 0, -jnp.sqrt(6)/6, 0]
      ],
      [
          [0, 0, 1, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, -jnp.sqrt(2)/2, 0, jnp.sqrt(2)/2, 0, 0, 0],
          [jnp.sqrt(3)/3, 0, 0, 0, 0, 0, jnp.sqrt(6)/3, 0, 0],
          [0, jnp.sqrt(2)/2, 0, 0, 0, 0, 0, jnp.sqrt(2)/2, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, -jnp.sqrt(6)/3],
          [0, jnp.sqrt(30)/10, 0, 0, 0, 0, 0, -jnp.sqrt(6)/6, 0],
          [0, 0, jnp.sqrt(10)/5, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, jnp.sqrt(30)/10, 0, jnp.sqrt(6)/6, 0, 0, 0],
          [0, 0, 0, 0, jnp.sqrt(6)/3, 0, 0, 0, 0]
      ],
      [
          [0, 0, 0, 1, 0, 0, 0, 0, 0],
          [0, 0, jnp.sqrt(2)/2, 0, jnp.sqrt(2)/2, 0, 0, 0, 0],
          [0, -jnp.sqrt(2)/2, 0, 0, 0, 0, 0, jnp.sqrt(2)/2, 0],
          [jnp.sqrt(3)/3, 0, 0, 0, 0, 0, -jnp.sqrt(6)/6, 0, jnp.sqrt(2)/2],
          [0, jnp.sqrt(30)/10, 0, 0, 0, 0, 0, jnp.sqrt(6)/6, 0],
          [0, 0, 0, 0, 0, 0, jnp.sqrt(2)/2, 0, jnp.sqrt(6)/6],
          [0, 0, 0, -jnp.sqrt(10)/10, 0, -jnp.sqrt(2)/2, 0, 0, 0],
          [0, 0, jnp.sqrt(30)/10, 0, -jnp.sqrt(6)/6, 0, 0, 0, 0],
          [0, 0, 0, jnp.sqrt(30)/10, 0, -jnp.sqrt(6)/6, 0, 0, 0]
      ],
      [
          [0, 0, 0, 0, 1, 0, 0, 0, 0],
          [0, 0, 0, jnp.sqrt(30)/10, 0, jnp.sqrt(6)/6, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, jnp.sqrt(6)/3],
          [0, jnp.sqrt(30)/10, 0, 0, 0, 0, 0, -jnp.sqrt(6)/6, 0],
          [jnp.sqrt(5)/5, 0, 0, 0, 0, 0, -jnp.sqrt(14)/7, 0, 0],
          [0, -jnp.sqrt(10)/10, 0, 0, 0, 0, 0, jnp.sqrt(42)/14, 0],
          [0, 0, 0, 0, -jnp.sqrt(14)/7, 0, 0, 0, 0],
          [0, 0, 0, jnp.sqrt(10)/10, 0, jnp.sqrt(42)/14, 0, 0, 0],
          [0, 0, -jnp.sqrt(10)/5, 0, 0, 0, 0, 0, 0]
      ],
      [
          [0, 0, 0, 0, 0, 1, 0, 0, 0],
          [0, 0, jnp.sqrt(30)/10, 0, -jnp.sqrt(6)/6, 0, 0, 0, 0],
          [0, jnp.sqrt(30)/10, 0, 0, 0, 0, 0, jnp.sqrt(6)/6, 0],
          [0, 0, 0, 0, 0, 0, -jnp.sqrt(2)/2, 0, -jnp.sqrt(6)/6],
          [0, jnp.sqrt(10)/10, 0, 0, 0, 0, 0, jnp.sqrt(42)/14, 0],
          [jnp.sqrt(5)/5, 0, 0, 0, 0, 0, jnp.sqrt(14)/14, 0, -jnp.sqrt(42)/14],
          [0, 0, 0, jnp.sqrt(30)/10, 0, jnp.sqrt(14)/14, 0, 0, 0],
          [0, 0, -jnp.sqrt(10)/10, 0, jnp.sqrt(42)/14, 0, 0, 0, 0],
          [0, 0, 0, jnp.sqrt(10)/10, 0, -jnp.sqrt(42)/14, 0, 0, 0]
      ],
      [
          [0, 0, 0, 0, 0, 0, 1, 0, 0],
          [0, -jnp.sqrt(10)/10, 0, 0, 0, 0, 0, -jnp.sqrt(2)/2, 0],
          [0, 0, jnp.sqrt(10)/5, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, -jnp.sqrt(10)/10, 0, jnp.sqrt(2)/2, 0, 0, 0],
          [0, 0, 0, 0, -jnp.sqrt(14)/7, 0, 0, 0, 0],
          [0, 0, 0, -jnp.sqrt(30)/10, 0, jnp.sqrt(14)/14, 0, 0, 0],
          [jnp.sqrt(5)/5, 0, 0, 0, 0, 0, jnp.sqrt(14)/7, 0, 0],
          [0, jnp.sqrt(30)/10, 0, 0, 0, 0, 0, jnp.sqrt(14)/14, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, -jnp.sqrt(14)/7]
      ],
      [
          [0, 0, 0, 0, 0, 0, 0, 1, 0],
          [0, 0, 0, 0, 0, 0, jnp.sqrt(2)/2, 0, -jnp.sqrt(6)/6],
          [0, 0, 0, jnp.sqrt(30)/10, 0, -jnp.sqrt(6)/6, 0, 0, 0],
          [0, 0, jnp.sqrt(30)/10, 0, jnp.sqrt(6)/6, 0, 0, 0, 0],
          [0, 0, 0, -jnp.sqrt(10)/10, 0, jnp.sqrt(42)/14, 0, 0, 0],
          [0, 0, jnp.sqrt(10)/10, 0, jnp.sqrt(42)/14, 0, 0, 0, 0],
          [0, -jnp.sqrt(30)/10, 0, 0, 0, 0, 0, jnp.sqrt(14)/14, 0],
          [jnp.sqrt(5)/5, 0, 0, 0, 0, 0, jnp.sqrt(14)/14, 0, jnp.sqrt(42)/14],
          [0, jnp.sqrt(10)/10, 0, 0, 0, 0, 0, jnp.sqrt(42)/14, 0]
      ],
      [
          [0, 0, 0, 0, 0, 0, 0, 0, 1],
          [0, -jnp.sqrt(30)/10, 0, 0, 0, 0, 0, jnp.sqrt(6)/6, 0],
          [0, 0, 0, 0, -jnp.sqrt(6)/3, 0, 0, 0, 0],
          [0, 0, 0, jnp.sqrt(30)/10, 0, jnp.sqrt(6)/6, 0, 0, 0],
          [0, 0, jnp.sqrt(10)/5, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, -jnp.sqrt(10)/10, 0, -jnp.sqrt(42)/14, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, -jnp.sqrt(14)/7],
          [0, -jnp.sqrt(10)/10, 0, 0, 0, 0, 0, jnp.sqrt(42)/14, 0],
          [jnp.sqrt(5)/5, 0, 0, 0, 0, 0, -jnp.sqrt(14)/7, 0, 0]
      ]
  ])
  # pyformat: enable


@pytest.mark.parametrize('cartesian_order', [False, True])
def test_clebsch_gordan(
    cartesian_order: bool,
    expected_cg: Float[Array, '...'],
    max_degree: int = 2,
):
  cg = e3x.so3.clebsch_gordan(
      max_degree, max_degree, max_degree, cartesian_order=cartesian_order
  )
  if cartesian_order:
    p = e3x.so3.irreps._cartesian_permutation(max_degree)
    expected_cg = expected_cg[p, :, :][:, p, :][:, :, p]
  assert jnp.allclose(cg, expected_cg, atol=1e-5)


@pytest.mark.parametrize('l1', [0, 1, 2])
@pytest.mark.parametrize('l2', [0, 1, 2])
@pytest.mark.parametrize('l3', [0, 1, 2])
@pytest.mark.parametrize('cartesian_order', [False, True])
def test_clebsch_gordan_for_degrees(
    l1: int,
    l2: int,
    l3: int,
    cartesian_order: bool,
    expected_cg: Float[Array, '...'],
) -> None:
  cg = e3x.so3.clebsch_gordan_for_degrees(
      degree1=l1, degree2=l2, degree3=l3, cartesian_order=cartesian_order
  )
  expected_cg = expected_cg[
      l1**2 : (l1 + 1) ** 2, l2**2 : (l2 + 1) ** 2, l3**2 : (l3 + 1) ** 2
  ]
  if cartesian_order:
    p1 = e3x.so3.irreps._cartesian_permutation_for_degree(l1)
    p2 = e3x.so3.irreps._cartesian_permutation_for_degree(l2)
    p3 = e3x.so3.irreps._cartesian_permutation_for_degree(l3)
    expected_cg = expected_cg[p1, :, :][:, p2, :][:, :, p3]
  assert jnp.allclose(cg, expected_cg, atol=1e-5)


@pytest.mark.parametrize(
    'x, degree, message',
    [
        (jnp.asarray(0), 0, 'must be a multi-dimensional array'),
        (jnp.zeros((3,)), 0, 'must have shape (..., 1)'),
        (jnp.zeros((1,)), 1, 'must have shape (..., 3)'),
        (jnp.zeros((1,)), 2, 'must have shape (..., 3, 3)'),
        (jnp.zeros((1,)), 3, 'must have shape (..., 3, 3, 3)'),
    ],
)
def test__check_tensor_shape(
    x: Float[Array, '...'], degree: int, message: str
) -> None:
  with pytest.raises(ValueError, match=re.escape(message)):
    e3x.so3.irreps._check_tensor_shape(x, degree=degree)


@subtests({
    'degree=0': dict(x=jnp.asarray([1.0]), degree=0, expected=True),
    'degree=1': dict(x=jnp.asarray([1.0, 1.0, 1.0]), degree=1, expected=True),
    'degree=2': dict(
        x=jnp.asarray([
            [+1.0, +2.0, -3.0],
            [+2.0, +1.0, -0.5],
            [-3.0, -0.5, -2.0],
        ]),
        degree=2,
        expected=True,
    ),
    'not symmetric': dict(
        x=jnp.asarray([
            [+1.0, -2.0, +3.0],
            [+2.0, +1.0, +0.5],
            [-3.0, -0.5, -2.0],
        ]),
        degree=2,
        expected=False,
    ),
    'not traceless': dict(
        x=jnp.asarray([
            [+1.0, +2.0, -3.0],
            [+2.0, +1.0, -0.5],
            [-3.0, -0.5, +2.0],
        ]),
        degree=2,
        expected=False,
    ),
})
def test_is_traceless_symmetric(
    x: Float[Array, '...'], degree: int, expected: bool
) -> None:
  assert e3x.so3.is_traceless_symmetric(x, degree) == expected


@pytest.mark.parametrize(
    'degree, expected',
    [
        (0, tuple()),
        (1, (jnp.asarray([0, 1, 2]),)),
        (2, (jnp.asarray([0, 0, 0, 1, 1, 2]), jnp.asarray([0, 1, 2, 1, 2, 2]))),
        (
            3,
            (
                jnp.asarray([0, 0, 0, 0, 0, 0, 1, 1, 1, 2]),
                jnp.asarray([0, 0, 0, 1, 1, 2, 1, 1, 2, 2]),
                jnp.asarray([0, 1, 2, 1, 2, 2, 1, 2, 2, 2]),
            ),
        ),
    ],
)
def test__tensor_compression_indices(
    degree: int, expected: tuple[Integer[Array, '...'], ...]
) -> None:
  for i, j in zip(e3x.so3.irreps._tensor_compression_indices(degree), expected):
    assert jnp.array_equal(i, j)


@pytest.mark.parametrize(
    'degree, expected',
    [
        (0, jnp.asarray([0])),
        (1, jnp.asarray([0, 1, 2])),
        (2, jnp.asarray([0, 1, 2, 1, 3, 4, 2, 4, 5])),
        (
            3,
            jnp.asarray([
                0,
                1,
                2,
                1,
                3,
                4,
                2,
                4,
                5,
                1,
                3,
                4,
                3,
                6,
                7,
                4,
                7,
                8,
                2,
                4,
                5,
                4,
                7,
                8,
                5,
                8,
                9,
            ]),
        ),
    ],
)
def test__tensor_expansion_indices(
    degree: int, expected: Integer[Array, '...']
) -> None:
  assert jnp.array_equal(
      e3x.so3.irreps._tensor_expansion_indices(degree), expected
  )


@subtests({
    "normalization='racah'": dict(
        x=jnp.asarray([1.0]),
        degree=0,
        cartesian_order=True,
        normalization='racah',
        expected=jnp.asarray([1.0]),
    ),
    "normalization='orthonormal'": dict(
        x=jnp.asarray([1.0]),
        degree=0,
        cartesian_order=True,
        normalization='orthonormal',
        expected=jnp.asarray([jnp.sqrt(4 * jnp.pi)]),
    ),
    'cartesian_order=True': dict(
        x=jnp.asarray([0.0, 1.0, 2.0]),
        degree=1,
        cartesian_order=True,
        normalization='racah',
        expected=jnp.asarray([0.0, 1.0, 2.0]),
    ),
    'cartesian_order=False': dict(
        x=jnp.asarray([1.0, 2.0, 0.0]),
        degree=1,
        cartesian_order=False,
        normalization='racah',
        expected=jnp.asarray([0.0, 1.0, 2.0]),
    ),
    'degree=2, r=(1, 0, 0)': dict(
        x=jnp.asarray([0.0, 0.0, -1 / 2, 0.0, +jnp.sqrt(3) / 2]),
        degree=2,
        cartesian_order=False,
        normalization='racah',
        expected=jnp.asarray([
            [2 / 3, 0.0, 0.0],
            [0.0, -1 / 3, 0.0],
            [0.0, 0.0, -1 / 3],
        ]),
    ),
    'degree=2, r=(0, 1, 0)': dict(
        x=jnp.asarray([0.0, 0.0, -1 / 2, 0.0, -jnp.sqrt(3) / 2]),
        degree=2,
        cartesian_order=False,
        normalization='racah',
        expected=jnp.asarray([
            [-1 / 3, 0.0, 0.0],
            [0.0, 2 / 3, 0.0],
            [0.0, 0.0, -1 / 3],
        ]),
    ),
    'degree=2, r=(0, 0, 1)': dict(
        x=jnp.asarray([0.0, 0.0, 1.0, 0.0, 0.0]),
        degree=2,
        cartesian_order=False,
        normalization='racah',
        expected=jnp.asarray([
            [-1 / 3, 0.0, 0.0],
            [0.0, -1 / 3, 0.0],
            [0.0, 0.0, 2 / 3],
        ]),
    ),
    'degree=2, r=(√⅓, √⅓, √⅓)': dict(
        x=jnp.asarray(
            [jnp.sqrt(1 / 3), jnp.sqrt(1 / 3), 0.0, jnp.sqrt(1 / 3), 0.0]
        ),
        degree=2,
        cartesian_order=False,
        normalization='racah',
        expected=jnp.asarray([
            [0.0, 1 / 3, 1 / 3],
            [1 / 3, 0.0, 1 / 3],
            [1 / 3, 1 / 3, 0.0],
        ]),
    ),
})
def test_irreps_to_tensor(
    x: Float[Array, '...'],
    degree: int,
    cartesian_order: bool,
    normalization: e3x.so3.Normalization,
    expected: Float[Array, '...'],
) -> None:
  y = e3x.so3.irreps_to_tensor(
      x=x,
      degree=degree,
      cartesian_order=cartesian_order,
      normalization=normalization,
  )
  assert jnp.allclose(y, expected, atol=1e-5)


def test_irreps_to_tensor_raises_when_input_has_incorrect_shape() -> None:
  x = jnp.ones(shape=(2, 6))
  with pytest.raises(ValueError, match=re.escape('must have shape (..., 5)')):
    e3x.so3.irreps_to_tensor(x=x, degree=2)


def test_irreps_to_tensor_raises_when_normalization_is_invalid() -> None:
  x = jnp.ones(shape=(2, 3))
  with pytest.raises(ValueError, match='foo'):
    e3x.so3.irreps_to_tensor(x=x, degree=1, normalization='foo')


@subtests({
    "normalization='racah'": dict(
        x=jnp.asarray([1.0]),
        degree=0,
        cartesian_order=True,
        normalization='racah',
        expected=jnp.asarray([1.0]),
    ),
    "normalization='orthonormal'": dict(
        x=jnp.asarray([1.0]),
        degree=0,
        cartesian_order=True,
        normalization='orthonormal',
        expected=jnp.asarray([1 / jnp.sqrt(4 * jnp.pi)]),
    ),
    'cartesian_order=True': dict(
        x=jnp.asarray([0.0, 1.0, 2.0]),
        degree=1,
        cartesian_order=True,
        normalization='racah',
        expected=jnp.asarray([0.0, 1.0, 2.0]),
    ),
    'cartesian_order=False': dict(
        x=jnp.asarray([0.0, 1.0, 2.0]),
        degree=1,
        cartesian_order=False,
        normalization='racah',
        expected=jnp.asarray([1.0, 2.0, 0.0]),
    ),
    'degree=2, r=(1, 0, 0)': dict(
        x=jnp.asarray([
            [2 / 3, 0.0, 0.0],
            [0.0, -1 / 3, 0.0],
            [0.0, 0.0, -1 / 3],
        ]),
        degree=2,
        cartesian_order=False,
        normalization='racah',
        expected=jnp.asarray([0.0, 0.0, -1 / 2, 0.0, +jnp.sqrt(3) / 2]),
    ),
    'degree=2, r=(0, 1, 0)': dict(
        x=jnp.asarray([
            [-1 / 3, 0.0, 0.0],
            [0.0, 2 / 3, 0.0],
            [0.0, 0.0, -1 / 3],
        ]),
        degree=2,
        cartesian_order=False,
        normalization='racah',
        expected=jnp.asarray([0.0, 0.0, -1 / 2, 0.0, -jnp.sqrt(3) / 2]),
    ),
    'degree=2, r=(0, 0, 1)': dict(
        x=jnp.asarray([
            [-1 / 3, 0.0, 0.0],
            [0.0, -1 / 3, 0.0],
            [0.0, 0.0, 2 / 3],
        ]),
        degree=2,
        cartesian_order=False,
        normalization='racah',
        expected=jnp.asarray([0.0, 0.0, 1.0, 0.0, 0.0]),
    ),
    'degree=2, r=(√⅓, √⅓, √⅓)': dict(
        x=jnp.asarray([
            [0.0, 1 / 3, 1 / 3],
            [1 / 3, 0.0, 1 / 3],
            [1 / 3, 1 / 3, 0.0],
        ]),
        degree=2,
        cartesian_order=False,
        normalization='racah',
        expected=jnp.asarray(
            [jnp.sqrt(1 / 3), jnp.sqrt(1 / 3), 0.0, jnp.sqrt(1 / 3), 0.0]
        ),
    ),
})
def test_tensor_to_irreps(
    x: Float[Array, '...'],
    degree: int,
    cartesian_order: bool,
    normalization: e3x.so3.Normalization,
    expected: Float[Array, '...'],
) -> None:
  y = e3x.so3.tensor_to_irreps(
      x=x,
      degree=degree,
      cartesian_order=cartesian_order,
      normalization=normalization,
  )
  assert jnp.allclose(y, expected, atol=1e-5)


def test_tensor_to_irreps_raises_when_input_has_incorrect_shape() -> None:
  x = jnp.ones(shape=(2, 6))
  with pytest.raises(
      ValueError, match=re.escape('must have shape (..., 3, 3)')
  ):
    e3x.so3.tensor_to_irreps(x=x, degree=2)


def test_tensor_to_irreps_raises_when_normalization_is_invalid() -> None:
  x = jnp.ones(shape=(2, 3))
  with pytest.raises(ValueError, match='foo'):
    e3x.so3.tensor_to_irreps(x=x, degree=1, normalization='foo')


@pytest.mark.parametrize('degree', [0, 1, 2, 3, 4])
@pytest.mark.parametrize('cartesian_order', [True, False])
@pytest.mark.parametrize(
    'normalization', e3x.so3._normalization.valid_normalizations
)
@pytest.mark.parametrize('batch_dims', [tuple(), (1, 2)])
def test_irreps_to_tensor_and_vice_versa(
    degree: int,
    cartesian_order: bool,
    normalization: e3x.so3.Normalization,
    batch_dims: tuple[int, ...],
) -> None:
  x = jax.random.normal(
      jax.random.PRNGKey(0), shape=(*batch_dims, 2 * degree + 1)
  )
  t = e3x.so3.irreps_to_tensor(  # Convert to tensor...
      x=x,
      degree=degree,
      cartesian_order=cartesian_order,
      normalization=normalization,
  )
  assert e3x.so3.is_traceless_symmetric(x=t, degree=degree)
  y = e3x.so3.tensor_to_irreps(  # ...and back again.
      x=t,
      degree=degree,
      cartesian_order=cartesian_order,
      normalization=normalization,
  )
  assert jnp.allclose(x, y, atol=1e-5)
