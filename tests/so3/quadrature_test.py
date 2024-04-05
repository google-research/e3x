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

from typing import Callable, Optional
import e3x
import jax.numpy as jnp
import jaxtyping
import pytest

Array = jaxtyping.Array
Float = jaxtyping.Float


# For reference of available Lebedev rules, see:
# https://people.sc.fsu.edu/~jburkardt/datasets/sphere_lebedev_rule/sphere_lebedev_rule.html
@pytest.mark.parametrize(
    'precision, num, expected_size',
    [
        (-100, None, 6),
        (0, None, 6),
        (3, None, 6),
        (4, None, 14),
        (5, None, 14),
        (6, None, 26),
        (7, None, 26),
        (8, None, 38),
        (126, None, 5810),
        (131, None, 5810),  # 131 is the highest available precision.
        (500, None, 5810),
        (None, -10, 6),
        (None, 0, 6),
        (None, 5, 6),
        (None, 6, 6),
        (None, 7, 6),
        (None, 13, 6),
        (None, 14, 14),
        (None, 15, 14),
        (None, 25, 14),
        (None, 26, 26),
        (None, 38, 38),
        (None, 5810, 5810),
        (None, 10000, 5810),
    ],
)
def test__load_grid(
    precision: Optional[int], num: Optional[int], expected_size: int
) -> None:
  _, w = e3x.so3.quadrature._load_grid(
      kind='Lebedev', precision=precision, num=num
  )
  assert w.size == expected_size


def test__load_grid_raises_with_invalid_kind() -> None:
  with pytest.raises(ValueError, match="kind='Foo' does not exist"):
    e3x.so3.quadrature._load_grid(kind='Foo')  # type: ignore


@pytest.mark.parametrize(
    'quadrature', [e3x.so3.lebedev_quadrature, e3x.so3.delley_quadrature]
)
@pytest.mark.parametrize('precision', list(range(0, 132)))
def test_quadrature(
    quadrature: Callable[
        [Optional[int], Optional[int]],
        tuple[Float[Array, 'num_points 3'], Float[Array, 'num_points']],
    ],
    precision: int,
) -> None:
  # The spherical harmonics are orthonormalized, so we can test whether the
  # quadrature rules work as expected by checking their orthonormalization.

  r, w = quadrature(precision, None)
  ylm = e3x.so3.spherical_harmonics(
      r,
      max_degree=min(precision // 2, 15),
      r_is_normalized=True,
      normalization='4pi',
  )
  eye = jnp.einsum('na,nb,n->ab', ylm, ylm, w)
  assert jnp.allclose(eye, jnp.eye(ylm.shape[-1]), atol=1e-5)


@pytest.mark.parametrize(
    'quadrature', [e3x.so3.lebedev_quadrature, e3x.so3.delley_quadrature]
)
@pytest.mark.parametrize(
    'precision, num, message',
    [
        (
            None,
            None,
            'Exactly one of precision=None or num=None must be specified.',
        ),
        (
            1,
            1,
            'Exactly one of precision=1 or num=1 must be specified.',
        ),
    ],
)
def test_quadrature_raises_with_invalid_inputs(
    quadrature: Callable[
        [Optional[int], Optional[int]],
        tuple[Float[Array, 'num_points 3'], Float[Array, 'num_points']],
    ],
    precision: Optional[int],
    num: Optional[int],
    message: str,
) -> None:
  with pytest.raises(ValueError, match=message):
    quadrature(precision, num)
