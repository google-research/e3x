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

import functools
import math
import re
from typing import Callable
import e3x
import jax
import jax.numpy as jnp
import jaxtyping
import pytest

Array = jaxtyping.Array
Float = jaxtyping.Float


@pytest.mark.parametrize(
    'r',
    [
        jnp.asarray([0.0, 1.0, 2.0]),
        jnp.asarray([[0.0, 1.0, 2.0], [-2.0, 0.0, -1.0]]),
    ],
)
@pytest.mark.parametrize('max_degree', [0, 1])
@pytest.mark.parametrize('num', [1, 3])
def test_basis(
    r: Float[Array, '... 3'],
    max_degree: int,
    num: int,
) -> None:
  radial_fn = lambda x, n: jnp.expand_dims(x, -1) ** jnp.arange(n)
  basis = e3x.nn.basis(r, max_degree=max_degree, num=num, radial_fn=radial_fn)
  assert basis.shape[-1] == num
  assert basis.shape[-2] == (max_degree + 1) ** 2
  assert basis.shape[-3] == 1  # Parity axis.
  assert basis.shape[:-3] == r.shape[:-1]
  assert jnp.allclose(
      basis[..., 0, 0, :], radial_fn(e3x.ops.norm(r, axis=-1), num), atol=1e-5
  )


@pytest.fixture(name='constant_radial_fn')
def fixture_constant_radial_fn() -> (
    Callable[[Float[Array, '...'], int], Float[Array, '... N']]
):
  """Radial function for testing purposes (always returns ones)."""
  return lambda x, n: jnp.repeat(jnp.ones_like(x)[..., None], n, axis=-1)


@pytest.mark.parametrize(
    'angular_fn, expected',
    [
        (
            functools.partial(
                e3x.so3.spherical_harmonics,
                r_is_normalized=True,
                normalization='racah',
            ),
            1.0,
        ),
        (
            functools.partial(
                e3x.so3.spherical_harmonics,
                r_is_normalized=False,
                normalization='orthonormal',
            ),
            math.sqrt(1 / (4 * math.pi)),
        ),
    ],
)
def test_basis_angular_fn(
    constant_radial_fn: Callable[
        [Float[Array, '...'], int], Float[Array, '... N']
    ],
    angular_fn: Callable[[Float[Array, '...']], Float[Array, '... N']],
    expected: float,
) -> None:
  basis = e3x.nn.basis(
      r=jnp.asarray([1.0, 0.0, 0.0]),
      max_degree=0,
      num=1,
      radial_fn=constant_radial_fn,
      angular_fn=angular_fn,
  )
  assert jnp.isclose(basis[0, 0, 0], expected)


@pytest.mark.parametrize(
    'cartesian_order, expected',
    [
        (True, jnp.asarray([1.0, 1.0, 0.0, 0.0])),
        (False, jnp.asarray([1.0, 0.0, 0.0, 1.0])),
    ],
)
def test_basis_cartesian_order(
    constant_radial_fn: Callable[
        [Float[Array, '...'], int], Float[Array, '... N']
    ],
    cartesian_order: bool,
    expected: Float[Array, '...'],
) -> None:
  basis = e3x.nn.basis(
      r=jnp.asarray([1.0, 0.0, 0.0]),
      max_degree=1,
      num=1,
      radial_fn=constant_radial_fn,
      cartesian_order=cartesian_order,
  )
  assert jnp.allclose(basis[0, :, 0], expected, atol=1e-5)


def test_basis_with_cutoff_fn() -> None:
  radial_fn = lambda x, n: jnp.repeat(jnp.ones_like(x)[..., None], n, axis=-1)
  cutoff_fn = lambda x: jnp.where(x < 1.0, 1.0 - x, 0.0)
  r = jnp.asarray([[0.5, 0.0, 0.0], [2.0, 0.0, 0.0]])
  basis, cutoff = e3x.nn.basis(
      r=r,
      max_degree=0,
      num=1,
      radial_fn=radial_fn,
      cutoff_fn=cutoff_fn,
      return_cutoff=True,
  )
  expected = jnp.asarray([0.5, 0.0])
  assert jnp.allclose(cutoff, expected, atol=1e-5)
  assert jnp.allclose(basis[..., 0, 0, 0], expected, atol=1e-5)


def test_basis_with_cutoff_fn_and_return_norm() -> None:
  radial_fn = lambda x, n: jnp.repeat(jnp.ones_like(x)[..., None], n, axis=-1)
  cutoff_fn = lambda x: jnp.where(x < 1.0, 1.0 - x, 0.0)
  r = jnp.asarray([[0.5, 0.0, 0.0], [2.0, 0.0, 0.0]])
  basis, cutoff, norm = e3x.nn.basis(
      r=r,
      max_degree=0,
      num=1,
      radial_fn=radial_fn,
      cutoff_fn=cutoff_fn,
      return_cutoff=True,
      return_norm=True,
  )
  expected = jnp.asarray([0.5, 0.0])
  assert jnp.allclose(cutoff, expected, atol=1e-5)
  assert jnp.allclose(basis[..., 0, 0, 0], expected, atol=1e-5)
  assert jnp.allclose(norm, jnp.array([0.5, 2.0]), atol=1e-5)


def test_basis_with_return_norm() -> None:
  radial_fn = lambda x, n: jnp.repeat(jnp.ones_like(x)[..., None], n, axis=-1)
  r = jnp.asarray([[0.5, 0.0, 0.0], [2.0, 0.0, 0.0]])
  _, norm = e3x.nn.basis(
      r=r,
      max_degree=0,
      num=1,
      radial_fn=radial_fn,
      return_norm=True,
  )
  assert jnp.allclose(norm, jnp.array([0.5, 2.0]), atol=1e-5)


def test_basis_with_damping_fn() -> None:
  radial_fn = lambda x, n: jnp.repeat(jnp.ones_like(x)[..., None], n, axis=-1)
  damping_fn = lambda x: jnp.where(x < 1.0, 0.0, 1.0)
  r = jnp.asarray([[0.5, 0.0, 0.0], [2.0, 0.0, 0.0]])
  basis = e3x.nn.basis(
      r=r,
      max_degree=1,
      num=1,
      radial_fn=radial_fn,
      damping_fn=damping_fn,
      cartesian_order=True,
  )
  expected = jnp.asarray(
      [[[[1.0], [0.0], [0.0], [0.0]]], [[[1.0], [1.0], [0.0], [0.0]]]]
  )
  assert jnp.allclose(basis, expected, atol=1e-5)


def test_basis_raises_with_invalid_shape() -> None:
  with pytest.raises(ValueError, match=re.escape('must have shape (..., 3)')):
    e3x.nn.basis(
        r=jnp.ones((1,)), max_degree=0, num=0, radial_fn=lambda x, n: x
    )


def test_basis_raises_when_cutoff_cannot_be_returned() -> None:
  with pytest.raises(ValueError, match='no cutoff_fn'):
    e3x.nn.basis(
        r=jnp.ones((1, 3)),
        max_degree=0,
        num=0,
        radial_fn=lambda x, n: x,
        return_cutoff=True,
    )


@pytest.mark.parametrize('gamma', [0.5, 1.0, 2.0])
@pytest.mark.parametrize('max_degree', [0, 1])
@pytest.mark.parametrize('num', [8, 16])
@pytest.mark.parametrize(
    'radial_fn', [e3x.nn.exponential_bernstein, e3x.nn.exponential_gaussian]
)
def test_exponential_basis_is_equivalent_to_direct_call(
    gamma: float,
    max_degree: int,
    num: int,
    radial_fn: Callable[[Float[Array, '...'], int], Float[Array, '... N']],
) -> None:
  r = jnp.array([[0.1, 0.1, -0.2], [1.0, 0.5, -3.0], [-1.0, 2.0, 0.0]])
  direct_call = e3x.nn.basis(
      r,
      max_degree=max_degree,
      num=num,
      radial_fn=functools.partial(radial_fn, gamma=gamma),
  )
  wrapper, _ = e3x.nn.ExponentialBasis(initial_gamma=gamma).init_with_output(
      jax.random.PRNGKey(0),
      r,
      max_degree=max_degree,
      num=num,
      radial_fn=radial_fn,
  )
  assert jnp.allclose(direct_call, wrapper)


def test_exponential_basis_raises_with_non_exponentially_mapped_radial_fn(
    constant_radial_fn: Callable[
        [Float[Array, '...'], int], Float[Array, '... N']
    ],
) -> None:
  with pytest.raises(TypeError, match='unexpected keyword'):
    e3x.nn.ExponentialBasis().init_with_output(
        jax.random.PRNGKey(0),
        jnp.array([[1.0, 0.0, 0.0]]),
        max_degree=0,
        num=4,
        radial_fn=constant_radial_fn,
    )
