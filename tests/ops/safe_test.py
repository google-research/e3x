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

from typing import Optional, Tuple, Union
import e3x
import jax
import jax.numpy as jnp
import jaxtyping
import pytest

Array = jaxtyping.Array
Float = jaxtyping.Float


@pytest.mark.parametrize(
    'x',
    [
        jnp.asarray([[0.8, 1.4, -0.7], [1.0, 1.0, 1.0]]),
        jnp.asarray([
            [[0.8, 1.4, -0.7], [1.0, 1.0, 1.0]],
            [[1.0, 1.0, 1.0], [0.8, 1.4, -0.7]],
        ]),
    ],
)
@pytest.mark.parametrize('axis', [None, -1, (-1, -2)])
@pytest.mark.parametrize('keepdims', [True, False])
def test_norm(
    x: Float[Array, '...'],
    axis: Optional[Union[int, Tuple[int, ...]]],
    keepdims: bool,
) -> None:
  value, grad = e3x.ops.evaluate_derivatives(
      lambda x: e3x.ops.norm(x, axis=axis, keepdims=keepdims),
      x,
      max_order=1,
  )
  expected_value, expected_grad = e3x.ops.evaluate_derivatives(
      lambda x: jnp.linalg.norm(x, axis=axis, keepdims=keepdims),
      x,
      max_order=1,
  )
  assert jnp.allclose(grad, expected_grad, atol=1e-5)
  assert jnp.allclose(value, expected_value, atol=1e-5)


@pytest.mark.parametrize(
    'x, expected_value, expected_grad',
    [
        (
            jnp.asarray([-1e20, 1e20, -1e20]),
            jnp.sqrt(3) * 1e20,
            jnp.asarray([-1 / jnp.sqrt(3), 1 / jnp.sqrt(3), -1 / jnp.sqrt(3)]),
        ),
        (
            jnp.asarray([0.0, 1e20, -1e20]),
            jnp.sqrt(2) * 1e20,
            jnp.asarray([0.0, 1 / jnp.sqrt(2), -1 / jnp.sqrt(2)]),
        ),
        (
            jnp.asarray([0.0, 0.0, -1e20]),
            1e20,
            jnp.asarray([0.0, 0.0, -1.0]),
        ),
        (
            jnp.asarray([1e-20, -1e-20, 1e-20]),
            jnp.sqrt(3) * 1e-20,
            jnp.asarray([1 / jnp.sqrt(3), -1 / jnp.sqrt(3), 1 / jnp.sqrt(3)]),
        ),
        (
            jnp.asarray([0.0, -1e-20, 1e-20]),
            jnp.sqrt(2) * 1e-20,
            jnp.asarray([0.0, -1 / jnp.sqrt(2), 1 / jnp.sqrt(2)]),
        ),
        (
            jnp.asarray([0.0, 0.0, 1e-20]),
            1e-20,
            jnp.asarray([0.0, 0.0, 1.0]),
        ),
        (
            jnp.asarray([0.0, 0.0, 0.0]),
            0.0,
            jnp.asarray([0.0, 0.0, 0.0]),
        ),
    ],
)
def test_norm_is_nan_safe(
    x: Float[Array, '...'],
    expected_value: Float[Array, '...'],
    expected_grad: Float[Array, '...'],
) -> None:
  value, grad = jax.value_and_grad(e3x.ops.norm)(x)
  assert jnp.allclose(value, expected_value, atol=1e-5)
  assert jnp.allclose(grad, expected_grad, atol=1e-5)


@pytest.mark.parametrize('keepdims', [True, False])
def test_norm_has_nan_safe_derivatives(keepdims: bool) -> None:
  finfo = jnp.finfo(jnp.float32)
  x = jnp.asarray([
      [0.0, 0.0],
      [finfo.tiny, 0.0],
      [finfo.eps, 0.0],
  ])
  norm = lambda x: e3x.ops.norm(x, keepdims=keepdims, axis=-1)
  for y in e3x.ops.evaluate_derivatives(norm, x, max_order=4):
    assert jnp.all(jnp.isfinite(y))


@pytest.mark.parametrize(
    'x',
    [
        jnp.asarray([[0.8, 1.4, -0.7], [1.0, 1.0, 1.0]]),
        jnp.asarray([
            [[0.8, 1.4, -0.7], [1.0, 1.0, 1.0]],
            [[1.0, 1.0, 1.0], [0.8, 1.4, -0.7]],
        ]),
    ],
)
@pytest.mark.parametrize('axis', [None, -1, (-1, -2)])
@pytest.mark.parametrize('keepdims', [True, False])
def test_normalize_and_return_norm(
    x: Float[Array, '...'],
    axis: Optional[Union[int, Tuple[int, ...]]],
    keepdims: bool,
) -> None:
  y, n = e3x.ops.normalize_and_return_norm(x, axis=axis, keepdims=keepdims)
  # Check that norm matches the result of jnp.linalg.norm.
  reference_n = jnp.linalg.norm(x, axis=axis, keepdims=keepdims)
  assert n.shape == reference_n.shape
  assert jnp.allclose(n, reference_n, atol=1e-5)
  # Check that output is normalized.
  assert jnp.allclose(jnp.linalg.norm(y, axis=axis), 1.0, atol=1e-5)


@pytest.mark.parametrize(
    'x',
    [
        jnp.asarray([[0.8, 1.4, -0.7], [1.0, 1.0, 1.0]]),
        jnp.asarray([
            [[0.8, 1.4, -0.7], [1.0, 1.0, 1.0]],
            [[1.0, 1.0, 1.0], [0.8, 1.4, -0.7]],
        ]),
    ],
)
@pytest.mark.parametrize('axis', [None, -1, (-1, -2)])
def test_normalize(
    x: Float[Array, '...'], axis: Optional[Union[int, Tuple[int, ...]]]
) -> None:
  y = e3x.ops.normalize(x, axis=axis)
  assert jnp.allclose(jnp.linalg.norm(y, axis=axis), 1.0, atol=1e-5)


def test_normalize_has_nan_safe_derivatives() -> None:
  finfo = jnp.finfo(jnp.float32)
  x = jnp.asarray([
      [0.0, 0.0],
      [finfo.tiny, 0.0],
      [finfo.eps, 0.0],
  ])
  normalize = lambda x: jnp.sum(e3x.ops.normalize(x, axis=-1))
  for y in e3x.ops.evaluate_derivatives(normalize, x, max_order=4):
    assert jnp.all(jnp.isfinite(y))
