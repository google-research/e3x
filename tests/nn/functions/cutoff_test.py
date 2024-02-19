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

from typing import Callable
import e3x
import jax.numpy as jnp
import jaxtyping
import pytest

Array = jaxtyping.Array
Float = jaxtyping.Float


@pytest.mark.parametrize(
    'cutoff_fn, cutoff, expected_value, expected_grad',
    [
        (
            e3x.nn.smooth_cutoff,
            1.0,
            jnp.asarray([
                1.00000000,
                0.95918936,
                0.82656544,
                0.56978285,
                0.16901326,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
            ]),
            jnp.asarray([
                -0.00000000,
                -0.41631480,
                -0.93714910,
                -1.66928570,
                -2.08658400,
                -0.00000000,
                -0.00000000,
                -0.00000000,
                -0.00000000,
                -0.00000000,
                -0.00000000,
            ]),
        ),
        (
            e3x.nn.smooth_cutoff,
            2.0,
            jnp.asarray([
                1.00000000,
                0.98994990,
                0.95918936,
                0.90583223,
                0.82656544,
                0.71653130,
                0.56978285,
                0.38259268,
                0.16901326,
                0.01407775,
                0.00000000,
            ]),
            jnp.asarray([
                -0.00000000,
                -0.10100498,
                -0.20815740,
                -0.32816050,
                -0.46857455,
                -0.63691670,
                -0.83464280,
                -1.02966130,
                -1.04329200,
                -0.35096905,
                -0.00000000,
            ]),
        ),
        (
            e3x.nn.cosine_cutoff,
            1.0,
            jnp.asarray([
                1.00000000,
                0.90450850,
                0.65450850,
                0.34549140,
                0.09549147,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
            ]),
            jnp.asarray([
                -0.00000000,
                -0.92329090,
                -1.49391620,
                -1.49391600,
                -0.92329085,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
            ]),
        ),
        (
            e3x.nn.cosine_cutoff,
            2.0,
            jnp.asarray([
                1.00000000,
                0.97552824,
                0.90450850,
                0.79389260,
                0.65450850,
                0.49999997,
                0.34549140,
                0.20610741,
                0.09549147,
                0.02447170,
                0.00000000,
            ]),
            jnp.asarray([
                -0.00000000,
                -0.24270140,
                -0.46164545,
                -0.63540053,
                -0.74695810,
                -0.78539820,
                -0.74695800,
                -0.63540050,
                -0.46164542,
                -0.24270123,
                0.00000000,
            ]),
        ),
    ],
)
def test_cutoff_fn(
    cutoff_fn: Callable[[Float[Array, '...'], float], Float[Array, '...']],
    cutoff: float,
    expected_value: Float[Array, '...'],
    expected_grad: Float[Array, '...'],
) -> None:
  x = jnp.linspace(0.0, 2.0, num=11)
  value, grad = e3x.ops.evaluate_derivatives(
      lambda x: cutoff_fn(x, cutoff), x, max_order=1
  )
  assert jnp.allclose(value, expected_value)
  assert jnp.allclose(grad, expected_grad)


@pytest.mark.parametrize(
    'cutoff_fn', [e3x.nn.smooth_cutoff, e3x.nn.cosine_cutoff]
)
def test_cutoff_fn_has_nan_safe_derivatives(
    cutoff_fn: Callable[[Float[Array, '...']], Float[Array, '...']],
) -> None:
  finfo = jnp.finfo(jnp.float32)
  x = jnp.asarray(
      [0.0, finfo.tiny, 1.0 - finfo.epsneg, 1.0, 1.0 + finfo.eps],
      dtype=jnp.float32,
  )
  for y in e3x.ops.evaluate_derivatives(cutoff_fn, x, max_order=4):
    assert jnp.all(jnp.isfinite(y))


@pytest.mark.parametrize(
    'cutoff_fn', [e3x.nn.smooth_cutoff, e3x.nn.cosine_cutoff]
)
@pytest.mark.parametrize('cutoff', [-1.0, 0.0])
def test_cutoff_fn_raises_with_invalid_cutoff(
    cutoff_fn: Callable[[Float[Array, '...'], float], Float[Array, '...']],
    cutoff: float,
) -> None:
  with pytest.raises(ValueError, match='cutoff must be larger than 0'):
    cutoff_fn(1, cutoff)
