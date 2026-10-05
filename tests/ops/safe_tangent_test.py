# Copyright 2026 The e3x Authors.
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
"""Regression coverage for finite norm derivatives at extreme scales."""

import e3x
import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize(
    'dtype, scale', [
        (jnp.float16, 1000.0),
        (jnp.float32, 1e20),
        (jnp.float32, 1e-20),
    ]
)
@pytest.mark.parametrize('axis', [None, -1, (0, 1)])
@pytest.mark.parametrize('keepdims', [False, True])
def test_norm_jvp_preserves_representable_tangent(dtype, scale, axis, keepdims):
  x = jnp.asarray([[3.0, 4.0], [6.0, 8.0]], dtype=dtype) * scale
  expected = np.sqrt(
      np.sum(np.asarray(x, dtype=np.float64)**2, axis=axis, keepdims=keepdims)
  )

  def norm(values):
    return e3x.ops.norm(values, axis=axis, keepdims=keepdims)

  for fn in (norm, jax.jit(norm)):
    value, tangent = jax.jvp(fn, (x,), (x,))
    assert value.dtype == dtype
    assert tangent.dtype == dtype
    tolerance = 2e-3 if dtype == jnp.float16 else 2e-6
    np.testing.assert_allclose(value, expected, rtol=tolerance, atol=0)
    # d/dt ||t*x|| at t=1 equals ||x||, without squaring x in its dtype.
    np.testing.assert_allclose(tangent, expected, rtol=tolerance, atol=0)


@pytest.mark.parametrize('scale', [1e20, 1e-20])
def test_coordinate_scale_derivative_matches_homogeneity(scale):
  positions = jnp.asarray([3.0, 4.0], dtype=jnp.float32) * scale

  def distance(multiplier):
    return e3x.ops.norm(multiplier * positions)

  for fn in (jax.jacfwd(distance), jax.jit(jax.jacfwd(distance))):
    np.testing.assert_allclose(
        fn(jnp.float32(1.0)), 5.0 * scale, rtol=2e-6, atol=0
    )


def test_zero_tangent_and_zero_vector_remain_finite():
  for x in (jnp.zeros(3), jnp.array([3e20, 4e20, 0.0])):
    _, tangent = jax.jvp(e3x.ops.norm, (x,), (jnp.zeros_like(x),))
    np.testing.assert_array_equal(tangent, 0.0)
