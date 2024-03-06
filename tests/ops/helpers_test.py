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

import e3x
import jax
import jax.numpy as jnp
import jaxtyping
import pytest

Array = jaxtyping.Array
Float = jaxtyping.Float


@pytest.mark.parametrize('x', [jnp.array(4.0), jnp.array([1.0, 2.0, 0.5])])
def test_inverse_softplus(x: Float[Array, '...']) -> None:
  assert jnp.allclose(x, jax.nn.softplus(e3x.ops.inverse_softplus(x)))


def test_evaluate_derivatives() -> None:
  f = lambda x: jnp.stack((x**3, jnp.sin(x), jnp.exp(x)), axis=-1)
  x = jnp.linspace(-jnp.pi, jnp.pi, 11)
  ys = e3x.ops.evaluate_derivatives(f, x, max_order=4)
  assert jnp.allclose(ys[0], f(x), atol=1e-5)
  assert jnp.allclose(
      ys[1], jnp.stack((3 * x**2, jnp.cos(x), jnp.exp(x)), axis=-1), atol=1e-5
  )
  assert jnp.allclose(
      ys[2], jnp.stack((6 * x, -jnp.sin(x), jnp.exp(x)), axis=-1), atol=1e-5
  )
  assert jnp.allclose(
      ys[3],
      jnp.stack((jnp.full(x.shape, 6), -jnp.cos(x), jnp.exp(x)), axis=-1),
      atol=1e-5,
  )
  assert jnp.allclose(
      ys[4],
      jnp.stack((jnp.zeros_like(x), jnp.sin(x), jnp.exp(x)), axis=-1),
      atol=1e-5,
  )


def test_evaluate_derivatives_raises_with_negative_max_order() -> None:
  with pytest.raises(ValueError, match='max_order must be >= 0'):
    e3x.ops.evaluate_derivatives(lambda x: x, jnp.asarray(0), max_order=-1)
