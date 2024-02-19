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

from typing import Callable, Union
import e3x
import jax
import jax.numpy as jnp
import jaxtyping
import pytest

Array = jaxtyping.Array
Float = jaxtyping.Float


def test__gated_linear() -> None:
  x = jnp.asarray([
      [
          [0.0, 1.0, 2.0],
          [1.0, 2.0, 3.0],
          [-1.0, -2.0, -3.0],
          [0.0, 0.4, 0.0],
      ],
      [
          [1.3, 0.1, -0.4],
          [2.3, -0.5, 1.3],
          [-0.1, -1.3, 3.1],
          [0.1, 0.1, 0.2],
      ],
  ])
  expected = x[0:1, 0:1, :] ** 2 * x
  assert jnp.allclose(
      e3x.nn.activations._gated_linear(g=lambda x: x**2, x=x),
      expected,
      atol=1e-5,
  )


def test__gated_linear_raises_with_invalid_shape() -> None:
  with pytest.raises(
      ValueError, match='shape of x must have at least three dimensions'
  ):
    e3x.nn.activations._gated_linear(g=lambda x: 1, x=jnp.asarray(0))


@pytest.mark.parametrize(
    'func, expected_func',
    [
        (e3x.nn.relu, jax.nn.relu),
        (e3x.nn.leaky_relu, jax.nn.leaky_relu),
        (e3x.nn.elu, jax.nn.elu),
        (e3x.nn.selu, jax.nn.selu),
        (e3x.nn.celu, jax.nn.celu),
        (e3x.nn.silu, jax.nn.silu),
        (e3x.nn.swish, jax.nn.swish),
        (
            lambda x: e3x.nn.gelu(x, approximate=True),
            lambda x: jax.nn.gelu(x, approximate=True),
        ),
        (
            lambda x: e3x.nn.gelu(x, approximate=False),
            lambda x: jax.nn.gelu(x, approximate=False),
        ),
        (e3x.nn.mish, lambda x: x * jnp.tanh(jax.nn.softplus(x))),
        (e3x.nn.serf, lambda x: x * jax.lax.erf((jax.nn.softplus(x)))),
        (
            e3x.nn.shifted_softplus,
            lambda x: jax.nn.softplus(x) - jax.nn.softplus(0),
        ),
        (e3x.nn.hard_tanh, jax.nn.hard_tanh),
        (e3x.nn.soft_sign, jax.nn.soft_sign),
        (e3x.nn.relu6, jax.nn.relu6),
        (e3x.nn.hard_silu, jax.nn.hard_silu),
        (e3x.nn.hard_swish, jax.nn.hard_swish),
        (e3x.nn.bent_identity, lambda x: x + (jnp.sqrt(x**2 + 1) - 1) / 2),
    ],
)
def test_activations(
    func: Callable[
        [
            Union[
                Float[Array, '... 1 (max_degree+1)**2 num_features'],
                Float[Array, '... 2 (max_degree+1)**2 num_features'],
            ]
        ],
        Union[
            Float[Array, '... 1 (max_degree+1)**2 num_features'],
            Float[Array, '... 2 (max_degree+1)**2 num_features'],
        ],
    ],
    expected_func: Callable[[Float[Array, '...']], Float[Array, '...']],
) -> None:
  x = jnp.expand_dims(jnp.linspace(-10.0, 10.0, 1000), axis=(0, 1))
  value, grad = e3x.ops.evaluate_derivatives(func, x, max_order=1)
  expected_value, expected_grad = e3x.ops.evaluate_derivatives(
      expected_func, x, max_order=1
  )
  assert jnp.allclose(value, expected_value, atol=1e-5)
  assert jnp.allclose(grad, expected_grad, atol=1e-5)


@pytest.mark.parametrize(
    'func',
    [
        e3x.nn.relu,
        e3x.nn.leaky_relu,
        e3x.nn.elu,
        e3x.nn.selu,
        e3x.nn.celu,
        e3x.nn.silu,
        e3x.nn.swish,
        e3x.nn.gelu,
        e3x.nn.mish,
        e3x.nn.serf,
        e3x.nn.shifted_softplus,
        e3x.nn.hard_tanh,
        e3x.nn.soft_sign,
        e3x.nn.relu6,
        e3x.nn.hard_silu,
        e3x.nn.hard_swish,
        e3x.nn.bent_identity,
    ],
)
def test_activations_have_nan_safe_derivatives(
    func: Callable[
        [
            Union[
                Float[Array, '... 1 (max_degree+1)**2 num_features'],
                Float[Array, '... 2 (max_degree+1)**2 num_features'],
            ]
        ],
        Union[
            Float[Array, '... 1 (max_degree+1)**2 num_features'],
            Float[Array, '... 2 (max_degree+1)**2 num_features'],
        ],
    ]
) -> None:
  finfo = jnp.finfo(jnp.float32)
  x = jnp.expand_dims(
      jnp.asarray(
          [
              -2 * finfo.eps,
              -finfo.eps,
              -2 * finfo.tiny,
              -finfo.tiny,
              0.0,
              finfo.tiny,
              2 * finfo.tiny,
              finfo.eps,
              2 * finfo.eps,
          ],
          dtype=jnp.float32,
      ),
      axis=(0, 1),
  )
  for y in e3x.ops.evaluate_derivatives(func, x, max_order=4):
    assert jnp.all(jnp.isfinite(y))


@pytest.mark.parametrize(
    'func',
    [
        e3x.nn.relu,
        e3x.nn.leaky_relu,
        e3x.nn.elu,
        e3x.nn.selu,
        e3x.nn.celu,
        e3x.nn.silu,
        e3x.nn.swish,
        e3x.nn.gelu,
        e3x.nn.mish,
        e3x.nn.serf,
        e3x.nn.shifted_softplus,
        e3x.nn.hard_tanh,
        e3x.nn.soft_sign,
        e3x.nn.relu6,
        e3x.nn.hard_silu,
        e3x.nn.hard_swish,
        e3x.nn.bent_identity,
    ],
)
def test_activations_preserve_equivariance(
    func: Callable[
        [
            Union[
                Float[Array, '... 1 (max_degree+1)**2 num_features'],
                Float[Array, '... 2 (max_degree+1)**2 num_features'],
            ]
        ],
        Union[
            Float[Array, '... 1 (max_degree+1)**2 num_features'],
            Float[Array, '... 2 (max_degree+1)**2 num_features'],
        ],
    ],
    num_parity: int = 2,
    max_degree: int = 2,
    features: int = 8,
) -> None:
  x_key, rot_key = jax.random.split(jax.random.PRNGKey(0), num=2)
  x = jax.random.normal(x_key, (num_parity, (max_degree + 1) ** 2, features))
  # Random rotation matrix.
  rot = e3x.so3.random_rotation(rot_key)
  wigner_d = e3x.so3.wigner_d(rot, max_degree=max_degree)
  # Rotated and reflected features (for checking equivariance).
  x_rot = e3x.nn.rotate(x, wigner_d)
  x_ref = e3x.nn.reflect(x)
  # Apply activation function.
  y = func(x)
  y_rot = func(x_rot)
  y_ref = func(x_ref)
  # Check for equivariance.
  assert jnp.allclose(e3x.nn.rotate(y, wigner_d), y_rot, atol=1e-5)
  assert jnp.allclose(e3x.nn.reflect(y), y_ref, atol=1e-5)
