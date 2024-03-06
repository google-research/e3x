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

"""Helper functions that simplify other operations."""

from typing import Callable, List
import jax
import jax.numpy as jnp
import jaxtyping

Array = jaxtyping.Array
Float = jaxtyping.Float
Num = jaxtyping.Num


def inverse_softplus(x: Float[Array, '...']) -> Float[Array, '...']:
  """Inverse of the softplus function (useful for parameter initialization)."""
  return x + jnp.log(-jnp.expm1(-x))


def evaluate_derivatives(
    f: Callable[[Num[Array, '...']], Num[Array, '...']],
    x: Num[Array, '...'],
    max_order: int,
) -> List[Num[Array, '...']]:
  """Evaluates the function f(x) and its derivatives up to a maximum order.

  Args:
    f: Function that takes an array x as input and returns an array as output.
    x: Values at which to evaluate the function f and its derivatives.
    max_order: Maximum order of derivatives to evaluate.

  Returns:
    A list of size max_order+1 containing f(x), f'(x), f''(x), etc., with the
    i-th entry corresponding to the derivative of f of order i.
  """
  if max_order < 0:
    raise ValueError(f'max_order must be >= 0, received {max_order}')

  def derivative(f):
    """Helper function that is used instead of a direct lambda."""
    return lambda x: jax.jvp(f, (x,), (jnp.ones_like(x),))[1]

  y = [None] * (max_order + 1)
  y[0] = f(x)
  for i in range(max_order):
    f = derivative(f)  # Using a lambda directly here raises RecursionError.
    y[i + 1] = f(x)
  return y
