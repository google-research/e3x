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

"""Numerically stable and nan-safe implementations of common operations.

In some cases, the operation itself is unproblematic, but certain inputs may
result in NaNs when taking derivatives. The implementations in this module do
not suffer from such issues.
"""

import functools
from typing import Any, Optional, Tuple, Union
import jax
import jax.numpy as jnp
import jaxtyping

Array = jaxtyping.Array
Float = jaxtyping.Float


@functools.partial(jax.custom_jvp, nondiff_argnums=(1, 2))
def norm(
    x: Float[Array, '*input_shape'],
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> Union[Float[Array, '*reduced_shape'], Float[Array, '*input_shape']]:
  """L2-norm of x along the specified axis.

  This functions is equivalent to calling jax.numpy.linalg.norm(x, axis=axis,
  keepdims=keepdims), with the exception that it is more numerically stable
  (very large absolute values in x will not cause overflow and very small
  absolute values in x will still lead to stable and well-defined derivatives).

  Args:
    x: Input array.
    axis: Axis or axes along which the L2-norm is computed. The default,
      axis=None, will compute the norm of all elements of the input array (as if
      it was one large vector). If axis is negative it counts from the last to
      the first axis.
    keepdims: If this is set to True, the axes which are reduced are left in the
      result as dimensions with size one. With this option, the result will
      broadcast correctly against the input array.

  Returns:
    The L2-norm of x along axis. An array with the same shape as x, with the
    specified axis removed. If x is a 0-d array, or if axis is None, a scalar is
    returned.
  """
  a = jnp.maximum(jnp.max(jnp.abs(x)), jnp.finfo(x.dtype).tiny)
  b = x / a
  return a * jnp.sqrt(jnp.sum(b * b, axis=axis, keepdims=keepdims))


@norm.defjvp
def _norm_jvp_impl(
    axis: Optional[Union[int, Tuple[int, ...]]],
    keepdims: bool,
    primals: Any,
    tangents: Any,
) -> Tuple[Any, Any]:
  """JVP implementation for norm function, should not be called directly."""
  (x,) = primals
  (x_dot,) = tangents
  primal_out = norm(x, axis=axis, keepdims=keepdims)
  masked_primal_out = jnp.where(primal_out > 0, primal_out, 1)
  if not keepdims and axis is not None:
    masked_primal_out = jnp.expand_dims(masked_primal_out, axis=axis)
  tangent_out = jnp.sum(
      x_dot * x / masked_primal_out,
      axis=axis,
      keepdims=keepdims,
  )
  return primal_out, tangent_out


def normalize_and_return_norm(
    x: Float[Array, '...'],
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> Tuple[Float[Array, '...'], Float[Array, '...']]:
  """Normalize x using the L2-norm along the specified axis and return its norm.

  If x has a norm of almost zero, it is left unnormalized, because normalization
  becomes numerically unstable.

  Args:
    x: Input array.
    axis: Axis or axes along which the L2-norm is computed. The default,
      axis=None, will compute the norm of all elements of the input array (as if
      it was one large vector). If axis is negative it counts from the last to
      the first axis.
    keepdims: If this is set to True, the axes which are reduced for computing
      the norm are left in the norm result as dimensions with size one.

  Returns:
    A tuple consisting of the normalized array and its norm.
  """
  n = norm(x, axis=axis, keepdims=True)
  # If n * n is smaller than finfo.tiny, the gradient becomes unstable.
  y = x / jnp.where(n * n > jnp.finfo(x.dtype).tiny, n, 1)
  if not keepdims:
    n = jnp.squeeze(n, axis=axis)
  return y, n


def normalize(
    x: Float[Array, '...'], axis: Optional[Union[int, Tuple[int, ...]]] = None
) -> Float[Array, '...']:
  """Normalize x using the L2-norm of x along the specified axis.

  If x has a norm of almost zero, it is left unnormalized, because normalization
  becomes numerically unstable.

  Args:
    x: Input array.
    axis: Axis or axes along which the L2-norm is computed. The default,
      axis=None, will compute the norm of all elements of the input array (as if
      it was one large vector). If axis is negative it counts from the last to
      the first axis.

  Returns:
    The normalized array with the same shape as x.
  """
  y, _ = normalize_and_return_norm(x, axis=axis, keepdims=True)
  return y
