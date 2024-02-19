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

"""Custom initializers.

This module contains implementations of common initializers for the
:class:`Tensor <e3x.nn.modules.Tensor>` layer. Standard
initialzers cannot be used, because the number of inputs and outputs cannot
be inferred from the shape of the kernel (instead, all valid coupling paths
are checked and the number of inputs and outputs determined accordingly).
"""

import itertools
from typing import Any, List, Literal, Protocol, Sequence, Tuple, Union
import jax
from jax._src.nn.initializers import _complex_truncated_normal
from jax._src.nn.initializers import _complex_uniform
import jax.numpy as jnp
import jaxtyping

Array = jaxtyping.Array
Bool = jaxtyping.Bool
Float = jaxtyping.Float
Shaped = jaxtyping.Shaped
UInt32 = jaxtyping.UInt32
Dtype = Any
Shape = Sequence[Union[int, Any]]
PRNGKey = UInt32[Array, '2']


def _parity_degree_index_parity_list(
    num_parity: int, num_degree: int
) -> List[Tuple[int, int, int]]:
  """Compute list of parity degree index combinations with associated parity.

  Args:
    num_parity: Number of parity channels (must be 1 or 2).
    num_degree: Number of degree channels.

  Returns:
    A list of tuples (p, l, d), where p is the parity index (0 or 1), l is the
    degree index (0...num_degree-1) and d is the associated parity of the
    features (even -> d=0 or odd -> d=1). When num_parity=2, p is always equal
    to d (by convention/design). When num_parity=1, even and odd features are
    stored in the same parity channel for memory efficiency (to avoid
    zero-padding) and features are even/odd if the degree is even/odd.

  Raises:
    ValueError: If num_parity is not 1 or 2.
  """
  if num_parity == 2:  # All entries for p=0 are even and for p=1 are odd.
    even = [(0, l, 0) for l in range(num_degree)]
    odd = [(1, l, 1) for l in range(num_degree)]
    return even + odd
  elif num_parity == 1:  # Entries are even/odd if l is even/odd.
    return [(0, l, l % 2) for l in range(num_degree)]
  else:
    raise ValueError(f'num_parity should be 1 or 2, received {num_parity}')


def _compute_tensor_fans_and_mask(
    shape: Shape, dtype: Dtype = jnp.float_
) -> Tuple[
    Float[Array, '1 1 1 1 S3 L3 1'],
    Float[Array, 'S1 L1 S2 L2 1 1 1'],
    Bool[Array, 'S1 L1 S2 L2 S3 L3 F'],
]:
  """Compute effective input and output sizes for tensor product kernels.

  The shape of the tensor kernel is always (S1, L1, S2, L2, S3, L3, F), where
  each S? is either 1 or 2, L? is short for (max_degree?+1)**2, and F denotes
  the number of features.

  This function also computes a mask for zeroing out parameters of "forbidden"
  coupling paths. In principle, the value of these parameters does not matter,
  because the corresponding paths need to be zero'd out in the tensor product
  anyway. However, it may be better to initialize the parameters to zero anyway
  for two reasons: (1) in case L1/L2-regularization is used, non-zero parameters
  (that are irrelevant) would contribute to the penalty, which might influence
  training in unwanted ways and (2) it might be confusing for users why these
  parameters have non-zero values.

  Args:
    shape: Shape of the tensor kernel (see above).
    dtype: Dtype of the returned fan_in/fan_out results.

  Returns:
    A tuple (fan_in, fan_out, mask), which contain the number of input paths,
    output paths, and a mask for zeroing out parity-forbidden coupling paths.
  """
  # Check that shape has the correct length: 3 dims for parity (p), 3 dims for
  # degree (d), 1 dim for features (f), stored (p1,l1,p2,l2,p3,l3,f).
  if len(shape) != 7:
    raise ValueError(f'shape should be len=7, received len={len(shape)}')
  # Initialize mask and fans.
  fan_in = jnp.zeros(
      (1, 1, 1, 1, *shape[-3:-1], 1), jax.dtypes.canonicalize_dtype(dtype)
  )
  fan_out = jnp.zeros(
      (*shape[:-3], 1, 1, 1), jax.dtypes.canonicalize_dtype(dtype)
  )
  mask = jnp.full(shape, True)
  # Generate lists of parity degree indices p, l with associated parity d.
  all_pld1 = _parity_degree_index_parity_list(
      num_parity=shape[0], num_degree=shape[1]
  )
  all_pld2 = _parity_degree_index_parity_list(
      num_parity=shape[2], num_degree=shape[3]
  )
  all_pld3 = _parity_degree_index_parity_list(
      num_parity=shape[4], num_degree=shape[5]
  )
  # Set values for all possible combinations of parity and degree indices.
  for pld1, pld2, pld3 in itertools.product(all_pld1, all_pld2, all_pld3):
    p1, l1, d1 = pld1
    p2, l2, d2 = pld2
    p3, l3, d3 = pld3
    if (d1 + d2) % 2 != d3 or not abs(l1 - l2) <= l3 <= l1 + l2:  # Forbidden.
      mask = mask.at[p1, l1, p2, l2, p3, l3, :].set(False)
    else:  # Allowed combination.
      fan_in = fan_in.at[0, 0, 0, 0, p3, l3, :].add(1)
      fan_out = fan_out.at[p1, l1, p2, l2, 0, 0, :].add(1)
  return fan_in, fan_out, mask


_valid_distributions = ('truncated_normal', 'normal', 'uniform')

Distribution = Literal[_valid_distributions]


def _random_array(
    key: PRNGKey,
    shape: Shape,
    distribution: Distribution = 'truncated_normal',
    variance: Union[float, Float[Array, '*shape']] = 1.0,
    dtype: Dtype = jnp.float_,
) -> Shaped[Array, '*shape']:
  """Draw a random array with mean zero and given variance from a distribution.

  Args:
    key: The PRNGKey used as the random key.
    shape: Shape of the output array.
    distribution: Random distribution to draw from, supported values are
      'truncated_normal', 'normal', and 'uniform'.
    variance: Desired variance of the output. Can be either a simple float, or
      an array that is broadcastable to the desired output shape (in case
      individual entries of the array should have different variance).
    dtype: The dtype for the returned values.

  Returns:
    An array of the desired shape containing random numbers of the desired
    variance drawn from the desired distribution.

  Raises:
    ValueError: If distribution is invalid.
  """
  if distribution == 'truncated_normal':
    if jnp.issubdtype(dtype, jnp.floating):
      # 0.879... is the stddev of a standard normal truncated to (-2, 2).
      stddev = jnp.sqrt(variance) / jnp.array(0.87962566103423978, dtype)
      return stddev * jax.random.truncated_normal(
          key=key, lower=-2, upper=2, shape=shape, dtype=dtype
      )
    else:
      # 0.953... is the stddev of a complex standard normal truncated to 2.
      stddev = jnp.sqrt(variance) / jnp.array(0.95311164380491208, dtype)
      return stddev * _complex_truncated_normal(
          key=key,
          upper=2,
          shape=jax.core.as_named_shape(shape),
          dtype=dtype,
      )
  elif distribution == 'normal':
    return jnp.sqrt(variance) * jax.random.normal(
        key=key, shape=shape, dtype=dtype
    )
  elif distribution == 'uniform':
    if jnp.issubdtype(dtype, jnp.floating):
      return jnp.sqrt(3 * variance) * jax.random.uniform(
          key=key, shape=shape, dtype=dtype, minval=-1, maxval=1
      )
    else:
      return jnp.sqrt(variance) * _complex_uniform(
          key=key, shape=jax.core.as_named_shape(shape), dtype=dtype
      )
  else:
    raise ValueError(f"invalid distribution '{distribution}' for _random_array")


_valid_modes = ('fan_in', 'fan_out', 'fan_avg')

Mode = Literal[_valid_modes]


class InitializerFn(Protocol):
  """Protocol for all standard initializer functions."""

  def __call__(
      self, key: PRNGKey, shape: Shape, dtype: Dtype = jnp.float_
  ) -> Shaped[Array, '*shape']:
    ...


def tensor_variance_scaling(
    scale: float,
    mode: Mode,
    distribution: Distribution,
    dtype: Dtype = jnp.float_,
) -> InitializerFn:
  """Variance scaling initializer for tensor product kernels.

  Equivalent to
  :func:`variance_scaling <jax.nn.initializers.variance_scaling>`, but for
  tensor product kernels.

  Args:
    scale: Scaling factor for the variance.
    mode: How the variance is computed, supported values are `'fan_in'`,
      `'fan_out'`, and `'fan_avg'`.
    distribution: Random distribution to draw from, supported values are
      `'truncated_normal'`, `'normal'`, and `'uniform'`.
    dtype: The desired dtype of the parameters.

  Returns:
    An initializer function.

  Raises:
    ValueError: If mode or distribution is invalid.
  """

  def init(
      key: PRNGKey, shape: Shape, dtype: Dtype = dtype
  ) -> Float[Array, '*Shape']:
    dtype = jax.dtypes.canonicalize_dtype(dtype)
    fan_in, fan_out, mask = _compute_tensor_fans_and_mask(shape)
    if mode == 'fan_in':
      denominator = fan_in
    elif mode == 'fan_out':
      denominator = fan_out
    elif mode == 'fan_avg':
      denominator = (fan_in + fan_out) / 2
    else:
      raise ValueError(
          f"invalid mode '{mode}' for variance scaling initializer"
      )
    # Prevent division by zero (values get masked out anyway).
    denominator = jnp.where(
        denominator > 0, denominator, jnp.ones_like(denominator)
    )
    variance = jnp.array(scale / denominator, dtype=dtype)

    params = _random_array(
        key=key,
        shape=shape,
        distribution=distribution,
        variance=variance,
        dtype=dtype,
    )

    return jnp.where(mask, params, 0)

  return init


def tensor_glorot_uniform(
    dtype: Dtype = jnp.float_,
) -> InitializerFn:
  """Glorot uniform initializer for tensor product kernels.

  Equivalent to :func:`glorot_uniform <jax.nn.initializers.glorot_uniform>`,
  but for tensor product kernels.

  Args:
    dtype: The desired dtype of the parameters.

  Returns:
    An initializer function.
  """
  return tensor_variance_scaling(1.0, 'fan_avg', 'uniform', dtype=dtype)


tensor_xavier_uniform = tensor_glorot_uniform


def tensor_glorot_normal(
    dtype: Dtype = jnp.float_,
) -> InitializerFn:
  """Glorot normal initializer for tensor product kernels.

  Equivalent to :func:`glorot_normal <jax.nn.initializers.glorot_normal>`,
  but for tensor product kernels.

  Args:
    dtype: The desired dtype of the parameters.

  Returns:
    An initializer function.
  """
  return tensor_variance_scaling(
      1.0, 'fan_avg', 'truncated_normal', dtype=dtype
  )


tensor_xavier_normal = tensor_glorot_normal


def tensor_lecun_uniform(
    dtype: Dtype = jnp.float_,
) -> InitializerFn:
  """Lecun uniform initializer for tensor product kernels.

  Equivalent to :func:`lecun_uniform <jax.nn.initializers.lecun_uniform>`,
  but for tensor product kernels.

  Args:
    dtype: The desired dtype of the parameters.

  Returns:
    An initializer function.
  """
  return tensor_variance_scaling(1.0, 'fan_in', 'uniform', dtype=dtype)


def tensor_lecun_normal(
    dtype: Dtype = jnp.float_,
) -> InitializerFn:
  """Lecun normal initializer for tensor product kernels.

  Equivalent to :func:`lecun_normal <jax.nn.initializers.lecun_normal>`, but
  for tensor product kernels.

  Args:
    dtype: The desired dtype of the parameters.

  Returns:
    An initializer function.
  """
  return tensor_variance_scaling(1.0, 'fan_in', 'truncated_normal', dtype=dtype)


def tensor_he_uniform(
    dtype: Dtype = jnp.float_,
) -> InitializerFn:
  """He uniform initializer for tensor product kernels.

  Equivalent to :func:`he_uniform <jax.nn.initializers.he_uniform>`, but for
  tensor product kernels.

  Args:
    dtype: The desired dtype of the parameters.

  Returns:
    An initializer function.
  """
  return tensor_variance_scaling(2.0, 'fan_in', 'uniform', dtype=dtype)


tensor_kaiming_uniform = tensor_he_uniform


def tensor_he_normal(
    dtype: Dtype = jnp.float_,
) -> InitializerFn:
  """He normal initializer for tensor product kernels.

  Equivalent to :func:`he_normal <jax.nn.initializers.he_normal>`, but for
  tensor product kernels.

  Args:
    dtype: The desired dtype of the parameters.

  Returns:
    An initializer function.
  """
  return tensor_variance_scaling(2.0, 'fan_in', 'truncated_normal', dtype=dtype)


tensor_kaiming_normal = tensor_he_normal


def _fused_tensor_init(
    distribution: Distribution,
    scale: Union[float, Float[Array, '*shape']] = 1.0,
    mask: Union[bool, Array] = True,
    dtype: Dtype = jnp.float_,
) -> InitializerFn:
  """General initializer for fused tensor product kernels."""

  def init(
      key: PRNGKey, shape: Shape, dtype: Dtype = dtype
  ) -> Float[Array, '*shape']:
    dtype = jax.dtypes.canonicalize_dtype(dtype)
    params = _random_array(
        key=key,
        shape=shape,
        distribution=distribution,
        variance=scale,
        dtype=dtype,
    )
    return jnp.where(mask, params, 0)

  return init


class FusedTensorInitializerFn(Protocol):
  """Protocol for fused tensor initializer functions."""

  def __call__(
      self,
      scale: Union[float, Float[Array, '*Shape']] = 1.0,
      mask: Union[bool, Array] = True,
      dtype: Dtype = jnp.float_,
  ) -> InitializerFn:
    ...


def fused_tensor_normal(
    scale: Union[float, Float[Array, '*Shape']] = 1.0,
    mask: Union[bool, Array] = True,
    dtype: Dtype = jnp.float_,
) -> InitializerFn:
  """Initializer for fused tensor product kernels with normal distribution.

  Args:
    scale: Scaling factor for the variance.
    mask: Mask for zeroing unused parameters.
    dtype: The desired dtype of the parameters.

  Returns:
    An initializer function.
  """
  return _fused_tensor_init(
      distribution='truncated_normal', scale=scale, mask=mask, dtype=dtype
  )


def fused_tensor_uniform(
    scale: Union[float, Float[Array, '*Shape']] = 1.0,
    mask: Union[bool, Array] = True,
    dtype: Dtype = jnp.float_,
) -> InitializerFn:
  """Initializer for fused tensor product kernels with uniform distribution.

  Args:
    scale: Scaling factor for the variance.
    mask: Mask for zeroing unused parameters.
    dtype: The desired dtype of the parameters.

  Returns:
    An initializer function.
  """
  return _fused_tensor_init(
      distribution='uniform', scale=scale, mask=mask, dtype=dtype
  )
