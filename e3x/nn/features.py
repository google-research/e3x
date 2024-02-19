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

"""Utility functions for transforming features."""

import math
from typing import Any, Optional, Sequence, Union
import jax.numpy as jnp
import jaxtyping

Array = jaxtyping.Array
Bool = jaxtyping.Bool
Float = jaxtyping.Float
Shape = Sequence[Union[int, Any]]


def _extract_max_degree_and_check_shape(shape: Shape) -> int:
  """Extract max_degree from feature shape and check for valid sizes.

  Valid feature shapes are at least three-dimensional:
  (..., 1, (max_degree+1)**2, num_features)  for SO(3) features, and
  (..., 2, (max_degree+1)**2, num_features)  for O(3) features, with max_degree
  being any positive integer or 0.

  Args:
    shape: The input shape to be checked.

  Returns:
    The value of `max_degree` extracted from shape.

  Raises:
    ValueError: If the shape is invalid.
  """

  # Check that shape has parity, degree, and feature channels.
  if len(shape) < 3:
    raise ValueError(
        f'shape of features must have at least length 3, received shape {shape}'
    )

  # Check that axis -3 (parity channels) has the correct size (1 or 2).
  if not (shape[-3] == 1 or shape[-3] == 2):
    raise ValueError(
        f'expected 1 or 2 for axis -3 of feature shape, received shape{shape}'
    )

  # Extract max_degree from size of axis -2 (degree channel).
  max_degree = round(math.sqrt(shape[-2]) - 1)

  # Check that axis -2 (degree channel) has a valid size.
  expected_size = (max_degree + 1) ** 2
  if shape[-2] != expected_size:
    raise ValueError(
        f'received invalid size {shape[-2]} for axis -2 of '
        f'feature shape, closest valid size is {expected_size}'
    )

  return max_degree


def even_degree_mask(max_degree: int) -> Bool[Array, '(max_degree+1)**2 1']:
  """Returns a mask than only keeps features with even degrees."""
  degrees = jnp.arange(max_degree + 1)
  return jnp.expand_dims(
      jnp.repeat(
          degrees % 2 == 0,
          2 * degrees + 1,
          total_repeat_length=(max_degree + 1) ** 2,
      ),
      axis=-1,
  )


def odd_degree_mask(max_degree: int) -> Bool[Array, '(max_degree+1)**2 1']:
  """Returns a mask than only keeps features with odd degrees."""
  return jnp.logical_not(even_degree_mask(max_degree))


def rotate(
    x: Union[
        Float[Array, '... 1 (max_degree+1)**2 num_features'],
        Float[Array, '... 2 (max_degree+1)**2 num_features'],
    ],
    wigner_d_matrix: Union[
        Float[Array, '(max_degree+1)**2 (max_degree+1)**2'],
        Float[Array, '... (max_degree+1)**2 (max_degree+1)**2'],
    ],
) -> Union[
    Float[Array, '... 1 (max_degree+1)**2 num_features'],
    Float[Array, '... 2 (max_degree+1)**2 num_features'],
]:
  """Rotates the input features with the given Wigner D-matrix (or -matrices).

  Args:
    x: Input features with shape (..., S, (L+1)**2, F).
    wigner_d_matrix: Wigner D-matrix with shape ((L+1)**2, (L+1)**2) or a batch
      of Wigner D-matrices with shape  (..., (L+1)**2, (L+1)**2).

  Returns:
    The rotated features.
  """
  max_degree = _extract_max_degree_and_check_shape(x.shape)
  dim = (max_degree + 1) ** 2

  if wigner_d_matrix.shape[-2:] != (dim, dim):
    raise ValueError(
        f'expected shape (..., {dim}, {dim}) for Wigner-D matrix, '
        f'received shape {wigner_d_matrix.shape}'
    )

  if wigner_d_matrix.ndim == 2:
    return jnp.einsum(
        '...slf,lm->...smf', x, wigner_d_matrix, optimize='optimal'
    )
  else:
    if x.shape[:-3] != wigner_d_matrix.shape[:-2]:
      raise ValueError(
          f'x and wigner_d_matrix have incompatible shapes {x.shape} and'
          f' {wigner_d_matrix.shape}'
      )
    return jnp.einsum(
        '...slf,...lm->...smf', x, wigner_d_matrix, optimize='optimal'
    )


def reflect(
    x: Union[
        Float[Array, '... 1 (max_degree+1)**2 num_features'],
        Float[Array, '... 2 (max_degree+1)**2 num_features'],
    ],
) -> Union[
    Float[Array, '... 1 (max_degree+1)**2 num_features'],
    Float[Array, '... 2 (max_degree+1)**2 num_features'],
]:
  """Reflects the input features into their mirror image.

  Args:
    x: Input features with shape (..., S, (L+1)**2, F).

  Returns:
    The reflected features (with the same shape as x).
  """
  max_degree = _extract_max_degree_and_check_shape(x.shape)

  if x.shape[-3] == 2:  # Odd parity features are all in axis 1.
    x = x.at[..., 1, :, :].multiply(-1)
  else:  # Odd parity features have odd degrees (only proper tensors).
    for l in range(1, max_degree + 1, 2):
      x = x.at[..., l**2 : (l + 1) ** 2, :].multiply(-1)
  return x


def change_max_degree_or_type(
    x: Union[
        Float[Array, '... 1 (max_degree+1)**2 num_features'],
        Float[Array, '... 2 (max_degree+1)**2 num_features'],
    ],
    max_degree: Optional[int] = None,
    include_pseudotensors: Optional[bool] = None,
) -> Union[
    Float[Array, '... 1 (max_degree+1)**2 num_features'],
    Float[Array, '... 2 (max_degree+1)**2 num_features'],
]:
  r"""Changes the maximum degree and/or type of features.

  When changing the maximum degree to a larger value, the features are padded
  with zeros to give them the correct shape. When changing the maximum degree to
  a smaller value, the superfluous feature channels are discarded. To change the
  type of features, include_pseudotensors must be specified as True (for both
  tensor and pseudotensor features) or False (for only tensor features). Similar
  to changing the maximum degree, the shape change is achieved either by padding
  with zeros or discarding the superfluous feature channels.

  Args:
    x: Input features with shape (..., S, (L+1)**2, F).
    max_degree: New max_degree (if ``None``, max_degree is auto-determined from
      the shape of ``x``)
    include_pseudotensors: If ``True``, both tensor and pseudotensor features
      are returned, if ``False``, only tensor features are returned (if
      ``None``, the type of features is auto-determined from the shape of
      ``x``).

  Returns:
    The reshaped features.
  """
  # Determine max_degree of features and whether they contain pseudotensors.
  in_max_degree = _extract_max_degree_and_check_shape(x.shape)
  input_has_pseudotensors = x.shape[-3] != 1

  # Determine desired max_degree and whether pseudotensors should be included.
  max_degree = in_max_degree if max_degree is None else max_degree
  include_pseudotensors = (
      input_has_pseudotensors
      if include_pseudotensors is None
      else include_pseudotensors
  )

  # Increase max_degree (slice).
  if max_degree < in_max_degree:
    x = x[..., : (max_degree + 1) ** 2, :]

  # Decrease max_degree (pad with zeros).
  elif max_degree > in_max_degree:
    pad_with = [(0, 0)] * x.ndim
    pad_with[-2] = (0, (max_degree + 1) ** 2 - (in_max_degree + 1) ** 2)
    x = jnp.pad(x, pad_with, mode='constant', constant_values=0)

  # Remove existing pseudotensor channel.
  if input_has_pseudotensors and not include_pseudotensors:
    x = jnp.concatenate(
        [
            x[..., l % 2 : l % 2 + 1, l**2 : (l + 1) ** 2, :]
            for l in range(max_degree + 1)
        ],
        axis=-2,
    )

  # Add non-existing pseudotensor channel.
  elif include_pseudotensors and not input_has_pseudotensors:
    # Even parity features.
    e = jnp.concatenate(
        [
            x[..., 0:1, l**2 : (l + 1) ** 2, :]
            if l % 2 == 0
            else jnp.zeros_like(x[..., 0:1, l**2 : (l + 1) ** 2, :])
            for l in range(max_degree + 1)
        ],
        axis=-2,
    )
    # Odd parity features.
    o = jnp.concatenate(
        [
            x[..., 0:1, l**2 : (l + 1) ** 2, :]
            if l % 2 != 0
            else jnp.zeros_like(x[..., 0:1, l**2 : (l + 1) ** 2, :])
            for l in range(max_degree + 1)
        ],
        axis=-2,
    )
    # Combined even and odd parity features.
    x = jnp.concatenate((e, o), axis=-3)

  return x


def add(
    *inputs: Union[
        Float[Array, '... 1 (max_degree+1)**2 num_features'],
        Float[Array, '... 2 (max_degree+1)**2 num_features'],
    ],
) -> Union[
    Float[Array, '... 1 (max_degree+1)**2 num_features'],
    Float[Array, '... 2 (max_degree+1)**2 num_features'],
]:
  r"""Add multiple input features.

  Adding :math:`\mathrm{O}(3)` or :math:`\mathrm{SO}(3)` features is not
  trivial, because only features with the same parity and degree may be added to
  each other. When features with different max_degree (or type) are added
  together naively, the operation either fails (because the shapes do not
  match), or worse, it may work, but produce incorrect results (because
  dimensions with size 1 are incorrectly broadcasted). This function makes sure
  features of different shape are "broadcasted" correctly.

  Args:
    *inputs: Input features.

  Returns:
    The sum of all input features.
  """
  if inputs[0].ndim < 3:
    raise ValueError(
        'all inputs must be at least three-dimensional, received inputs with '
        f'shape {inputs[0].shape} ({inputs[0].ndim} dimensions) at position 0'
    )

  # Determine output shape and check for inconsistencies.
  max_degree = 0
  has_pseudotensors = False
  features = inputs[0].shape[-1]
  batch_shape = inputs[0].shape[:-3]
  dtype = inputs[0].dtype
  for i, x in enumerate(inputs):
    # Checks for consistent shapes and dtypes.
    if x.shape[:-3] != batch_shape:
      raise ValueError(
          'all inputs must have the same leading dimensions (batch shape), '
          f'received input with leading dimensions {x.shape[:-3]} at position '
          f'{i}, expected {batch_shape}'
      )
    if x.shape[-1] != features:
      raise ValueError(
          'all inputs must have the same number of features, received input '
          f'with {x.shape[-1]} features at position {i}, expected {features} '
          'features'
      )
    if x.dtype != dtype:
      raise ValueError(
          'all inputs must have the same dtype, received input with dtype='
          f'{x.dtype} at position {i}, expected dtype={dtype}'
      )
    # General shape check.
    try:
      x_max_degree = _extract_max_degree_and_check_shape(x.shape)
    except ValueError as exception:
      raise ValueError(
          f'input at position {i} has invalid shape {x.shape}'
      ) from exception
    max_degree = max(max_degree, x_max_degree)
    has_pseudotensors = has_pseudotensors or x.shape[-3] == 2

  # Perform addition.
  y = jnp.zeros_like(
      inputs[0],
      dtype=dtype,
      shape=(
          *batch_shape,
          2 if has_pseudotensors else 1,
          (max_degree + 1) ** 2,
          features,
      ),
  )
  for x in inputs:
    y += change_max_degree_or_type(
        x, max_degree=max_degree, include_pseudotensors=has_pseudotensors
    )
  return y
