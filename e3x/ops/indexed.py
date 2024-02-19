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

r"""Utility functions for performing indexed operations.

.. _IndexedOps:
"""

import functools
from typing import Any, Literal, Optional, Sequence, Tuple, Union
import jax
from jax import lax
import jax.numpy as jnp
import jaxtyping

Array = jaxtyping.Array
Bool = jaxtyping.Bool
Float = jaxtyping.Float
Integer = jaxtyping.Integer
Num = jaxtyping.Num
Shaped = jaxtyping.Shaped
Shape = Sequence[Union[int, Any]]
Dtype = Any
_valid_index_types = ('dense', 'sparse')
IndexType = Literal[_valid_index_types]


def _determine_index_type(
    *,
    adj_idx: Optional[Integer[Array, '... N M']] = None,
    dst_idx: Optional[Integer[Array, '... P']] = None,
    src_idx: Optional[Integer[Array, '... P']] = None,
) -> IndexType:
  """Helper function to determine which IndexType is used.

  If both dst_idx and src_idx are given, this function also checks that their
  shapes are consistent.

  Args:
    adj_idx: Adjacency indices, or None.
    dst_idx: Destination indices, or None.
    src_idx: Source indices, or None.

  Returns:
    IndexType 'dense' if adj_idx is not None and dst_idx and src_idx are None,
    or IndexType 'sparse' if at least one of dst_idx or src_idx is not None and
    adj_idx is None.

  Raises:
    RuntimeError: If neither dense nor sparse index lists are provided, or if
      both are provided.
    ValueError: If the shapes of dst_idx and src_idx do not match.
  """

  dense = adj_idx is not None
  sparse = src_idx is not None or dst_idx is not None

  # Check that either dense or sparse indices are provided.
  if not (dense or sparse):
    raise RuntimeError(
        "either dense indices ('adj_idx') or sparse indices ('dst_idx' "
        "and 'src_idx') must be provided"
    )

  # Check that not both dense and sparse indices are provided.
  if dense and sparse:
    raise RuntimeError(
        'could not determine whether to use dense or sparse '
        'indices, because both were provided'
    )

  # Check that shapes of sparse indices are consistent.
  if sparse and dst_idx is not None and src_idx is not None:
    if dst_idx.shape != src_idx.shape:
      raise ValueError(
          f'the shapes of dst_idx ({dst_idx.shape}) and src_idx '
          f'({src_idx.shape}) do not match'
      )

  # Return the index type.
  if dense:
    return 'dense'
  elif sparse:
    return 'sparse'
  else:  # Protection from potential bugs if other valid values are added.
    assert False, 'IndexType could not be detected!'


def dense_pairwise_indices(
    num: int, mask_self: bool = True
) -> Union[Integer[Array, '... num num-1'], Integer[Array, '... num num']]:
  """Generates a dense index list for all pairs of ``num`` elements.

  Args:
    num: Number of elements.
    mask_self: Whether self-connections (loops) should be masked out.

  Returns:
    A dense index list specifying all pairs.
  """
  if num < 1:
    raise ValueError(f'num must be larger than 0, received {num}')

  adj_idx = jnp.tile(jnp.arange(num), num)
  if mask_self:
    adj_idx = jnp.delete(adj_idx, slice(0, num * num, num + 1))
  return jnp.reshape(adj_idx, (num, -1))


def sparse_pairwise_indices(num: int, mask_self: bool = True) -> Union[
    Tuple[Integer[Array, '... num*(num-1)'], Integer[Array, '... num*(num-1)']],
    Tuple[Integer[Array, '... num**2'], Integer[Array, '... num**2']],
]:
  """Generates a sparse index list for all pairs of ``num`` elements.

  Args:
    num: Number of elements.
    mask_self: Whether self-connections (loops) should be masked out.

  Returns:
    A sparse index list specifying all pairs returned as a tuple ``(dst_idx,
    src_idx)`` with destination and source indices, respectively.
  """
  if num < 1:
    raise ValueError(f'num must be larger than 0, received {num}')

  dst_idx = jnp.repeat(jnp.arange(num), num)
  src_idx = jnp.tile(jnp.arange(num), num)
  if mask_self:
    dst_idx = jnp.delete(dst_idx, slice(0, num * num, num + 1))
    src_idx = jnp.delete(src_idx, slice(0, num * num, num + 1))
  return dst_idx, src_idx


def dense_to_sparse_indices(
    adj_idx: Integer[Array, '... N M'], pad_idx: Optional[int] = None
) -> Tuple[Integer[Array, '... P'], Integer[Array, '... P']]:
  """Converts a dense index list to a sparse index list.

  Note: Not compatible with ``jax.jit`` or ``jax.vmap`` because boolean arrays
  are not concrete.

  Args:
    adj_idx: Adjcency indices.
    pad_idx: All indices >= ``pad_idx`` are assumed to be padding. If this is
      ``None``, all indices are assumed to be non-padding entries.

  Returns:
    The corresponding sparse index list returned as a tuple ``(dst_idx,
    src_idx)`` with destination and source indices, respectively.
  """
  # Shape check.
  if adj_idx.ndim < 2:
    raise ValueError('adj_idx must have at least two dimensions')
  # Handle size 0.
  if adj_idx.size == 0:
    dst_idx = jnp.empty_like(adj_idx, shape=(*adj_idx.shape[:-2], 0))
    src_idx = dst_idx
    return dst_idx, src_idx
  # Determine padding index.
  pad_idx = jnp.max(adj_idx) + 1 if pad_idx is None else pad_idx
  # Remember original batch dimensions (might also be empty).
  batch_dims = adj_idx.shape[:-2]
  # Reshape so looping over batch dimension is possible.
  adj_idx = jnp.reshape(adj_idx, (-1, *adj_idx.shape[-2:]))
  # Fill lists of dst_idx and src_idx.
  dst_idx = []
  src_idx = []
  max_size = 0
  for b in range(adj_idx.shape[0]):  # Loop over batch dimension.
    dst_idx.append([])
    src_idx.append([])
    for i in range(adj_idx.shape[-2]):
      mask = adj_idx[b, i] < pad_idx
      src_idx[b].append(adj_idx[b, i, mask])
      dst_idx[b].append(
          jnp.full((len(src_idx[b][-1]),), i, dtype=adj_idx.dtype)
      )
    dst_idx[b] = jnp.concatenate(dst_idx[b])
    src_idx[b] = jnp.concatenate(src_idx[b])
    max_size = max(max_size, len(dst_idx[b]))
  # Pad all dst_idx and src_idx to max_size and stack them.
  for b in range(len(dst_idx)):
    pad_width = ((0, max_size - len(dst_idx[b])),)
    dst_idx[b] = jnp.pad(dst_idx[b], pad_width, constant_values=pad_idx)
    src_idx[b] = jnp.pad(src_idx[b], pad_width, constant_values=pad_idx)
  dst_idx = jnp.stack(dst_idx)
  src_idx = jnp.stack(src_idx)
  # Reshape to original batch dimension.
  dst_idx = jnp.reshape(dst_idx, (*batch_dims, dst_idx.shape[1]))
  src_idx = jnp.reshape(src_idx, (*batch_dims, src_idx.shape[1]))
  return dst_idx, src_idx


def sparse_to_dense_indices(
    dst_idx: Integer[Array, '... P'],
    src_idx: Integer[Array, '... P'],
    pad_idx: Optional[int] = None,
) -> Integer[Array, '... N M']:
  """Converts a sparse index list to a dense index list.

  Note: Not compatible with ``jax.jit`` or ``jax.vmap`` because boolean arrays
  are not concrete.

  Args:
    dst_idx: Destination indices.
    src_idx: Source indices.
    pad_idx: All indices >= ``pad_idx`` are assumed to be padding. If this is
      ``None``, all indices are assumed to be non-padding entries.

  Returns:
    The corresponding dense index list.
  """
  # Shape check.
  if dst_idx.shape != src_idx.shape:
    raise ValueError(
        f'dst_idx and src_idx have incompatible shapes {dst_idx.shape} and '
        f'{src_idx.shape}'
    )
  # Handle size 0.
  if dst_idx.size == 0:
    return jnp.empty_like(dst_idx, shape=(*dst_idx.shape[:-1], 0, 0))
  # Determine padding index.
  pad_idx = jnp.max(dst_idx) + 1 if pad_idx is None else pad_idx
  # Remember original batch dimensions (might also be empty).
  batch_dims = dst_idx.shape[:-1]
  # Reshape so looping over batch dimension is possible.
  dst_idx = jnp.reshape(dst_idx, (-1, dst_idx.shape[-1]))
  src_idx = jnp.reshape(src_idx, (-1, src_idx.shape[-1]))
  # Determine how many entries are necessary for each index and batch.
  counts = jnp.array(
      [jnp.count_nonzero(dst_idx == i, axis=-1) for i in range(pad_idx)]
  ).T
  max_count = jnp.max(counts)
  # Fill adj_idx.
  adj_idx = jnp.full(
      (dst_idx.shape[0], pad_idx, max_count), pad_idx, dtype=dst_idx.dtype
  )
  for b in range(adj_idx.shape[0]):  # Loop over batch dimension.
    for i in range(adj_idx.shape[1]):
      adj_idx = adj_idx.at[b, i, : counts[b, i]].set(
          src_idx[b, dst_idx[b] == i]
      )
  # Reshape to original batch dimension.
  adj_idx = jnp.reshape(
      adj_idx, (*batch_dims, adj_idx.shape[1], adj_idx.shape[2])
  )
  return adj_idx


def _gather_dense(
    inputs: Shaped[Array, '_*indexable_by_idx'],
    idx: Integer[Array, '... N M'],
) -> Shaped[Array, '_*input_shape_indexed_by_idx']:
  """Gather from inputs with a dense index list.

  Args:
    inputs: Inputs to gather from.
    idx: Dense index list used for gathering.

  Returns:
    An array with the gathered values.
  """
  # Shape checks.
  if idx.ndim < 2:
    raise ValueError('idx must have at least two dimensions')
  if inputs.ndim < idx.ndim - 1:
    raise ValueError(
        f'inputs ({inputs.ndim} dimensions) must have at least '
        f'{idx.ndim-1} dimensions for gathering with idx of shape {idx.shape}'
    )
  if idx.shape[:-2] != inputs.shape[: idx.ndim - 2]:
    raise ValueError(
        f'idx with shape {idx.shape} is incompatible for gathering from inputs '
        f'with shape {inputs.shape}'
    )

  # There are batch dimensions, needs vmapping.
  if idx.ndim > 2:
    # Reshape to have a single batch dimension.
    reshaped_inputs = jnp.reshape(inputs, (-1, *inputs.shape[idx.ndim - 2 :]))
    reshaped_idx = jnp.reshape(idx, (-1, *idx.shape[-2:]))
    # Run vmap over batch dimension.
    gathered_inputs = jax.vmap(lambda x, i: x[i], in_axes=(0, 0), out_axes=0)(
        reshaped_inputs, reshaped_idx
    )
    # Reshape to original batch dimensions.
    gathered_inputs = jnp.reshape(
        gathered_inputs,
        (*idx.shape[: idx.ndim - 2], *gathered_inputs.shape[1:]),
    )
  # There are no batch dimensions, so gathering is trivial.
  else:
    gathered_inputs = inputs[idx]
  return gathered_inputs


def _gather_sparse(
    inputs: Shaped[Array, '_*indexable_by_idx'],
    idx: Integer[Array, '... P'],
) -> Shaped[Array, '_*input_shape_indexed_by_idx']:
  """Gather from inputs with a sparse index list.

  Args:
    inputs: Inputs to gather from.
    idx: Sparse index list used for gathering (only the source indices are
      necessary).

  Returns:
    An array with the gathered values.
  """
  # Shape checks.
  if idx.ndim < 1:
    raise ValueError('idx must have at least one dimension')
  if inputs.ndim < idx.ndim:
    raise ValueError(
        f'inputs ({inputs.ndim} dimensions) must have at least '
        f'{idx.ndim} dimensions for gathering with idx of shape '
        f'{idx.shape}'
    )
  if idx.shape[: idx.ndim - 1] != inputs.shape[: idx.ndim - 1]:
    raise ValueError(
        f'idx with shape {idx.shape} is incompatible for gathering from inputs '
        f'with shape {inputs.shape}'
    )

  # There are batch dimensions, needs vmapping.
  if idx.ndim > 1:
    # Reshape to have a single batch dimension.
    reshaped_inputs = jnp.reshape(inputs, (-1, *inputs.shape[idx.ndim - 1 :]))
    reshaped_idx = jnp.reshape(idx, (-1, idx.shape[-1]))
    # Run vmap over batch dimension.
    gathered_inputs = jax.vmap(lambda x, i: x[i], in_axes=(0, 0), out_axes=0)(
        reshaped_inputs, reshaped_idx
    )
    # Reshape to original batch dimensions.
    gathered_inputs = jnp.reshape(
        gathered_inputs, (*idx.shape[:-1], *gathered_inputs.shape[1:])
    )
  # There are no batch dimensions, gathering is trivial.
  else:
    gathered_inputs = inputs[idx]
  return gathered_inputs


def gather_src(
    inputs: Shaped[Array, '_*indexable_by_adj_idx_or_src_idx'],
    *,
    adj_idx: Optional[Integer[Array, '... N M']] = None,
    src_idx: Optional[Integer[Array, '... P']] = None,
    **_,
) -> Shaped[Array, '_*input_shape_indexed_by_adj_idx_or_src_idx']:
  """Gather from inputs (source) according to dense or sparse index list.

  Args:
    inputs: Inputs to gather from.
    adj_idx: Adjacency indices (if dense index list is used), or ``None`` (if
      sparse index list is used).
    src_idx: Source indices (if sparse index list is used), or ``None`` (if
      dense index list is used).

  Returns:
    An array with the gathered values.

  Raises:
    RuntimeError: If neither dense nor sparse index lists are provided, or if
      both are provided.
  """
  index_type = _determine_index_type(adj_idx=adj_idx, src_idx=src_idx)
  if index_type == 'dense':
    gathered_inputs = _gather_dense(inputs, adj_idx)
  elif index_type == 'sparse':
    gathered_inputs = _gather_sparse(inputs, src_idx)
  else:  # Protection from potential bugs if other valid values are added.
    assert False, f"Missing implementation for index_type '{index_type}'."
    gathered_inputs = inputs  # Silence typechecker.
  return gathered_inputs


def gather_dst(
    inputs: Shaped[Array, '_*indexable_by_adj_idx_or_src_idx'],
    *,
    adj_idx: Optional[Integer[Array, '... N M']] = None,
    dst_idx: Optional[Integer[Array, '... P']] = None,
    **_,
) -> Shaped[Array, '_*input_shape_indexed_by_adj_idx_or_dst_idx']:
  """Gather from inputs (destination) according to dense or sparse index list.

  Args:
    inputs: Inputs to gather from.
    adj_idx: Adjacency indices (if dense index list is used), or ``None`` (if
      sparse index list is used).
    dst_idx: Destination indices (if sparse index list is used), or ``None`` (if
      dense index list is used).

  Returns:
    An array with the gathered values.

  Raises:
    RuntimeError: If neither dense nor sparse index lists are provided, or if
      both are provided.
  """
  index_type = _determine_index_type(adj_idx=adj_idx, dst_idx=dst_idx)
  if index_type == 'dense' and adj_idx is not None:
    outputs = jnp.expand_dims(inputs, axis=adj_idx.ndim - 1)
  elif index_type == 'sparse':
    outputs = _gather_sparse(inputs, dst_idx)
  else:  # Protection from potential bugs if other valid values are added.
    assert False, f"Missing implementation for index_type='{index_type}'."
    outputs = inputs  # Silence typechecker.
  return outputs


_valid_reduction_modes = ('sum', 'min', 'max')
ReductionMode = Literal[_valid_reduction_modes]


def _indexed_reduce(
    reduction_mode: ReductionMode,
    inputs: Num[Array, '_*indexable_by_relevant_idx'],
    keepdims: Optional[bool] = None,
    *,
    adj_idx: Optional[Integer[Array, '... N M']] = None,
    where: Optional[Bool[Array, '... N M']] = None,
    dst_idx: Optional[Integer[Array, '... P']] = None,
    num_segments: Optional[int] = None,
    indices_are_sorted: bool = False,
    **_,
) -> Num[Array, '_*input_shape_indexed_by_relevant_idx']:
  """Performs a reduction on inputs according to sparse or dense index lists.

  Args:
    reduction_mode: ``'sum'``, ``'min'``, or ``'max'``.
    inputs: Inputs to perform the reduction on.
    keepdims: Whether to keep the dimensions of the inputs (so that outputs can
      be broadcasted to the shape of the inputs). In the case of dense index
      lists, one axis will be reduced to size 1, in the case of sparse index
      lists, the shape of inputs and outputs will be the same.
    adj_idx: Adjacency indices.
    where: Mask to specify which values to reduce over, required for dense index
      lists.
    dst_idx: Destination indices.
    num_segments: Number of segments after reduction, required for sparse index
      lists.
    indices_are_sorted: If ``True``, ``dst_idx`` is assumed to be sorted, which
      may increase performance (only used for sparse index lists).

  Returns:
    An array with the results of the reduction operation.

  Raises:
    RuntimeError: If neither dense nor sparse index lists are provided, or if
      both are provided.
    ValueError: If inputs are not floating point or integer dtype.
  """

  index_type = _determine_index_type(adj_idx=adj_idx, dst_idx=dst_idx)

  # Check input dtype.
  if jnp.issubdtype(inputs.dtype, jnp.integer):
    dtype_info = jnp.iinfo
  elif jnp.issubdtype(inputs.dtype, jnp.floating):
    dtype_info = jnp.finfo
  else:
    raise TypeError(
        'inputs must be floating point or integer dtype, received '
        f'{inputs.dtype}'
    )

  # Dense index list.
  if index_type == 'dense' and adj_idx is not None:
    # Check that where is given.
    if where is None:
      raise TypeError(
          "for dense indices, 'where' is a required argument, received None"
      )
    # Shape checks.
    if inputs.ndim < adj_idx.ndim:
      raise ValueError(
          f'inputs ({inputs.ndim} dimensions) must have at least '
          f'as many dimensions as adj_idx ({adj_idx.ndim} dimensions)'
      )
    if adj_idx.shape != inputs.shape[: adj_idx.ndim]:
      raise ValueError('adj_idx and inputs have incompatible shapes.')

    # Reduction.
    where = jnp.reshape(
        where, (*adj_idx.shape, *(1,) * (inputs.ndim - adj_idx.ndim))
    )
    axis = adj_idx.ndim - 1
    if reduction_mode == 'sum':
      outputs = jnp.sum(inputs, axis=axis, where=where, keepdims=keepdims)
    elif reduction_mode == 'min':
      outputs = jnp.min(
          inputs,
          axis=axis,
          where=where,
          keepdims=keepdims,
          initial=dtype_info(inputs.dtype).max,
      )
    elif reduction_mode == 'max':
      outputs = jnp.max(
          inputs,
          axis=axis,
          where=where,
          keepdims=keepdims,
          initial=dtype_info(inputs.dtype).min,
      )
    else:  # Protection from potential bugs if other valid values are added.
      assert (
          False
      ), f"Missing implementation for reduction_mode '{reduction_mode}'."
      outputs = inputs  # Silence typechecker.

  # Sparse neighborlist.
  elif index_type == 'sparse' and dst_idx is not None:
    # Check that num_segments is given.
    if num_segments is None:
      raise TypeError(
          "for sparse indices, 'num_segments' is a required "
          'argument, received None'
      )
    # Shape checks.
    if inputs.ndim < dst_idx.ndim:
      raise ValueError(
          f'inputs ({inputs.ndim} dimensions) must have at least '
          f'as many dimensions as dst_idx ({dst_idx.ndim} dimensions)'
      )
    if dst_idx.shape != inputs.shape[: dst_idx.ndim]:
      raise ValueError(
          f'dst_idx and inputs have incompatible shapes {dst_idx.shape} and '
          f'{inputs.shape}'
      )

    # Choose correct segmented reduction function.
    if reduction_mode == 'sum':
      segment_reduce = jax.ops.segment_sum
    elif reduction_mode == 'min':
      segment_reduce = jax.ops.segment_min
    elif reduction_mode == 'max':
      segment_reduce = jax.ops.segment_max
    else:  # Protection from potential bugs if other valid values are added.
      assert (
          False
      ), f"Missing implementation for reduction_mode '{reduction_mode}'."
      segment_reduce = lambda x: x  # Silence typechecker.

    # There are batch dimensions, needs vmapping.
    if dst_idx.ndim > 1:
      # Reshape to have a single batch dimension.
      reshaped_inputs = jnp.reshape(
          inputs, (-1, *inputs.shape[dst_idx.ndim - 1 :])
      )
      reshaped_dst_idx = jnp.reshape(dst_idx, (-1, dst_idx.shape[-1]))
      # Run vmap over batch dimension.
      outputs = jax.vmap(
          functools.partial(
              segment_reduce,
              num_segments=num_segments,
              indices_are_sorted=indices_are_sorted,
          ),
          in_axes=(0, 0),
          out_axes=0,
      )(reshaped_inputs, reshaped_dst_idx)
      # Reshape to original batch dimensions.
      outputs = jnp.reshape(outputs, (*dst_idx.shape[:-1], *outputs.shape[1:]))
    # There are no batch dimensions, trivial reduction.
    else:
      outputs = segment_reduce(
          inputs,
          dst_idx,
          num_segments=num_segments,
          indices_are_sorted=indices_are_sorted,
      )
    if keepdims:
      outputs = gather_src(outputs, src_idx=dst_idx)

  # Protection from potential bugs if other valid values are added.
  else:
    assert False, f"Missing implementation for index_type '{index_type}'."
    outputs = inputs  # Silence typechecker.

  return outputs


def indexed_sum(
    inputs: Num[Array, '_*indexable_by_relevant_idx'],
    keepdims: Optional[bool] = None,
    *,
    adj_idx: Optional[Integer[Array, '... N M']] = None,
    where: Optional[Bool[Array, '... N M']] = None,
    dst_idx: Optional[Integer[Array, '... P']] = None,
    num_segments: Optional[int] = None,
    indices_are_sorted: bool = False,
    **_,
) -> Num[Array, '_*input_shape_indexed_by_relevant_idx']:
  """Performs a sum over inputs according to sparse or dense index lists.

  Args:
    inputs: Inputs to perform the sum over.
    keepdims: Whether to keep the dimensions of the inputs (so that outputs can
      be broadcasted to the shape of the inputs). In the case of dense index
      lists, one axis will be reduced to size 1, in the case of sparse index
      lists, the shape of inputs and outputs will be the same.
    adj_idx: Adjacency indices.
    where: Mask to specify which values to sum over, required for dense index
      lists.
    dst_idx: Destination indices.
    num_segments: Number of segments after summation, required for sparse index
      lists.
    indices_are_sorted: If ``True``, ``dst_idx`` is assumed to be sorted, which
      may increase performance (only used for sparse index lists).

  Returns:
    An array with the results of the summation.

  Raises:
    RuntimeError: If neither dense nor sparse index lists are provided, or if
      both are provided.
    ValueError: If inputs are not floating point or integer dtype.
  """
  return _indexed_reduce(
      'sum',
      inputs=inputs,
      keepdims=keepdims,
      adj_idx=adj_idx,
      where=where,
      dst_idx=dst_idx,
      num_segments=num_segments,
      indices_are_sorted=indices_are_sorted,
  )


def indexed_min(
    inputs: Num[Array, '_*indexable_by_relevant_idx'],
    keepdims: Optional[bool] = None,
    *,
    adj_idx: Optional[Integer[Array, '... N M']] = None,
    where: Optional[Bool[Array, '... N M']] = None,
    dst_idx: Optional[Integer[Array, '... P']] = None,
    num_segments: Optional[int] = None,
    indices_are_sorted: bool = False,
    **_,
) -> Num[Array, '_*input_shape_indexed_by_relevant_idx']:
  """Determines the minimum of inputs according to sparse or dense index lists.

  Args:
    inputs: Inputs for which to determine the minimum.
    keepdims: Whether to keep the dimensions of the inputs (so that outputs can
      be broadcasted to the shape of the inputs). In the case of dense index
      lists, one axis will be reduced to size 1, in the case of sparse index
      lists, the shape of inputs and outputs will be the same.
    adj_idx: Adjacency indices.
    where: Mask to specify which values to take the minimum from, required for
      dense index lists.
    dst_idx: Destination indices.
    num_segments: Number of segments after taking the minimum, required for
      sparse index lists.
    indices_are_sorted: If ``True``, ``dst_idx`` is assumed to be sorted, which
      may increase performance (only used for sparse index lists).

  Returns:
    An array with the minimum values.

  Raises:
    RuntimeError: If neither dense nor sparse index lists are provided, or if
      both are provided.
    ValueError: If inputs are not floating point or integer dtype.
  """
  return _indexed_reduce(
      'min',
      inputs=inputs,
      keepdims=keepdims,
      adj_idx=adj_idx,
      where=where,
      dst_idx=dst_idx,
      num_segments=num_segments,
      indices_are_sorted=indices_are_sorted,
  )


def indexed_max(
    inputs: Num[Array, '_*indexable_by_relevant_idx'],
    keepdims: Optional[bool] = None,
    *,
    adj_idx: Optional[Integer[Array, '... N M']] = None,
    where: Optional[Bool[Array, '... N M']] = None,
    dst_idx: Optional[Integer[Array, '... P']] = None,
    num_segments: Optional[int] = None,
    indices_are_sorted: bool = False,
    **_,
) -> Num[Array, '_*input_shape_indexed_by_relevant_idx']:
  """Determines the maximum of inputs according to sparse or dense index lists.

  Args:
    inputs: Inputs for which to determine the maximum.
    keepdims: Whether to keep the dimensions of the inputs (so that outputs can
      be broadcasted to the shape of the inputs). In the case of dense index
      lists, one axis will be reduced to size 1, in the case of sparse index
      lists, the shape of inputs and outputs will be the same.
    adj_idx: Adjacency indices.
    where: Mask to specify which values to take the maximum from, required for
      dense index lists.
    dst_idx: Destination indices.
    num_segments: Number of segments after taking the maximum, required for
      sparse index lists.
    indices_are_sorted: If ``True``, ``dst_idx`` is assumed to be sorted, which
      may increase performance (only used for sparse index lists).

  Returns:
    An array with the maximum values.

  Raises:
    RuntimeError: If neither dense nor sparse index lists are provided, or if
      both are provided.
    ValueError: If inputs are not floating point or integer dtype.
  """
  return _indexed_reduce(
      'max',
      inputs=inputs,
      keepdims=keepdims,
      adj_idx=adj_idx,
      where=where,
      dst_idx=dst_idx,
      num_segments=num_segments,
      indices_are_sorted=indices_are_sorted,
  )


def indexed_softmax(
    inputs: Union[Float[Array, '... N M'], Float[Array, '... P']],
    multiplicative_mask: Optional[
        Union[Float[Array, '... N M'], Float[Array, '... P']]
    ] = None,
    *,
    adj_idx: Optional[Integer[Array, '... N M']] = None,
    where: Optional[Bool[Array, '... N M']] = None,
    dst_idx: Optional[Integer[Array, '... P']] = None,
    num_segments: Optional[int] = None,
    indices_are_sorted: bool = False,
    **_,
) -> Union[Float[Array, '... N M'], Float[Array, '... P']]:
  """Determines the softmax of inputs according to sparse or dense index lists.

  Args:
    inputs: Inputs for which to compute the softmax.
    multiplicative_mask: Optional mask to multiply with the raw exponentials
      (before normalization). This can be used for example for smooth cutoffs.
    adj_idx: Adjacency indices.
    where: Mask to specify which values to take the maximum from, required for
      dense index lists.
    dst_idx: Destination indices.
    num_segments: Number of segments after taking the maximum, required for
      sparse index lists.
    indices_are_sorted: If ``True``, ``dst_idx`` is assumed to be sorted, which
      may increase performance (only used for sparse index lists).

  Returns:
    An array with the softmax values.

  Raises:
    RuntimeError: If neither dense nor sparse index lists are provided, or if
      both are provided.
    ValueError: If the shape of multiplicative mask does not match the shape of
      inputs, or inputs are not floating point dtype.
  """
  # Shape checks.
  if multiplicative_mask is not None:
    if multiplicative_mask.shape != inputs.shape:
      raise ValueError(
          'multiplicative_mask and inputs have incompatible '
          f'shapes {multiplicative_mask.shape} and {inputs.shape}'
      )

  # Check input dtype.
  if not jnp.issubdtype(inputs.dtype, jnp.floating):
    raise ValueError(
        f'inputs must be floating point dtype, received {inputs.dtype}'
    )

  # The max input is subtracted from all inputs before applying exp for
  # increased numerical stability (prevents overflow).
  maximum = indexed_max(
      inputs=inputs,
      keepdims=True,
      adj_idx=adj_idx,
      where=where,
      dst_idx=dst_idx,
      num_segments=num_segments,
      indices_are_sorted=indices_are_sorted,
  )
  numerator = jnp.exp(inputs - lax.stop_gradient(maximum))
  if multiplicative_mask is not None:
    numerator *= multiplicative_mask
  denominator = indexed_sum(
      inputs=numerator,
      keepdims=True,
      adj_idx=adj_idx,
      where=where,
      dst_idx=dst_idx,
      num_segments=num_segments,
      indices_are_sorted=indices_are_sorted,
  )
  return numerator / denominator
