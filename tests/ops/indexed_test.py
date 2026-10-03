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

from typing import Optional
import e3x
from ..testing import subtests
import jax
import jax.numpy as jnp
import numpy as np
import jaxtyping
import pytest

Array = jaxtyping.Array
Bool = jaxtyping.Bool
Float = jaxtyping.Float
Integer = jaxtyping.Integer


@subtests({
    'adj_idx is given': dict(
        adj_idx=jnp.asarray([[0]]), dst_idx=None, src_idx=None, expected='dense'
    ),
    'dst_idx is given': dict(
        adj_idx=None, dst_idx=jnp.asarray([0]), src_idx=None, expected='sparse'
    ),
    'src_idx is given': dict(
        adj_idx=None, dst_idx=None, src_idx=jnp.asarray([0]), expected='sparse'
    ),
    'dst_idx and src_idx are given': dict(
        adj_idx=None,
        dst_idx=jnp.asarray([0]),
        src_idx=jnp.asarray([0]),
        expected='sparse',
    ),
})
def test_determine_index_type(
    adj_idx: Optional[Integer[Array, '...']],
    dst_idx: Optional[Integer[Array, '...']],
    src_idx: Optional[Integer[Array, '...']],
    expected: e3x.ops.indexed.IndexType,
) -> None:
  assert (
      e3x.ops.indexed._determine_index_type(  # pylint: disable=protected-access
          adj_idx=adj_idx, dst_idx=dst_idx, src_idx=src_idx
      )
      == expected
  )


@subtests({
    'neither dense nor sparse is given': dict(
        adj_idx=None,
        dst_idx=None,
        src_idx=None,
        message='either dense .* or sparse .* must be provided',
    ),
    'both dense and sparse are given': dict(
        adj_idx=jnp.asarray([[0]]),
        dst_idx=jnp.asarray([0]),
        src_idx=jnp.asarray([0]),
        message='could not determine .* both were provided',
    ),
})
def test_determine_index_type_raises_if_index_type_undeterminable(
    adj_idx: Optional[Integer[Array, '...']],
    dst_idx: Optional[Integer[Array, '...']],
    src_idx: Optional[Integer[Array, '...']],
    message: str,
) -> None:
  with pytest.raises(RuntimeError, match=message):
    e3x.ops.indexed._determine_index_type(  # pylint: disable=protected-access
        adj_idx=adj_idx, dst_idx=dst_idx, src_idx=src_idx
    )


def test_determine_index_type_raises_if_shapes_do_not_match() -> None:
  with pytest.raises(ValueError, match='shapes .* do not match'):
    e3x.ops.indexed._determine_index_type(  # pylint: disable=protected-access
        adj_idx=None, dst_idx=jnp.asarray([0]), src_idx=jnp.asarray([0, 0])
    )


@subtests({
    'num=1, mask_self=True': dict(
        num=1, mask_self=True, expected=jnp.asarray([[]])
    ),
    'num=1, mask_self=False': dict(
        num=1, mask_self=False, expected=jnp.asarray([[0]])
    ),
    'num=2, mask_self=True': dict(
        num=2, mask_self=True, expected=jnp.asarray([[1], [0]])
    ),
    'num=2, mask_self=False': dict(
        num=2, mask_self=False, expected=jnp.asarray(2 * [[0, 1]])
    ),
    'num=3, mask_self=True': dict(
        num=3, mask_self=True, expected=jnp.asarray([[1, 2], [0, 2], [0, 1]])
    ),
    'num=3, mask_self=False': dict(
        num=3, mask_self=False, expected=jnp.asarray(3 * [[0, 1, 2]])
    ),
})
def test_dense_pairwise_indices(
    num: int, mask_self: bool, expected: Integer[Array, '...']
) -> None:
  assert jnp.array_equal(
      e3x.ops.dense_pairwise_indices(num=num, mask_self=mask_self), expected
  )


@pytest.mark.parametrize('num', [-1, 0])
@pytest.mark.parametrize('mask_self', [True, False])
def test_dense_pairwise_indices_raises_if_num_negative_or_zero(
    num: int, mask_self: bool
) -> None:
  with pytest.raises(
      ValueError, match=f'must be larger than 0, received {num}'
  ):
    e3x.ops.dense_pairwise_indices(num=num, mask_self=mask_self)


@subtests({
    'num=1, mask_self=True': dict(
        num=1, mask_self=True, expected=(jnp.asarray([]), jnp.asarray([]))
    ),
    'num=1, mask_self=False': dict(
        num=1, mask_self=False, expected=(jnp.asarray([0]), jnp.asarray([0]))
    ),
    'num=2, mask_self=True': dict(
        num=2,
        mask_self=True,
        expected=(jnp.asarray([0, 1]), jnp.asarray([1, 0])),
    ),
    'num=2, mask_self=False': dict(
        num=2,
        mask_self=False,
        expected=(jnp.asarray([0, 0, 1, 1]), jnp.asarray([0, 1, 0, 1])),
    ),
    'num=3, mask_self=True': dict(
        num=3,
        mask_self=True,
        expected=(
            jnp.asarray([0, 0, 1, 1, 2, 2]),
            jnp.asarray([1, 2, 0, 2, 0, 1]),
        ),
    ),
    'num=3, mask_self=False': dict(
        num=3,
        mask_self=False,
        expected=(
            jnp.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2]),
            jnp.asarray([0, 1, 2, 0, 1, 2, 0, 1, 2]),
        ),
    ),
})
def test_sparse_pairwise_indices(
    num: int, mask_self: bool, expected: Integer[Array, '...']
) -> None:
  assert jnp.array_equal(  # pytype: disable=wrong-arg-types  # jnp-type
      e3x.ops.sparse_pairwise_indices(num=num, mask_self=mask_self), expected  # pyrefly: ignore[bad-argument-type]
  )


@pytest.mark.parametrize('num', [-1, 0])
@pytest.mark.parametrize('mask_self', [True, False])
def test_sparse_pairwise_indices_raises_if_num_negative_or_zero(
    num: int, mask_self: bool
) -> None:
  with pytest.raises(
      ValueError, match=f'must be larger than 0, received {num}'
  ):
    e3x.ops.sparse_pairwise_indices(num=num, mask_self=mask_self)


@subtests({
    'empty input': dict(
        adj_idx=jnp.empty(shape=(0, 0), dtype=jnp.int32),
        pad_idx=None,
        expected_dst_idx=jnp.empty(shape=(0,), dtype=jnp.int32),
        expected_src_idx=jnp.empty(shape=(0,), dtype=jnp.int32),
    ),
    'empty input with batch dimensions': dict(
        adj_idx=jnp.empty(shape=(2, 3, 0, 0), dtype=jnp.int32),
        pad_idx=None,
        expected_dst_idx=jnp.empty(shape=(2, 3, 0), dtype=jnp.int32),
        expected_src_idx=jnp.empty(shape=(2, 3, 0), dtype=jnp.int32),
    ),
    'no padding': dict(
        adj_idx=jnp.asarray([[1, 2], [0, 2], [0, 1]]),
        pad_idx=None,
        expected_dst_idx=jnp.asarray([0, 0, 1, 1, 2, 2]),
        expected_src_idx=jnp.asarray([1, 2, 0, 2, 0, 1]),
    ),
    'with padding': dict(
        adj_idx=jnp.asarray([[0, 1], [0, 2]]),
        pad_idx=2,
        expected_dst_idx=jnp.asarray([0, 0, 1]),
        expected_src_idx=jnp.asarray([0, 1, 0]),
    ),
    'with padding and batch dimensions': dict(
        adj_idx=jnp.asarray([
            [[0, 1, 2], [1, 2, 3], [2, 3, 3]],
            [[0, 3, 3], [0, 1, 3], [0, 1, 3]],
        ]),
        pad_idx=3,
        expected_dst_idx=jnp.asarray([[0, 0, 0, 1, 1, 2], [0, 1, 1, 2, 2, 3]]),
        expected_src_idx=jnp.asarray([[0, 1, 2, 1, 2, 2], [0, 0, 1, 0, 1, 3]]),
    ),
})
def test_dense_to_sparse_indices(
    adj_idx: Integer[Array, '...'],
    pad_idx: Optional[int],
    expected_dst_idx: Integer[Array, '...'],
    expected_src_idx: Integer[Array, '...'],
) -> None:
  dst_idx, src_idx = e3x.ops.dense_to_sparse_indices(
      adj_idx=adj_idx, pad_idx=pad_idx
  )
  assert jnp.array_equal(dst_idx, expected_dst_idx)
  assert jnp.array_equal(src_idx, expected_src_idx)


@pytest.mark.parametrize('pad_idx', [None, 0])
def test_dense_to_sparse_raises_with_incorrect_input_shape(
    pad_idx: Optional[int],
) -> None:
  with pytest.raises(ValueError, match='at least two dimensions'):
    e3x.ops.dense_to_sparse_indices(adj_idx=jnp.asarray([]), pad_idx=pad_idx)


@subtests({
    'empty input': dict(
        dst_idx=jnp.empty(shape=(0,), dtype=jnp.int32),
        src_idx=jnp.empty(shape=(0,), dtype=jnp.int32),
        pad_idx=None,
        expected_adj_idx=jnp.empty(shape=(0, 0), dtype=jnp.int32),
    ),
    'empty input with batch dimensions': dict(
        dst_idx=jnp.empty(shape=(2, 3, 0), dtype=jnp.int32),
        src_idx=jnp.empty(shape=(2, 3, 0), dtype=jnp.int32),
        pad_idx=None,
        expected_adj_idx=jnp.empty(shape=(2, 3, 0, 0), dtype=jnp.int32),
    ),
    'no padding': dict(
        dst_idx=jnp.asarray([0, 0, 1]),
        src_idx=jnp.asarray([0, 1, 0]),
        pad_idx=None,
        expected_adj_idx=jnp.asarray([[0, 1], [0, 2]]),
    ),
    'with padding': dict(
        dst_idx=jnp.asarray([0, 0, 1, 2]),
        src_idx=jnp.asarray([0, 1, 0, 2]),
        pad_idx=2,
        expected_adj_idx=jnp.asarray([[0, 1], [0, 2]]),
    ),
    'with padding and batch dimensions': dict(
        dst_idx=jnp.asarray([[0, 0, 0, 1, 1, 2], [0, 1, 1, 2, 2, 3]]),
        src_idx=jnp.asarray([[0, 1, 2, 1, 2, 2], [0, 0, 1, 0, 1, 3]]),
        pad_idx=3,
        expected_adj_idx=jnp.asarray([
            [[0, 1, 2], [1, 2, 3], [2, 3, 3]],
            [[0, 3, 3], [0, 1, 3], [0, 1, 3]],
        ]),
    ),
})
def test_sparse_to_dense_indices(
    dst_idx: Integer[Array, '...'],
    src_idx: Integer[Array, '...'],
    pad_idx: Optional[int],
    expected_adj_idx: Integer[Array, '...'],
) -> None:
  adj_idx = e3x.ops.sparse_to_dense_indices(
      dst_idx=dst_idx, src_idx=src_idx, pad_idx=pad_idx
  )
  assert jnp.array_equal(adj_idx, expected_adj_idx)


@pytest.mark.parametrize('pad_idx', [None, 0])
def test_sparse_to_dense_indices_with_incompatible_input_shapes(
    pad_idx: Optional[int],
) -> None:
  with pytest.raises(ValueError, match='incompatible shapes'):
    e3x.ops.sparse_to_dense_indices(
        dst_idx=jnp.asarray([0, 0]), src_idx=jnp.asarray([1]), pad_idx=pad_idx
    )


@subtests({
    'one-dimensional': dict(
        inputs=jnp.asarray([10, 11, 12]),
        idx=jnp.asarray([[1, 2], [0, 3], [1, 3]]),
        expected=jnp.asarray([[11, 12], [10, 12], [11, 12]]),
    ),
    'multi-dimensional': dict(
        inputs=jnp.asarray([[10, 20], [11, 21], [12, 22]]),
        idx=jnp.asarray([[1, 2], [0, 3], [1, 3]]),
        expected=jnp.asarray([
            [[11, 21], [12, 22]],
            [[10, 20], [12, 22]],
            [[11, 21], [12, 22]],
        ]),
    ),
    'with batch dimensions': dict(
        inputs=jnp.asarray([[10, 11, 12], [20, 21, 22]]),
        idx=jnp.asarray([
            [[1, 2], [0, 3], [1, 3]],
            [[0, 2], [1, 2], [3, 3]],
        ]),
        expected=jnp.asarray(
            [[[11, 12], [10, 12], [11, 12]], [[20, 22], [21, 22], [22, 22]]]
        ),
    ),
})
def test_gather_dense(
    inputs: Array, idx: Integer[Array, '...'], expected: Array
) -> None:
  assert jnp.array_equal(e3x.ops.indexed._gather_dense(inputs, idx), expected)  # pylint: disable=protected-access


@subtests({
    'idx not two-dimensional': dict(
        inputs=jnp.zeros(shape=(1,), dtype=jnp.float32),
        idx=jnp.zeros(shape=(1,), dtype=jnp.int32),
        message='at least two dimensions',
    ),
    'inputs have too few dimensions': dict(
        inputs=jnp.zeros(shape=(1,), dtype=jnp.float32),
        idx=jnp.zeros(shape=(3, 2, 1), dtype=jnp.int32),
        message='must have at least 2 dimensions for gathering with idx',
    ),
    'shape mismatch': dict(
        inputs=jnp.zeros(shape=(2, 3, 1), dtype=jnp.float32),
        idx=jnp.zeros(shape=(3, 2, 1), dtype=jnp.int32),
        message='incompatible',
    ),
})
def test_gather_dense_raises_with_invalid_inputs(
    inputs: Array, idx: Integer[Array, '...'], message: str
) -> None:
  with pytest.raises(ValueError, match=message):
    e3x.ops.indexed._gather_dense(inputs, idx)  # pylint: disable=protected-access


@subtests({
    'one-dimensional': dict(
        inputs=jnp.asarray([10, 11, 12]),
        idx=jnp.asarray([0, 0, 1, 1, 2]),
        expected=jnp.asarray([10, 10, 11, 11, 12]),
    ),
    'multi-dimensional': dict(
        inputs=jnp.asarray([[10, 20], [11, 21], [12, 22]]),
        idx=jnp.asarray([0, 0, 1, 1, 2]),
        expected=jnp.asarray(
            [[10, 20], [10, 20], [11, 21], [11, 21], [12, 22]]
        ),
    ),
    'with batch dimensions': dict(
        inputs=jnp.asarray([[10, 11, 12], [20, 21, 22]]),
        idx=jnp.asarray([[0, 0, 1, 2], [1, 3, 3, 3]]),
        expected=jnp.asarray([[10, 10, 11, 12], [21, 22, 22, 22]]),
    ),
})
def test_gather_sparse(
    inputs: Array, idx: Integer[Array, '...'], expected: Array
) -> None:
  assert jnp.array_equal(e3x.ops.indexed._gather_sparse(inputs, idx), expected)  # pylint: disable=protected-access


@subtests({
    'idx not one-dimensional': dict(
        inputs=jnp.zeros(shape=(1,), dtype=jnp.float32),
        idx=jnp.zeros(shape=(), dtype=jnp.int32),
        message='at least one dimension',
    ),
    'inputs have too few dimensions': dict(
        inputs=jnp.zeros(shape=(1,), dtype=jnp.float32),
        idx=jnp.zeros(shape=(2, 1), dtype=jnp.int32),
        message='must have at least 2 dimensions for gathering with idx',
    ),
    'shape mismatch': dict(
        inputs=jnp.zeros(shape=(2, 3, 1), dtype=jnp.float32),
        idx=jnp.zeros(shape=(3, 2, 1), dtype=jnp.int32),
        message='incompatible',
    ),
})
def test_gather_sparse_raises_with_invalid_inputs(
    inputs: Array, idx: Integer[Array, '...'], message: str
) -> None:
  with pytest.raises(ValueError, match=message):
    e3x.ops.indexed._gather_sparse(inputs, idx)  # pylint: disable=protected-access


@subtests({
    'dense index list': dict(
        inputs=jnp.asarray([10, 11]),
        adj_idx=jnp.asarray([[1], [0]]),
        src_idx=None,
        expected=jnp.asarray([[11], [10]]),
    ),
    'dense index list batched': dict(
        inputs=jnp.asarray([[10, 11], [20, 21]]),
        adj_idx=jnp.asarray([[[0, 1], [0, 1]], [[0, 0], [0, 1]]]),
        src_idx=None,
        expected=jnp.asarray([[[10, 11], [10, 11]], [[20, 20], [20, 21]]]),
    ),
    'sparse index list': dict(
        inputs=jnp.asarray([10, 11]),
        adj_idx=None,
        src_idx=jnp.asarray([1, 0]),
        expected=jnp.asarray([11, 10]),
    ),
    'sparse index list batched': dict(
        inputs=jnp.asarray([[10, 11], [20, 21]]),
        adj_idx=None,
        src_idx=jnp.asarray([[0, 0, 1], [0, 1, 1]]),
        expected=jnp.asarray([[10, 10, 11], [20, 21, 21]]),
    ),
})
def test_gather_src(
    inputs: Array,
    adj_idx: Optional[Integer[Array, '...']],
    src_idx: Optional[Integer[Array, '...']],
    expected: Array,
) -> None:
  assert jnp.array_equal(
      e3x.ops.gather_src(
          inputs, adj_idx=adj_idx, src_idx=src_idx, dst_idx=None
      ),
      expected,
  )


@subtests({
    'dense index list': dict(
        inputs=jnp.asarray([10, 11]),
        adj_idx=jnp.asarray([[1], [0]]),
        dst_idx=None,
        expected=jnp.asarray([[10], [11]]),
    ),
    'dense index list batched': dict(
        inputs=jnp.asarray([[10, 11], [20, 21]]),
        adj_idx=jnp.asarray([[[0, 1], [0, 1]], [[0, 0], [0, 1]]]),
        dst_idx=None,
        expected=jnp.asarray([[[10], [11]], [[20], [21]]]),
    ),
    'sparse index list': dict(
        inputs=jnp.asarray([10, 11]),
        adj_idx=None,
        dst_idx=jnp.asarray([1, 0]),
        expected=jnp.asarray([11, 10]),
    ),
    'sparse index list batched': dict(
        inputs=jnp.asarray([[10, 11], [20, 21]]),
        adj_idx=None,
        dst_idx=jnp.asarray([[0, 0, 1], [0, 1, 1]]),
        expected=jnp.asarray([[10, 10, 11], [20, 21, 21]]),
    ),
})
def test_gather_dst(
    inputs: Array,
    adj_idx: Optional[Integer[Array, '...']],
    dst_idx: Optional[Integer[Array, '...']],
    expected: Array,
) -> None:
  assert jnp.array_equal(
      e3x.ops.gather_dst(
          inputs, adj_idx=adj_idx, dst_idx=dst_idx, src_idx=None
      ),
      expected,
  )


@pytest.mark.parametrize('reduction_mode', ['sum', 'min', 'max'])
@subtests({
    'neither dense nor sparse is given': dict(
        adj_idx=None,
        dst_idx=None,
        message='either dense .* or sparse .* must be provided',
    ),
    'both dense and sparse are given': dict(
        adj_idx=jnp.asarray([[0]]),
        dst_idx=jnp.asarray([0]),
        message='could not determine .* both were provided',
    ),
})
def test_indexed_reduce_raises_if_index_type_undeterminable(
    reduction_mode: e3x.ops.indexed.ReductionMode,
    adj_idx: Optional[Integer[Array, '...']],
    dst_idx: Optional[Integer[Array, '...']],
    message: str,
) -> None:
  with pytest.raises(RuntimeError, match=message):
    e3x.ops.indexed._indexed_reduce(  # pylint: disable=protected-access
        reduction_mode,
        inputs=jnp.asarray([10]),
        adj_idx=adj_idx,
        dst_idx=dst_idx,
    )


@pytest.mark.parametrize('reduction_mode', ['sum', 'min', 'max'])
@subtests({
    'where is missing': dict(
        adj_idx=jnp.asarray([[0]]), dst_idx=None, message='where'
    ),
    'num_segments is missing': dict(
        adj_idx=None, dst_idx=jnp.asarray([0]), message='num_segments'
    ),
})
def test_indexed_reduce_raises_if_auxiliary_inputs_are_missing(
    reduction_mode: e3x.ops.indexed.ReductionMode,
    adj_idx: Optional[Integer[Array, '...']],
    dst_idx: Optional[Integer[Array, '...']],
    message: str,
) -> None:
  with pytest.raises(TypeError, match=message):
    e3x.ops.indexed._indexed_reduce(  # pylint: disable=protected-access
        reduction_mode,
        inputs=jnp.asarray([10]),
        adj_idx=adj_idx,
        dst_idx=dst_idx,
    )


@pytest.mark.parametrize('reduction_mode', ['sum', 'min', 'max'])
@subtests({
    'dense indices, inputs too few dimensions': dict(
        inputs=jnp.zeros(shape=(3,), dtype=jnp.float32),
        adj_idx=jnp.zeros(shape=(3, 2), dtype=jnp.int32),
        where=jnp.full(shape=(3, 2), fill_value=True, dtype=bool),
        dst_idx=None,
        num_segments=None,
        message='must have at least as many dimensions as adj_idx',
    ),
    'dense indices, shapes do not match': dict(
        inputs=jnp.zeros(shape=(2, 3), dtype=jnp.float32),
        adj_idx=jnp.zeros(shape=(3, 2), dtype=jnp.int32),
        where=jnp.full(shape=(3, 2), fill_value=True, dtype=bool),
        dst_idx=None,
        num_segments=None,
        message='incompatible shapes',
    ),
    'sparse indices, inputs too few dimensions': dict(
        inputs=jnp.zeros(shape=(3,), dtype=jnp.float32),
        adj_idx=None,
        where=None,
        dst_idx=jnp.zeros(shape=(3, 2), dtype=jnp.int32),
        num_segments=1,
        message='must have at least as many dimensions as dst_idx',
    ),
    'sparse indices, shapes do not match': dict(
        inputs=jnp.zeros(shape=(2, 3), dtype=jnp.float32),
        adj_idx=None,
        where=None,
        dst_idx=jnp.zeros(shape=(3, 2), dtype=jnp.int32),
        num_segments=0,
        message='incompatible shapes',
    ),
})
def test_indexed_reduce_raises_if_shapes_do_not_match(
    reduction_mode: e3x.ops.indexed.ReductionMode,
    inputs: Array,
    adj_idx: Optional[Integer[Array, '...']],
    where: Optional[Bool[Array, '...']],
    dst_idx: Optional[Integer[Array, '...']],
    num_segments: Optional[int],
    message: str,
) -> None:
  with pytest.raises(ValueError, match=message):
    e3x.ops.indexed._indexed_reduce(  # pylint: disable=protected-access
        reduction_mode,
        inputs=inputs,
        adj_idx=adj_idx,
        where=where,
        dst_idx=dst_idx,
        num_segments=num_segments,
    )


@pytest.mark.parametrize('reduction_mode', ['sum', 'min', 'max'])
@pytest.mark.parametrize(
    'inputs',
    [
        jnp.zeros(shape=(1,), dtype=bool),
        jnp.zeros(shape=(1,), dtype=jnp.complex64),
    ],
)
def test_indexed_reduce_raises_if_dtype_is_unsupported(
    reduction_mode: e3x.ops.indexed.ReductionMode, inputs: Array
) -> None:
  with pytest.raises(
      TypeError, match='must be floating point or integer dtype'
  ):
    e3x.ops.indexed._indexed_reduce(  # pylint: disable=protected-access
        reduction_mode,
        inputs=inputs,
        adj_idx=jnp.zeros(shape=(1,), dtype=jnp.int32),
    )


@subtests({
    'dense index list, keepdims=False': dict(
        inputs=jnp.asarray([10, 11, 12]),
        keepdims=False,
        adj_idx=jnp.asarray([[0, 1], [1, 2], [0, 3]]),
        where=jnp.asarray([[True, True], [True, True], [True, False]]),
        dst_idx=None,
        src_idx=None,
        num_segments=None,
        indices_are_sorted=False,
        expected=jnp.asarray([21, 23, 10]),
    ),
    'dense index list, keepdims=True': dict(
        inputs=jnp.asarray([10.0, 11.0, 12.0]),
        keepdims=True,
        adj_idx=jnp.asarray([[0, 1], [1, 2], [0, 3]]),
        where=jnp.asarray([[True, True], [True, True], [True, False]]),
        dst_idx=None,
        src_idx=None,
        num_segments=None,
        indices_are_sorted=False,
        expected=jnp.asarray([[21.0], [23.0], [10.0]]),
    ),
    'dense index list batched': dict(
        inputs=jnp.asarray([[10, 11, 12], [20, 21, 22]]),
        keepdims=False,
        adj_idx=jnp.asarray(
            [[[0, 1], [1, 2], [0, 3]], [[1, 2], [1, 3], [0, 2]]]
        ),
        where=jnp.asarray([
            [[True, True], [True, True], [True, False]],
            [[True, True], [True, False], [True, True]],
        ]),
        dst_idx=None,
        src_idx=None,
        num_segments=None,
        indices_are_sorted=False,
        expected=jnp.asarray([[21, 23, 10], [43, 21, 42]]),
    ),
    'sparse index list, keepdims=False': dict(
        inputs=jnp.asarray([10, 11, 12]),
        keepdims=False,
        adj_idx=None,
        where=None,
        dst_idx=jnp.asarray([0, 0, 1, 1, 2, 3]),
        src_idx=jnp.asarray([0, 1, 1, 2, 0, 3]),
        num_segments=3,
        indices_are_sorted=True,
        expected=jnp.asarray([21, 23, 10]),
    ),
    'sparse index list, keepdims=True': dict(
        inputs=jnp.asarray([10.0, 11.0, 12.0]),
        keepdims=True,
        adj_idx=None,
        where=None,
        dst_idx=jnp.asarray([0, 0, 1, 1, 2, 3]),
        src_idx=jnp.asarray([0, 1, 1, 2, 0, 3]),
        num_segments=3,
        indices_are_sorted=True,
        expected=jnp.asarray([21.0, 21.0, 23.0, 23.0, 10.0, 10.0]),
    ),
    'sparse index list batched': dict(
        inputs=jnp.asarray([[10, 11, 12], [20, 21, 22]]),
        keepdims=False,
        adj_idx=None,
        where=None,
        dst_idx=jnp.asarray([[0, 0, 1, 1, 2, 3], [0, 0, 1, 3, 2, 2]]),
        src_idx=jnp.asarray([[0, 1, 1, 2, 0, 3], [1, 2, 1, 3, 0, 2]]),
        num_segments=3,
        indices_are_sorted=False,
        expected=jnp.asarray([[21, 23, 10], [43, 21, 42]]),
    ),
})
def test_indexed_sum(
    inputs: Array,
    keepdims: bool,
    adj_idx: Optional[Integer[Array, '...']],
    where: Optional[Bool[Array, '...']],
    dst_idx: Optional[Integer[Array, '...']],
    src_idx: Optional[Integer[Array, '...']],
    num_segments: Optional[int],
    indices_are_sorted: bool,
    expected: Array,
) -> None:
  gathered_inputs = e3x.ops.gather_src(inputs, adj_idx=adj_idx, src_idx=src_idx)
  assert jnp.array_equal(
      e3x.ops.indexed_sum(
          inputs=gathered_inputs,
          keepdims=keepdims,
          adj_idx=adj_idx,
          where=where,
          dst_idx=dst_idx,
          num_segments=num_segments,
          indices_are_sorted=indices_are_sorted,
      ),
      expected,
  )


@subtests({
    'dense index list, keepdims=False': dict(
        inputs=jnp.asarray([10, 11, 12]),
        keepdims=False,
        adj_idx=jnp.asarray([[0, 1], [1, 2], [0, 3]]),
        where=jnp.asarray([[True, True], [True, True], [True, False]]),
        dst_idx=None,
        src_idx=None,
        num_segments=None,
        indices_are_sorted=False,
        expected=jnp.asarray([10, 11, 10]),
    ),
    'dense index list, keepdims=True': dict(
        inputs=jnp.asarray([10.0, 11.0, 12.0]),
        keepdims=True,
        adj_idx=jnp.asarray([[0, 1], [1, 2], [0, 3]]),
        where=jnp.asarray([[True, True], [True, True], [True, False]]),
        dst_idx=None,
        src_idx=None,
        num_segments=None,
        indices_are_sorted=False,
        expected=jnp.asarray([[10.0], [11.0], [10.0]]),
    ),
    'dense index list batched': dict(
        inputs=jnp.asarray([[10, 11, 12], [20, 21, 22]]),
        keepdims=False,
        adj_idx=jnp.asarray(
            [[[0, 1], [1, 2], [0, 3]], [[1, 2], [1, 3], [0, 2]]]
        ),
        where=jnp.asarray([
            [[True, True], [True, True], [True, False]],
            [[True, True], [True, False], [True, True]],
        ]),
        dst_idx=None,
        src_idx=None,
        num_segments=None,
        indices_are_sorted=False,
        expected=jnp.asarray([[10, 11, 10], [21, 21, 20]]),
    ),
    'sparse index list, keepdims=False': dict(
        inputs=jnp.asarray([10, 11, 12]),
        keepdims=False,
        adj_idx=None,
        where=None,
        dst_idx=jnp.asarray([0, 0, 1, 1, 2, 3]),
        src_idx=jnp.asarray([0, 1, 1, 2, 0, 3]),
        num_segments=3,
        indices_are_sorted=True,
        expected=jnp.asarray([10, 11, 10]),
    ),
    'sparse index list, keepdims=True': dict(
        inputs=jnp.asarray([10.0, 11.0, 12.0]),
        keepdims=True,
        adj_idx=None,
        where=None,
        dst_idx=jnp.asarray([0, 0, 1, 1, 2, 3]),
        src_idx=jnp.asarray([0, 1, 1, 2, 0, 3]),
        num_segments=3,
        indices_are_sorted=True,
        expected=jnp.asarray([10.0, 10.0, 11.0, 11.0, 10.0, 10.0]),
    ),
    'sparse index list batched': dict(
        inputs=jnp.asarray([[10, 11, 12], [20, 21, 22]]),
        keepdims=False,
        adj_idx=None,
        where=None,
        dst_idx=jnp.asarray([[0, 0, 1, 1, 2, 3], [0, 0, 1, 3, 2, 2]]),
        src_idx=jnp.asarray([[0, 1, 1, 2, 0, 3], [1, 2, 1, 3, 0, 2]]),
        num_segments=3,
        indices_are_sorted=False,
        expected=jnp.asarray([[10, 11, 10], [21, 21, 20]]),
    ),
})
def test_indexed_min(
    inputs: Array,
    keepdims: bool,
    adj_idx: Optional[Integer[Array, '...']],
    where: Optional[Bool[Array, '...']],
    dst_idx: Optional[Integer[Array, '...']],
    src_idx: Optional[Integer[Array, '...']],
    num_segments: Optional[int],
    indices_are_sorted: bool,
    expected: Array,
) -> None:
  gathered_inputs = e3x.ops.gather_src(inputs, adj_idx=adj_idx, src_idx=src_idx)
  assert jnp.array_equal(
      e3x.ops.indexed_min(
          inputs=gathered_inputs,
          keepdims=keepdims,
          adj_idx=adj_idx,
          where=where,
          dst_idx=dst_idx,
          num_segments=num_segments,
          indices_are_sorted=indices_are_sorted,
      ),
      expected,
  )


@subtests({
    'dense index list, keepdims=False': dict(
        inputs=jnp.asarray([10, 11, 12]),
        keepdims=False,
        adj_idx=jnp.asarray([[0, 1], [1, 2], [0, 3]]),
        where=jnp.asarray([[True, True], [True, True], [True, False]]),
        dst_idx=None,
        src_idx=None,
        num_segments=None,
        indices_are_sorted=False,
        expected=jnp.asarray([11, 12, 10]),
    ),
    'dense index list, keepdims=True': dict(
        inputs=jnp.asarray([10.0, 11.0, 12.0]),
        keepdims=True,
        adj_idx=jnp.asarray([[0, 1], [1, 2], [0, 3]]),
        where=jnp.asarray([[True, True], [True, True], [True, False]]),
        dst_idx=None,
        src_idx=None,
        num_segments=None,
        indices_are_sorted=False,
        expected=jnp.asarray([[11.0], [12.0], [10.0]]),
    ),
    'dense index list batched': dict(
        inputs=jnp.asarray([[10, 11, 12], [20, 21, 22]]),
        keepdims=False,
        adj_idx=jnp.asarray(
            [[[0, 1], [1, 2], [0, 3]], [[1, 2], [1, 3], [0, 2]]]
        ),
        where=jnp.asarray([
            [[True, True], [True, True], [True, False]],
            [[True, True], [True, False], [True, True]],
        ]),
        dst_idx=None,
        src_idx=None,
        num_segments=None,
        indices_are_sorted=False,
        expected=jnp.asarray([[11, 12, 10], [22, 21, 22]]),
    ),
    'sparse index list, keepdims=False': dict(
        inputs=jnp.asarray([10, 11, 12]),
        keepdims=False,
        adj_idx=None,
        where=None,
        dst_idx=jnp.asarray([0, 0, 1, 1, 2, 3]),
        src_idx=jnp.asarray([0, 1, 1, 2, 0, 3]),
        num_segments=3,
        indices_are_sorted=True,
        expected=jnp.asarray([11, 12, 10]),
    ),
    'sparse index list, keepdims=True': dict(
        inputs=jnp.asarray([10.0, 11.0, 12.0]),
        keepdims=True,
        adj_idx=None,
        where=None,
        dst_idx=jnp.asarray([0, 0, 1, 1, 2, 3]),
        src_idx=jnp.asarray([0, 1, 1, 2, 0, 3]),
        num_segments=3,
        indices_are_sorted=True,
        expected=jnp.asarray([11.0, 11.0, 12.0, 12.0, 10.0, 10.0]),
    ),
    'sparse index list batched': dict(
        inputs=jnp.asarray([[10, 11, 12], [20, 21, 22]]),
        keepdims=False,
        adj_idx=None,
        where=None,
        dst_idx=jnp.asarray([[0, 0, 1, 1, 2, 3], [0, 0, 1, 3, 2, 2]]),
        src_idx=jnp.asarray([[0, 1, 1, 2, 0, 3], [1, 2, 1, 3, 0, 2]]),
        num_segments=3,
        indices_are_sorted=False,
        expected=jnp.asarray([[11, 12, 10], [22, 21, 22]]),
    ),
})
def test_indexed_max(
    inputs: Array,
    keepdims: bool,
    adj_idx: Optional[Integer[Array, '...']],
    where: Optional[Bool[Array, '...']],
    dst_idx: Optional[Integer[Array, '...']],
    src_idx: Optional[Integer[Array, '...']],
    num_segments: Optional[int],
    indices_are_sorted: bool,
    expected: Array,
) -> None:
  gathered_inputs = e3x.ops.gather_src(inputs, adj_idx=adj_idx, src_idx=src_idx)
  assert jnp.array_equal(
      e3x.ops.indexed_max(
          inputs=gathered_inputs,
          keepdims=keepdims,
          adj_idx=adj_idx,
          where=where,
          dst_idx=dst_idx,
          num_segments=num_segments,
          indices_are_sorted=indices_are_sorted,
      ),
      expected,
  )


@subtests({
    'dense indices': dict(
        inputs=jnp.asarray([[-1.0, 1.0, 0.5], [-1.0, 1.0, 0.5]]),
        multiplicative_mask=None,
        adj_idx=jnp.asarray([[0, 1, 3], [2, 3, 3]]),
        where=jnp.asarray([[True, True, False], [True, False, False]]),
        dst_idx=None,
        num_segments=None,
        indices_are_sorted=False,
        expected=jnp.asarray([[0.11920292, 0.880797, 0.0], [1.0, 0.0, 0.0]]),
    ),
    'dense indices batched': dict(
        inputs=jnp.asarray([
            [[-1.0, 1.0, 0.5], [-1.0, 1.0, 0.5]],
            [[-1.0, 1.0, 0.5], [-1.0, 1.0, 0.5]],
        ]),
        multiplicative_mask=None,
        adj_idx=jnp.asarray([
            [[0, 1, 3], [2, 3, 3]],
            [[2, 3, 3], [0, 1, 3]],
        ]),
        where=jnp.asarray([
            [[True, True, False], [True, False, False]],
            [[True, False, False], [True, True, False]],
        ]),
        dst_idx=None,
        num_segments=None,
        indices_are_sorted=False,
        expected=jnp.asarray([
            [[0.11920292, 0.880797, 0.0], [1.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [0.11920292, 0.880797, 0.0]],
        ]),
    ),
    'dense indices with multiplicative mask': dict(
        inputs=jnp.asarray([[-1.0, 1.0, 0.5], [-1.0, 1.0, 0.5]]),
        multiplicative_mask=jnp.asarray([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
        adj_idx=jnp.asarray([[0, 1, 3], [2, 3, 3]]),
        where=jnp.asarray([[True, True, False], [True, False, False]]),
        dst_idx=None,
        num_segments=None,
        indices_are_sorted=False,
        expected=jnp.asarray([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
    ),
    'sparse indices': dict(
        inputs=jnp.asarray([-1.0, 1.0, 0.5]),
        multiplicative_mask=None,
        adj_idx=None,
        where=None,
        dst_idx=jnp.asarray([0, 0, 1]),
        num_segments=2,
        indices_are_sorted=True,
        expected=jnp.asarray([0.11920292, 0.880797, 1.0]),
    ),
    'sparse indices batched': dict(
        inputs=jnp.asarray([[-1.0, 1.0, 0.5], [0.5, -1.0, 1.0]]),
        multiplicative_mask=None,
        adj_idx=None,
        where=None,
        dst_idx=jnp.asarray([[0, 0, 1], [0, 1, 1]]),
        num_segments=2,
        indices_are_sorted=True,
        expected=jnp.asarray(
            [[0.11920292, 0.880797, 1.0], [1.0, 0.11920292, 0.880797]]
        ),
    ),
    'sparse indices with multiplicative mask': dict(
        inputs=jnp.asarray([-1.0, 1.0, 0.5]),
        multiplicative_mask=jnp.asarray([0.0, 1.0, 1.0]),
        adj_idx=None,
        where=None,
        dst_idx=jnp.asarray([0, 0, 1]),
        num_segments=2,
        indices_are_sorted=True,
        expected=jnp.asarray([0.0, 1.0, 1.0]),
    ),
})
def test_indexed_softmax(
    inputs: Float[Array, '...'],
    multiplicative_mask: Optional[Float[Array, '...']],
    adj_idx: Optional[Integer[Array, '...']],
    where: Optional[Bool[Array, '...']],
    dst_idx: Optional[Integer[Array, '...']],
    num_segments: Optional[int],
    indices_are_sorted: bool,
    expected: Array,
) -> None:
  result = e3x.ops.indexed_softmax(
      inputs=inputs,
      multiplicative_mask=multiplicative_mask,
      adj_idx=adj_idx,
      where=where,
      dst_idx=dst_idx,
      num_segments=num_segments,
      indices_are_sorted=indices_are_sorted,
  )
  if where is not None:  # Extract only entries with well-defined values.
    result = result[where]
    expected = expected[where]
  assert jnp.allclose(result, expected, atol=1e-6)


@pytest.mark.parametrize('padding', [1000.0, -1000.0, np.inf, -np.inf, np.nan])
@pytest.mark.parametrize('use_cutoff', [False, True])
def test_dense_softmax_masked_logits_have_zero_finite_gradients(padding, use_cutoff):
  inputs = jnp.array([[0.0, 1.0, padding]])
  where = jnp.array([[True, True, False]])
  adj_idx = jnp.array([[0, 1, 2]])
  cutoff = jnp.array([[0.5, 1.0, 2.0]]) if use_cutoff else None

  def output(x):
    return e3x.ops.indexed_softmax(
        x, adj_idx=adj_idx, where=where, multiplicative_mask=cutoff
    )

  def objective(x):
    return jnp.where(where, output(x) * jnp.array([[1.0, 3.0, 0.0]]), 0.0).sum()

  def reference(x):
    weights = jnp.exp(x - jnp.max(x))
    if use_cutoff:
      weights = weights * jnp.array([0.5, 1.0])
    weights = weights / weights.sum()
    return (weights * jnp.array([1.0, 3.0])).sum()

  expected_value, expected_gradient = jax.value_and_grad(reference)(inputs[0, :2])
  for call in (jax.value_and_grad(objective), jax.jit(jax.value_and_grad(objective))):
    value, gradient = call(inputs)
    np.testing.assert_allclose(value, expected_value, rtol=1e-6)
    np.testing.assert_allclose(gradient[0, :2], expected_gradient, rtol=1e-5, atol=1e-7)
    np.testing.assert_array_equal(gradient[0, 2:], 0.0)
    assert bool(jnp.all(jnp.isfinite(gradient)))
  np.testing.assert_array_equal(output(inputs)[0, 2:], 0.0)


def test_dense_softmax_empty_neighborhoods_are_zero_under_vmap_and_jit():
  values = jnp.array([[[0.0, 2.0], [1000.0, np.nan]], [[1.0, 4.0], [np.inf, -np.inf]]])
  where = jnp.array([[True, True], [False, False]])
  adj_idx = jnp.array([[0, 1], [0, 1]])

  def apply(x):
    return e3x.ops.indexed_softmax(x, adj_idx=adj_idx, where=where)

  output = jax.jit(jax.vmap(apply))(values)
  np.testing.assert_allclose(
      output[:, 0], jax.nn.softmax(values[:, 0], axis=-1), rtol=1e-6
  )
  np.testing.assert_array_equal(output[:, 1], 0.0)
  gradient = jax.jit(jax.grad(lambda x: jax.vmap(apply)(x).sum()))(values)
  assert bool(jnp.all(jnp.isfinite(gradient)))
  np.testing.assert_array_equal(gradient[:, 1], 0.0)


def test_dense_softmax_matches_sparse_values_and_cutoff_gradients():
  values = jnp.array([[0.0, 1.0, 1000.0], [3.0, 4.0, -1000.0]])
  where = jnp.array([[True, True, False], [True, True, False]])
  adj_idx = jnp.array([[0, 1, 2], [0, 1, 2]])
  cutoff = jnp.array([[0.4, 0.8, 0.2], [0.3, 0.9, 0.5]])

  def dense(c):
    return e3x.ops.indexed_softmax(
        values, adj_idx=adj_idx, where=where, multiplicative_mask=c
    )

  def sparse(c):
    return e3x.ops.indexed_softmax(
        values[:, :2].reshape(-1),
        multiplicative_mask=c[:, :2].reshape(-1),
        dst_idx=jnp.array([0, 0, 1, 1]),
        num_segments=2,
    )

  np.testing.assert_allclose(
      dense(cutoff)[:, :2].reshape(-1), sparse(cutoff), rtol=1e-6
  )
  coefficients = jnp.array([1.0, 3.0, 2.0, 4.0])
  grad_dense = jax.grad(lambda c: (dense(c)[:, :2].reshape(-1) * coefficients).sum())(
      cutoff
  )
  grad_sparse = jax.grad(lambda c: (sparse(c) * coefficients).sum())(cutoff)
  np.testing.assert_allclose(grad_dense, grad_sparse, rtol=1e-5, atol=1e-6)


def test_self_attention_backward_remains_finite_with_an_empty_neighborhood():
  inputs = jnp.arange(6.0, dtype=jnp.float32).reshape(3, 1, 1, 2) / 4.0
  adj_idx = jnp.array([[0, 1], [0, 1], [0, 0]])
  where = jnp.array([[True, True], [True, True], [False, False]])
  attention = e3x.nn.SelfAttention(
      num_heads=1,
      use_relative_positional_encoding_qk=False,
      use_relative_positional_encoding_v=False,
  )
  variables = attention.init(
      jax.random.PRNGKey(4), inputs=inputs, adj_idx=adj_idx, where=where
  )

  def loss(params):
    return jnp.square(
        attention.apply(params, inputs=inputs, adj_idx=adj_idx, where=where) - 1.0
    ).sum()

  value, gradients = jax.jit(jax.value_and_grad(loss))(variables)
  assert bool(jnp.isfinite(value))
  for gradient in jax.tree_util.tree_leaves(gradients):
    assert bool(jnp.all(jnp.isfinite(gradient)))
  updated = jax.tree_util.tree_map(lambda p, g: p - 0.001 * g, variables, gradients)
  assert float(loss(updated)) < float(value)
