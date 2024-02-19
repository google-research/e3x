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

import math
from typing import Any, List, Sequence, Tuple, Union
import e3x
from ..testing import subtests
import jax
import jax.numpy as jnp
import jaxtyping
import pytest

Array = jaxtyping.Array
Float = jaxtyping.Float
DType = Any
Shape = Sequence[Union[int, Any]]


@pytest.mark.parametrize(
    'num_parity, num_degree, expected',
    [
        (1, 1, [(0, 0, 0)]),
        (2, 1, [(0, 0, 0), (1, 0, 1)]),
        (1, 2, [(0, 0, 0), (0, 1, 1)]),
        (2, 2, [(0, 0, 0), (0, 1, 0), (1, 0, 1), (1, 1, 1)]),
        (1, 3, [(0, 0, 0), (0, 1, 1), (0, 2, 0)]),
        (
            2,
            3,
            [(0, 0, 0), (0, 1, 0), (0, 2, 0), (1, 0, 1), (1, 1, 1), (1, 2, 1)],
        ),
        (1, 4, [(0, 0, 0), (0, 1, 1), (0, 2, 0), (0, 3, 1)]),
        (
            2,
            4,
            [
                (0, 0, 0),
                (0, 1, 0),
                (0, 2, 0),
                (0, 3, 0),
                (1, 0, 1),
                (1, 1, 1),
                (1, 2, 1),
                (1, 3, 1),
            ],
        ),
    ],
)
def test__parity_degree_index_parity_list(
    num_parity: int, num_degree: int, expected: List[Tuple[int, int, int]]
) -> None:
  assert (
      e3x.nn.initializers._parity_degree_index_parity_list(
          num_parity, num_degree
      )
      == expected
  )


def test__parity_degree_index_parity_list_raises_with_invalid_num_parity() -> (
    None
):
  with pytest.raises(ValueError, match='num_parity should be 1 or 2'):
    e3x.nn.initializers._parity_degree_index_parity_list(
        num_parity=3, num_degree=1
    )


@subtests({
    'float variance': dict(
        shape=(1000000,),
        variance=0.5,
        expected_std=math.sqrt(0.5),
    ),
    'array variance': dict(
        shape=(1000000, 3),
        variance=jnp.asarray([1.0, 2.0, 0.7]),
        expected_std=jnp.asarray([1.0, math.sqrt(2.0), math.sqrt(0.7)]),
    ),
})
@pytest.mark.parametrize(
    'distribution', [*e3x.nn.initializers._valid_distributions]
)
@pytest.mark.parametrize('dtype', [jnp.float_, jnp.complex_])
def test__random_array(
    shape: Shape,
    distribution: e3x.nn.initializers.Distribution,
    variance: Union[float, Float[Array, '...']],
    dtype: DType,
    expected_std: Union[float, Float[Array, '...']],
) -> None:
  array = e3x.nn.initializers._random_array(
      key=jax.random.PRNGKey(0),
      shape=shape,
      distribution=distribution,
      variance=variance,
      dtype=dtype,
  )
  assert jnp.isclose(jnp.mean(array), 0.0, atol=0.1)
  assert jnp.allclose(jnp.std(array, axis=0), expected_std, atol=0.1)


def test__random_array_raises_with_invalid_distribution() -> None:
  with pytest.raises(ValueError, match='invalid distribution'):
    e3x.nn.initializers._random_array(
        key=jax.random.PRNGKey(0), shape=(1,), distribution='foo'  # pytype: disable=wrong-arg-types
    )


@pytest.mark.parametrize(
    'shape, expected_fan_in, expected_fan_out, expected_mask',
    [
        (
            (1, 1, 1, 1, 1, 1, 1),
            jnp.asarray([[[[[[[1.0]]]]]]]),
            jnp.asarray([[[[[[[1.0]]]]]]]),
            jnp.asarray([[[[[[[True]]]]]]]),
        ),
        (
            (1, 1, 1, 1, 1, 1, 3),
            jnp.asarray([[[[[[[1.0]]]]]]]),
            jnp.asarray([[[[[[[1.0]]]]]]]),
            jnp.asarray([[[[[[[True, True, True]]]]]]]),
        ),
        (
            (2, 1, 1, 1, 1, 1, 1),
            jnp.asarray([[[[[[[1.0]]]]]]]),
            jnp.asarray([[[[[[[1.0]]]]]], [[[[[[0.0]]]]]]]),
            jnp.asarray([[[[[[[True]]]]]], [[[[[[False]]]]]]]),
        ),
        (
            (1, 1, 2, 1, 1, 1, 1),
            jnp.asarray([[[[[[[1.0]]]]]]]),
            jnp.asarray([[[[[[[1.0]]]], [[[[0.0]]]]]]]),
            jnp.asarray([[[[[[[True]]]], [[[[False]]]]]]]),
        ),
        (
            (1, 1, 1, 1, 2, 1, 1),
            jnp.asarray([[[[[[[1.0]], [[0.0]]]]]]]),
            jnp.asarray([[[[[[[1.0]]]]]]]),
            jnp.asarray([[[[[[[True]], [[False]]]]]]]),
        ),
        (
            (2, 1, 2, 1, 2, 1, 1),
            jnp.asarray([[[[[[[2.0]], [[2.0]]]]]]]),
            jnp.asarray(
                [[[[[[[1.0]]]], [[[[1.0]]]]]], [[[[[[1.0]]]], [[[[1.0]]]]]]]
            ),
            jnp.asarray([
                [[[[[[True]], [[False]]]], [[[[False]], [[True]]]]]],
                [[[[[[False]], [[True]]]], [[[[True]], [[False]]]]]],
            ]),
        ),
        (
            (2, 2, 2, 2, 2, 2, 1),
            jnp.asarray([[[[[[[4.0], [6.0]], [[4.0], [6.0]]]]]]]),
            jnp.asarray([
                [
                    [[[[[1.0]]], [[[1.0]]]], [[[[1.0]]], [[[1.0]]]]],
                    [[[[[1.0]]], [[[2.0]]]], [[[[1.0]]], [[[2.0]]]]],
                ],
                [
                    [[[[[1.0]]], [[[1.0]]]], [[[[1.0]]], [[[1.0]]]]],
                    [[[[[1.0]]], [[[2.0]]]], [[[[1.0]]], [[[2.0]]]]],
                ],
            ]),
            jnp.asarray([
                [
                    [
                        [
                            [[[True], [False]], [[False], [False]]],
                            [[[False], [True]], [[False], [False]]],
                        ],
                        [
                            [[[False], [False]], [[True], [False]]],
                            [[[False], [False]], [[False], [True]]],
                        ],
                    ],
                    [
                        [
                            [[[False], [True]], [[False], [False]]],
                            [[[True], [True]], [[False], [False]]],
                        ],
                        [
                            [[[False], [False]], [[False], [True]]],
                            [[[False], [False]], [[True], [True]]],
                        ],
                    ],
                ],
                [
                    [
                        [
                            [[[False], [False]], [[True], [False]]],
                            [[[False], [False]], [[False], [True]]],
                        ],
                        [
                            [[[True], [False]], [[False], [False]]],
                            [[[False], [True]], [[False], [False]]],
                        ],
                    ],
                    [
                        [
                            [[[False], [False]], [[False], [True]]],
                            [[[False], [False]], [[True], [True]]],
                        ],
                        [
                            [[[False], [True]], [[False], [False]]],
                            [[[True], [True]], [[False], [False]]],
                        ],
                    ],
                ],
            ]),
        ),
    ],
)
def test__compute_tensor_fans_and_mask(
    shape: Shape,
    expected_fan_in: Any,
    expected_fan_out: Any,
    expected_mask: Any,
) -> None:
  fan_in, fan_out, mask = e3x.nn.initializers._compute_tensor_fans_and_mask(
      shape
  )
  assert jnp.array_equal(fan_in, expected_fan_in)
  assert jnp.array_equal(fan_out, expected_fan_out)
  assert jnp.array_equal(mask, expected_mask)


def test_tensor_variance_scaling() -> None:
  key = jax.random.PRNGKey(0)
  shape = (1, 1, 1, 1, 1, 1, 1)
  expected = jax.random.normal(key, shape)
  output = e3x.nn.initializers.tensor_variance_scaling(
      scale=1.0, mode='fan_avg', distribution='normal'
  )(key=key, shape=shape)
  assert jnp.array_equal(output, expected)


def test_tensor_variance_scaling_raises_with_invalid_mode() -> None:
  with pytest.raises(ValueError, match='invalid mode'):
    e3x.nn.initializers.tensor_variance_scaling(
        scale=1.0, mode='foo', distribution='normal'  # pytype: disable=wrong-arg-types
    )(key=jax.random.PRNGKey(0), shape=(1, 1, 1, 1, 1, 1, 1))


def test_tensor_variance_scaling_raises_with_invalid_distribution() -> None:
  with pytest.raises(ValueError, match='invalid distribution'):
    e3x.nn.initializers.tensor_variance_scaling(
        scale=1.0, mode='fan_avg', distribution='foo'  # pytype: disable=wrong-arg-types
    )(key=jax.random.PRNGKey(0), shape=(1, 1, 1, 1, 1, 1, 1))


@pytest.mark.parametrize(
    'distribution', [*e3x.nn.initializers._valid_distributions]
)
def test__fused_tensor_init(
    distribution: e3x.nn.initializers.Distribution,
) -> None:
  key = jax.random.PRNGKey(0)
  shape = (16,)
  expected = e3x.nn.initializers._random_array(key, shape, distribution)
  output = e3x.nn.initializers._fused_tensor_init(distribution=distribution)(
      key, shape
  )
  assert jnp.array_equal(output, expected)


def test_fused_tensor_normal() -> None:
  key = jax.random.PRNGKey(0)
  shape = (16,)
  expected = e3x.nn.initializers._fused_tensor_init(
      distribution='truncated_normal'
  )(key, shape)
  output = e3x.nn.initializers.fused_tensor_normal()(key, shape)
  assert jnp.array_equal(output, expected)


def test_fused_tensor_uniform() -> None:
  key = jax.random.PRNGKey(0)
  shape = (16,)
  expected = e3x.nn.initializers._fused_tensor_init(distribution='uniform')(
      key, shape
  )
  output = e3x.nn.initializers.fused_tensor_uniform()(key, shape)
  assert jnp.array_equal(output, expected)
