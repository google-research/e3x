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

import re
from typing import Any, Optional, Sequence, Tuple, Union
import e3x
from ..testing import subtests
import jax
import jax.numpy as jnp
import jaxtyping
import pytest

Array = jaxtyping.Array
Bool = jaxtyping.Bool
Float = jaxtyping.Float
Num = jaxtyping.Num
Shaped = jaxtyping.Shaped
Shape = Sequence[Union[int, Any]]


@pytest.mark.parametrize(
    'shape, expected',
    [
        ((1, 1, 8), 0),
        ((3, 1, 2, 4, 16), 1),
        ((1, 9, 2), 2),
        ((128, 2, 16, 32), 3),
    ],
)
def test__extract_max_degree_and_check_shape(
    shape: Shape, expected: int
) -> None:
  degree = e3x.nn.features._extract_max_degree_and_check_shape(shape)
  assert degree == expected


def test__extract_max_degree_and_check_shape_raises_with_invalid_length() -> (
    None
):
  with pytest.raises(ValueError, match='must have at least length 3'):
    e3x.nn.features._extract_max_degree_and_check_shape((1, 2))


def test__extract_max_degree_and_check_shape_raises_with_invalid_parity() -> (
    None
):
  with pytest.raises(ValueError, match='expected 1 or 2 for axis -3'):
    e3x.nn.features._extract_max_degree_and_check_shape((3, 1, 1))


def test__extract_max_degree_and_check_shape_raises_with_invalid_degree() -> (
    None
):
  with pytest.raises(ValueError, match='closest valid size is 9'):
    e3x.nn.features._extract_max_degree_and_check_shape((1, 8, 1))


@pytest.mark.parametrize(
    'max_degree, expected',
    [
        (0, jnp.asarray([[True]])),
        (1, jnp.asarray([[True], [False], [False], [False]])),
        (
            2,
            jnp.asarray([
                [True],
                [False],
                [False],
                [False],
                [True],
                [True],
                [True],
                [True],
                [True],
            ]),
        ),
    ],
)
def test_even_degree_mask(
    max_degree: int, expected: Bool[Array, '(max_degree+1)**2 1']
) -> None:
  assert jnp.array_equal(e3x.nn.features.even_degree_mask(max_degree), expected)


@pytest.mark.parametrize(
    'max_degree, expected',
    [
        (0, jnp.asarray([[False]])),
        (1, jnp.asarray([[False], [True], [True], [True]])),
        (
            2,
            jnp.asarray([
                [False],
                [True],
                [True],
                [True],
                [False],
                [False],
                [False],
                [False],
                [False],
            ]),
        ),
    ],
)
def test_odd_degree_mask(
    max_degree: int, expected: Bool[Array, '(max_degree+1)**2 1']
) -> None:
  assert jnp.array_equal(e3x.nn.features.odd_degree_mask(max_degree), expected)


@pytest.mark.parametrize('num_batch', [1, 4])
@pytest.mark.parametrize('max_degree', [0, 2])
def test_rotate(num_batch: int, max_degree: int, num_features: int = 8) -> None:
  # Draw random vectors.
  r_key, rot_key = jax.random.split(jax.random.PRNGKey(0), num=2)
  r = jax.random.normal(r_key, (num_batch, num_features, 3))
  # Draw random rotation matrices and rotate vectors.
  rot = e3x.so3.random_rotation(rot_key, num=num_batch)
  if num_batch == 1:  # Add missing batch dimension for num_batch = 1.
    rot = jnp.expand_dims(rot, axis=0)
  r_rot = jnp.einsum('...fa,...ab->...fb', r, rot)
  # Expand (rotated) vectors in spherical harmonics.
  x = e3x.so3.spherical_harmonics(r, max_degree=max_degree)
  x_rot = e3x.so3.spherical_harmonics(r_rot, max_degree=max_degree)
  # Swap axes and reshape to simulate features.
  x = jnp.expand_dims(jnp.swapaxes(x, -1, -2), axis=-3)
  x_rot = jnp.expand_dims(jnp.swapaxes(x_rot, -1, -2), axis=-3)
  # Construct Wigner-D matrices from random rotation matrices.
  wigner_d = e3x.so3.wigner_d(rot, max_degree=max_degree)
  # Squeeze batch dimensions (removed batch dimensions if num_batch = 1).
  if num_batch == 1:
    x = jnp.squeeze(x, axis=0)
    x_rot = jnp.squeeze(x_rot, axis=0)
    wigner_d = jnp.squeeze(wigner_d, axis=0)
  # Check that rotated features match features obtained from rotated vectors.
  assert jnp.allclose(e3x.nn.rotate(x, wigner_d), x_rot, atol=1e-5)


@pytest.mark.parametrize(
    'wigner_d, message',
    [
        (jnp.zeros((3, 3)), 'expected shape (..., 4, 4)'),
        (jnp.zeros((3, 4, 4)), 'incompatible shapes'),
    ],
)
def test_rotate_raises_with_invalid_wigner_d_matrix_shapes(
    wigner_d: Float[Array, '...'], message: str
) -> None:
  x = jnp.zeros((2, 1, 4, 8))
  with pytest.raises(ValueError, match=re.escape(message)):
    e3x.nn.rotate(x, wigner_d)


@pytest.mark.parametrize(
    'x, expected',
    [
        (jnp.asarray([[[1.0]]]), jnp.asarray([[[1.0]]])),
        (
            jnp.asarray([[[0.1, 0.2]], [[1.1, -1.2]]]),
            jnp.asarray([[[0.1, 0.2]], [[-1.1, 1.2]]]),
        ),
        (
            jnp.asarray([[[0.3], [0.2], [-0.8], [1.9]]]),
            jnp.asarray([[[0.3], [-0.2], [0.8], [-1.9]]]),
        ),
        (
            jnp.asarray(
                [[[0.3], [0.2], [-0.8], [1.9]], [[1.3], [-0.1], [1.2], [0.4]]]
            ),
            jnp.asarray(
                [[[0.3], [0.2], [-0.8], [1.9]], [[-1.3], [0.1], [-1.2], [-0.4]]]
            ),
        ),
    ],
)
def test_reflect(
    x: Union[
        Float[Array, '... 1 (max_degree+1)**2 num_features'],
        Float[Array, '... 2 (max_degree+1)**2 num_features'],
    ],
    expected: Union[
        Float[Array, '... 1 (max_degree+1)**2 num_features'],
        Float[Array, '... 2 (max_degree+1)**2 num_features'],
    ],
) -> None:
  assert jnp.array_equal(e3x.nn.reflect(x), expected)


@subtests({
    'apply no changes': dict(
        x=jnp.asarray([[[0.0000, 0.0001]]]),
        max_degree=None,
        include_pseudotensors=None,
        expected=jnp.asarray([[[0.0000, 0.0001]]]),
    ),
    'apply no changes (batched)': dict(
        x=jnp.asarray([[[[0.0000, 0.0001]]], [[[1.0000, 1.0001]]]]),
        max_degree=None,
        include_pseudotensors=None,
        expected=jnp.asarray([[[[0.0000, 0.0001]]], [[[1.0000, 1.0001]]]]),
    ),
    'increase max_degree (no pseudotensors)': dict(
        x=jnp.asarray([[[0.0000, 0.0001]]]),
        max_degree=1,
        include_pseudotensors=None,
        expected=jnp.asarray(([[[0.0000, 0.0001], [0, 0], [0, 0], [0, 0]]])),
    ),
    'increase max_degree (no pseudotensors, batched)': dict(
        x=jnp.asarray([[[[0.0000, 0.0001]]], [[[1.0000, 1.0001]]]]),
        max_degree=1,
        include_pseudotensors=None,
        expected=jnp.asarray(([
            [[[0.0000, 0.0001], [0, 0], [0, 0], [0, 0]]],
            [[[1.0000, 1.0001], [0, 0], [0, 0], [0, 0]]],
        ])),
    ),
    'increase max_degree (with pseudotensors)': dict(
        x=jnp.asarray([[[0.0000, 0.0001]], [[0.1000, 0.1001]]]),
        max_degree=1,
        include_pseudotensors=None,
        expected=jnp.asarray(([
            [[0.0000, 0.0001], [0, 0], [0, 0], [0, 0]],
            [[0.1000, 0.1001], [0, 0], [0, 0], [0, 0]],
        ])),
    ),
    'increase max_degree (with pseudotensors, batched)': dict(
        x=jnp.asarray([
            [[[0.0000, 0.0001]], [[0.1000, 0.1001]]],
            [[[1.0000, 1.0001]], [[1.1000, 1.1001]]],
        ]),
        max_degree=1,
        include_pseudotensors=None,
        expected=jnp.asarray(([
            [
                [[0.0000, 0.0001], [0, 0], [0, 0], [0, 0]],
                [[0.1000, 0.1001], [0, 0], [0, 0], [0, 0]],
            ],
            [
                [[1.0000, 1.0001], [0, 0], [0, 0], [0, 0]],
                [[1.1000, 1.1001], [0, 0], [0, 0], [0, 0]],
            ],
        ])),
    ),
    'decrease max_degree (no pseudotensors)': dict(
        x=jnp.asarray(([[
            [0.0000, 0.0001],
            [0.0100, 0.0101],
            [0.0110, 0.0111],
            [0.0120, 0.0121],
        ]])),
        max_degree=0,
        include_pseudotensors=None,
        expected=jnp.asarray(([[[0.0000, 0.0001]]])),
    ),
    'decrease max_degree (no pseudotensors, batched)': dict(
        x=jnp.asarray(([
            [[
                [0.0000, 0.0001],
                [0.0100, 0.0101],
                [0.0110, 0.0111],
                [0.0120, 0.0121],
            ]],
            [[
                [1.0000, 1.0001],
                [1.0100, 1.0101],
                [1.0110, 1.0111],
                [1.0120, 1.0121],
            ]],
        ])),
        max_degree=0,
        include_pseudotensors=None,
        expected=jnp.asarray(([[[[0.0000, 0.0001]]], [[[1.0000, 1.0001]]]])),
    ),
    'decrease max_degree (with pseudotensors)': dict(
        x=jnp.asarray(([
            [
                [0.0000, 0.0001],
                [0.0100, 0.0101],
                [0.0110, 0.0111],
                [0.0120, 0.0121],
            ],
            [
                [0.1000, 0.1001],
                [0.1100, 0.1101],
                [0.1110, 0.1111],
                [0.1120, 0.1121],
            ],
        ])),
        max_degree=0,
        include_pseudotensors=None,
        expected=jnp.asarray(([[[0.0000, 0.0001]], [[0.1000, 0.1001]]])),
    ),
    'decrease max_degree (with pseudotensors, batched)': dict(
        x=jnp.asarray(([
            [
                [
                    [0.0000, 0.0001],
                    [0.0100, 0.0101],
                    [0.0110, 0.0111],
                    [0.0120, 0.0121],
                ],
                [
                    [0.1000, 0.1001],
                    [0.1100, 0.1101],
                    [0.1110, 0.1111],
                    [0.1120, 0.1121],
                ],
            ],
            [
                [
                    [1.0000, 1.0001],
                    [1.0100, 1.0101],
                    [1.0110, 1.0111],
                    [1.0120, 1.0121],
                ],
                [
                    [1.1000, 1.1001],
                    [1.1100, 1.1101],
                    [1.1110, 1.1111],
                    [1.1120, 1.1121],
                ],
            ],
        ])),
        max_degree=0,
        include_pseudotensors=None,
        expected=jnp.asarray(([
            [[[0.0000, 0.0001]], [[0.1000, 0.1001]]],
            [[[1.0000, 1.0001]], [[1.1000, 1.1001]]],
        ])),
    ),
    'add pseudotensors': dict(
        x=jnp.asarray([[
            [0.0000, 0.0001],
            [0.0100, 0.0101],
            [0.0110, 0.0111],
            [0.0120, 0.0121],
        ]]),
        max_degree=None,
        include_pseudotensors=True,
        expected=jnp.asarray([
            [[0.0000, 0.0001], [0, 0], [0, 0], [0, 0]],
            [[0, 0], [0.0100, 0.0101], [0.0110, 0.0111], [0.0120, 0.0121]],
        ]),
    ),
    'add pseudotensors (batched)': dict(
        x=jnp.asarray([
            [[
                [0.0000, 0.0001],
                [0.0100, 0.0101],
                [0.0110, 0.0111],
                [0.0120, 0.0121],
            ]],
            [[
                [1.0000, 1.0001],
                [1.0100, 1.0101],
                [1.0110, 1.0111],
                [1.0120, 1.0121],
            ]],
        ]),
        max_degree=None,
        include_pseudotensors=True,
        expected=jnp.asarray([
            [
                [[0.0000, 0.0001], [0, 0], [0, 0], [0, 0]],
                [[0, 0], [0.0100, 0.0101], [0.0110, 0.0111], [0.0120, 0.0121]],
            ],
            [
                [[1.0000, 1.0001], [0, 0], [0, 0], [0, 0]],
                [[0, 0], [1.0100, 1.0101], [1.0110, 1.0111], [1.0120, 1.0121]],
            ],
        ]),
    ),
    'remove pseudotensors': dict(
        x=jnp.asarray([
            [
                [0.0000, 0.0001],
                [0.0100, 0.0101],
                [0.0110, 0.0111],
                [0.0120, 0.0121],
            ],
            [
                [0.1000, 0.1001],
                [0.1100, 0.1101],
                [0.1110, 0.1111],
                [0.1120, 0.1121],
            ],
        ]),
        max_degree=None,
        include_pseudotensors=False,
        expected=jnp.asarray([[
            [0.0000, 0.0001],
            [0.1100, 0.1101],
            [0.1110, 0.1111],
            [0.1120, 0.1121],
        ]]),
    ),
    'remove pseudotensors (batched)': dict(
        x=jnp.asarray([
            [
                [
                    [0.0000, 0.0001],
                    [0.0100, 0.0101],
                    [0.0110, 0.0111],
                    [0.0120, 0.0121],
                ],
                [
                    [0.1000, 0.1001],
                    [0.1100, 0.1101],
                    [0.1110, 0.1111],
                    [0.1120, 0.1121],
                ],
            ],
            [
                [
                    [1.0000, 1.0001],
                    [1.0100, 1.0101],
                    [1.0110, 1.0111],
                    [1.0120, 1.0121],
                ],
                [
                    [1.1000, 1.1001],
                    [1.1100, 1.1101],
                    [1.1110, 1.1111],
                    [1.1120, 1.1121],
                ],
            ],
        ]),
        max_degree=None,
        include_pseudotensors=False,
        expected=jnp.asarray([
            [[
                [0.0000, 0.0001],
                [0.1100, 0.1101],
                [0.1110, 0.1111],
                [0.1120, 0.1121],
            ]],
            [[
                [1.0000, 1.0001],
                [1.1100, 1.1101],
                [1.1110, 1.1111],
                [1.1120, 1.1121],
            ]],
        ]),
    ),
    'decrease max_degree and remove pseudotensors': dict(
        x=jnp.asarray(
            [
                [
                    [0.0000, 0.0001],
                    [0.0100, 0.0101],
                    [0.0110, 0.0111],
                    [0.0120, 0.0121],
                    [0.0200, 0.0201],
                    [0.0210, 0.0211],
                    [0.0220, 0.0221],
                    [0.0230, 0.0231],
                    [0.0240, 0.0241],
                ],
                [
                    [0.1000, 0.1001],
                    [0.1100, 0.1101],
                    [0.1110, 0.1111],
                    [0.1120, 0.1121],
                    [0.1200, 0.1201],
                    [0.1210, 0.1211],
                    [0.1220, 0.1221],
                    [0.1230, 0.1231],
                    [0.1240, 0.1241],
                ],
            ],
        ),
        max_degree=1,
        include_pseudotensors=False,
        expected=jnp.asarray([
            [
                [0.0000, 0.0001],
                [0.1100, 0.1101],
                [0.1110, 0.1111],
                [0.1120, 0.1121],
            ],
        ]),
    ),
    'decrease max_degree and remove pseudotensors (batched)': dict(
        x=jnp.asarray([
            [
                [
                    [0.0000, 0.0001],
                    [0.0100, 0.0101],
                    [0.0110, 0.0111],
                    [0.0120, 0.0121],
                    [0.0200, 0.0201],
                    [0.0210, 0.0211],
                    [0.0220, 0.0221],
                    [0.0230, 0.0231],
                    [0.0240, 0.0241],
                ],
                [
                    [0.1000, 0.1001],
                    [0.1100, 0.1101],
                    [0.1110, 0.1111],
                    [0.1120, 0.1121],
                    [0.1200, 0.1201],
                    [0.1210, 0.1211],
                    [0.1220, 0.1221],
                    [0.1230, 0.1231],
                    [0.1240, 0.1241],
                ],
            ],
            [
                [
                    [1.0000, 1.0001],
                    [1.0100, 1.0101],
                    [1.0110, 1.0111],
                    [1.0120, 1.0121],
                    [1.0200, 1.0201],
                    [1.0210, 1.0211],
                    [1.0220, 1.0221],
                    [1.0230, 1.0231],
                    [1.0240, 1.0241],
                ],
                [
                    [1.1000, 1.1001],
                    [1.1100, 1.1101],
                    [1.1110, 1.1111],
                    [1.1120, 1.1121],
                    [1.1200, 1.1201],
                    [1.1210, 1.1211],
                    [1.1220, 1.1221],
                    [1.1230, 1.1231],
                    [1.1240, 1.1241],
                ],
            ],
        ]),
        max_degree=1,
        include_pseudotensors=False,
        expected=jnp.asarray([
            [[
                [0.0000, 0.0001],
                [0.1100, 0.1101],
                [0.1110, 0.1111],
                [0.1120, 0.1121],
            ]],
            [[
                [1.0000, 1.0001],
                [1.1100, 1.1101],
                [1.1110, 1.1111],
                [1.1120, 1.1121],
            ]],
        ]),
    ),
    'increase max_degree and remove pseudotensors': dict(
        x=jnp.asarray(
            [
                [
                    [0.0000, 0.0001],
                    [0.0100, 0.0101],
                    [0.0110, 0.0111],
                    [0.0120, 0.0121],
                ],
                [
                    [0.1000, 0.1001],
                    [0.1100, 0.1101],
                    [0.1110, 0.1111],
                    [0.1120, 0.1121],
                ],
            ],
        ),
        max_degree=2,
        include_pseudotensors=False,
        expected=jnp.asarray(
            [[
                [0.0000, 0.0001],
                [0.1100, 0.1101],
                [0.1110, 0.1111],
                [0.1120, 0.1121],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
            ]],
        ),
    ),
    'increase max_degree and remove pseudotensors (batched)': dict(
        x=jnp.asarray([
            [
                [
                    [0.0000, 0.0001],
                    [0.0100, 0.0101],
                    [0.0110, 0.0111],
                    [0.0120, 0.0121],
                ],
                [
                    [0.1000, 0.1001],
                    [0.1100, 0.1101],
                    [0.1110, 0.1111],
                    [0.1120, 0.1121],
                ],
            ],
            [
                [
                    [1.0000, 1.0001],
                    [1.0100, 1.0101],
                    [1.0110, 1.0111],
                    [1.0120, 1.0121],
                ],
                [
                    [1.1000, 1.1001],
                    [1.1100, 1.1101],
                    [1.1110, 1.1111],
                    [1.1120, 1.1121],
                ],
            ],
        ]),
        max_degree=2,
        include_pseudotensors=False,
        expected=jnp.asarray([
            [
                [
                    [0.0000, 0.0001],
                    [0.1100, 0.1101],
                    [0.1110, 0.1111],
                    [0.1120, 0.1121],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                ],
            ],
            [
                [
                    [1.0000, 1.0001],
                    [1.1100, 1.1101],
                    [1.1110, 1.1111],
                    [1.1120, 1.1121],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                ],
            ],
        ]),
    ),
    'decrease max_degree and add pseudotensors': dict(
        x=jnp.asarray(
            [[
                [0.0000, 0.0001],
                [0.0100, 0.0101],
                [0.0110, 0.0111],
                [0.0120, 0.0121],
                [0.0200, 0.0201],
                [0.0210, 0.0211],
                [0.0220, 0.0221],
                [0.0230, 0.0231],
                [0.0240, 0.0241],
            ]],
        ),
        max_degree=1,
        include_pseudotensors=True,
        expected=jnp.asarray(
            [
                [
                    [0.0000, 0.0001],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                ],
                [
                    [0, 0],
                    [0.0100, 0.0101],
                    [0.0110, 0.0111],
                    [0.0120, 0.0121],
                ],
            ],
        ),
    ),
    'decrease max_degree and add pseudotensors (batched)': dict(
        x=jnp.asarray([
            [[
                [0.0000, 0.0001],
                [0.0100, 0.0101],
                [0.0110, 0.0111],
                [0.0120, 0.0121],
                [0.0200, 0.0201],
                [0.0210, 0.0211],
                [0.0220, 0.0221],
                [0.0230, 0.0231],
                [0.0240, 0.0241],
            ]],
            [[
                [1.0000, 1.0001],
                [1.0100, 1.0101],
                [1.0110, 1.0111],
                [1.0120, 1.0121],
                [1.0200, 1.0201],
                [1.0210, 1.0211],
                [1.0220, 1.0221],
                [1.0230, 1.0231],
                [1.0240, 1.0241],
            ]],
        ]),
        max_degree=1,
        include_pseudotensors=True,
        expected=jnp.asarray([
            [
                [
                    [0.0000, 0.0001],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                ],
                [
                    [0, 0],
                    [0.0100, 0.0101],
                    [0.0110, 0.0111],
                    [0.0120, 0.0121],
                ],
            ],
            [
                [
                    [1.0000, 1.0001],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                ],
                [
                    [0, 0],
                    [1.0100, 1.0101],
                    [1.0110, 1.0111],
                    [1.0120, 1.0121],
                ],
            ],
        ]),
    ),
    'increase max_degree and add pseudotensors': dict(
        x=jnp.asarray([[
            [0.0000, 0.0001],
            [0.0100, 0.0101],
            [0.0110, 0.0111],
            [0.0120, 0.0121],
        ]]),
        max_degree=2,
        include_pseudotensors=True,
        expected=jnp.asarray(
            [
                [
                    [0.0000, 0.0001],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                ],
                [
                    [0, 0],
                    [0.0100, 0.0101],
                    [0.0110, 0.0111],
                    [0.0120, 0.0121],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                ],
            ],
        ),
    ),
    'increase max_degree and add pseudotensors (batched)': dict(
        x=jnp.asarray([
            [[
                [0.0000, 0.0001],
                [0.0100, 0.0101],
                [0.0110, 0.0111],
                [0.0120, 0.0121],
            ]],
            [[
                [1.0000, 1.0001],
                [1.0100, 1.0101],
                [1.0110, 1.0111],
                [1.0120, 1.0121],
            ]],
        ]),
        max_degree=2,
        include_pseudotensors=True,
        expected=jnp.asarray([
            [
                [
                    [0.0000, 0.0001],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                ],
                [
                    [0, 0],
                    [0.0100, 0.0101],
                    [0.0110, 0.0111],
                    [0.0120, 0.0121],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                ],
            ],
            [
                [
                    [1.0000, 1.0001],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                ],
                [
                    [0, 0],
                    [1.0100, 1.0101],
                    [1.0110, 1.0111],
                    [1.0120, 1.0121],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                ],
            ],
        ]),
    ),
})
def test_change_max_degree_or_type(
    x: Union[
        Float[Array, '... 1 (max_degree+1)**2 num_features'],
        Float[Array, '... 2 (max_degree+1)**2 num_features'],
    ],
    max_degree: Optional[int],
    include_pseudotensors: Optional[bool],
    expected: Union[
        Float[Array, '... 1 (max_degree+1)**2 num_features'],
        Float[Array, '... 2 (max_degree+1)**2 num_features'],
    ],
) -> None:
  y = e3x.nn.change_max_degree_or_type(
      x, max_degree=max_degree, include_pseudotensors=include_pseudotensors
  )
  assert jnp.array_equal(y, expected)


@pytest.mark.parametrize(
    'x, error_message',
    [
        (jnp.empty((1, 1)), 'shape of features must have at least length 3'),
        (jnp.empty((3, 1, 1)), 'expected 1 or 2 for axis -3 of feature shape'),
    ],
)
def test_change_max_degree_or_type_raises_with_invalid_shape(
    x: Shaped[Array, '...'], error_message: str
) -> None:
  with pytest.raises(ValueError, match=re.escape(error_message)):
    e3x.nn.change_max_degree_or_type(x)


@pytest.mark.parametrize(
    'inputs, expected',
    [
        ((jnp.asarray([[[1.0]]]),), jnp.asarray([[[1.0]]])),
        (
            (jnp.asarray([[[1.0]]]), jnp.asarray([[[1.0]]])),
            jnp.asarray([[[2.0]]]),
        ),
        (
            (
                jnp.asarray([[[1.0]]]),
                jnp.asarray([[[1.0]]]),
                jnp.asarray([[[1.0]]]),
            ),
            jnp.asarray([[[3.0]]]),
        ),
        (
            (
                jnp.asarray([[[0.0], [0.1], [0.2], [0.3]]]),
                jnp.asarray([[[1.0], [1.1], [1.2], [1.3]]]),
            ),
            jnp.asarray([[[1.0], [1.2], [1.4], [1.6]]]),
        ),
        (
            (
                jnp.asarray([[[0.5]]]),
                jnp.asarray([[[1.0], [1.1], [1.2], [1.3]]]),
            ),
            jnp.asarray([[[1.5], [1.1], [1.2], [1.3]]]),
        ),
        (
            (
                jnp.asarray([[[0.5], [0.1], [0.2], [0.3]]]),
                jnp.asarray([
                    [[1.0], [1.1], [1.2], [1.3]],
                    [[-1.0], [-1.1], [-1.2], [-1.3]],
                ]),
            ),
            jnp.asarray([
                [[1.5], [1.1], [1.2], [1.3]],
                [[-1.0], [-1.0], [-1.0], [-1.0]],
            ]),
        ),
        (
            (
                jnp.asarray([
                    [[1.0], [1.1], [1.2], [1.3]],
                    [[-1.0], [-1.1], [-1.2], [-1.3]],
                ]),
                jnp.asarray([
                    [[1.0], [1.1], [1.2], [1.3]],
                    [[-1.0], [-1.1], [-1.2], [-1.3]],
                ]),
            ),
            jnp.asarray([
                [[2.0], [2.2], [2.4], [2.6]],
                [[-2.0], [-2.2], [-2.4], [-2.6]],
            ]),
        ),
    ],
)
def test_add(
    inputs: Tuple[
        Union[
            Float[Array, '... 1 (max_degree+1)**2 num_features'],
            Float[Array, '... 2 (max_degree+1)**2 num_features'],
        ],
        ...,
    ],
    expected: Union[
        Float[Array, '... 1 (max_degree+1)**2 num_features'],
        Float[Array, '... 2 (max_degree+1)**2 num_features'],
    ],
) -> None:
  assert jnp.allclose(e3x.nn.add(*inputs), expected, atol=1e-5)


@pytest.mark.parametrize(
    'inputs, message',
    [
        (
            (jnp.asarray([[[1.0]]]), jnp.asarray([[[[1.0]]]])),
            'must have the same leading dimensions',
        ),
        (
            (jnp.asarray([[[1.0, 1.0]]]), jnp.asarray([[[1.0]]])),
            'must have the same number of features',
        ),
        (
            (
                jnp.asarray([[[1.0]]], dtype=jnp.float32),
                jnp.asarray([[[1]]], dtype=jnp.int32),
            ),
            'must have the same dtype',
        ),
    ],
)
def test_add_raises_with_invalid_shapes(
    inputs: Tuple[Num[Array, '...'], ...], message: str
) -> None:
  with pytest.raises(ValueError, match=re.escape(message)):
    e3x.nn.add(*inputs)
