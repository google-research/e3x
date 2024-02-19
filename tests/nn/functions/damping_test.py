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
import jax.numpy as jnp
import jaxtyping
import pytest

Array = jaxtyping.Array
Float = jaxtyping.Float


@pytest.mark.parametrize(
    'x, gamma, expected',
    [
        (
            jnp.asarray([0.0, 1e6]),
            1.0,
            jnp.asarray([0.0, 1.0]),
        ),
        (
            jnp.asarray(
                [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            ),
            1.0,
            jnp.asarray([
                0.0000000e00,
                7.4230666e-05,
                1.0925499e-02,
                5.7366449e-02,
                1.3091008e-01,
                2.1406102e-01,
                2.9630405e-01,
                3.7290415e-01,
                4.4221187e-01,
                5.0403088e-01,
                5.5879271e-01,
            ]),
        ),
        (
            jnp.asarray([[0.0, 0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8, 0.9]]),
            2.0,
            jnp.asarray([
                [0.0000000, 0.01092550, 0.13091008, 0.29630405, 0.44221187],
                [0.5587927, 0.64985067, 0.72085965, 0.77649087, 0.82034224],
            ]),
        ),
    ],
)
def test_smooth_damping(
    x: Float[Array, '...'],
    gamma: float,
    expected: Float[Array, '...'],
) -> None:
  assert jnp.allclose(e3x.nn.smooth_damping(x, gamma=gamma), expected)


def test_smooth_damping_has_nan_safe_derivatives() -> None:
  finfo = jnp.finfo(jnp.float32)
  x = jnp.asarray([0.0, finfo.tiny, finfo.eps])
  for y in e3x.ops.evaluate_derivatives(e3x.nn.smooth_damping, x, max_order=4):
    assert jnp.all(jnp.isfinite(y))
