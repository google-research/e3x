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
import pytest


@pytest.mark.parametrize('x0, x1', [(0.0, 1.0), (-1.0, 2.0), (0.4, 0.6)])
def test_smooth_switch(x0: float, x1: float) -> None:
  x = jnp.linspace(x0, x1, num=11)
  expected_value = jnp.asarray([  # Same for both test cases.
      0.0000000e00,
      4.5352624e-04,
      3.7413392e-02,
      1.6116680e-01,
      3.2702142e-01,
      5.0000000e-01,
      6.7297864e-01,
      8.3883321e-01,
      9.6258658e-01,
      9.9954647e-01,
      1.0000000e00,
  ])
  expected_grad = jnp.asarray([
      0.0000000,
      0.0397434,
      0.8284502,
      1.5398244,
      1.7206353,
      1.7320508,
      1.7206357,
      1.5398246,
      0.8284502,
      0.0397433,
      0.0000000,
  ]) / (x1 - x0)
  value, grad = e3x.ops.evaluate_derivatives(
      lambda x: e3x.nn.smooth_switch(x, x0=x0, x1=x1), x, max_order=1
  )
  assert jnp.allclose(value, expected_value, atol=1e-5)
  assert jnp.allclose(grad, expected_grad, atol=1e-5)


def test_smooth_switch_has_nan_safe_derivatives() -> None:
  finfo = jnp.finfo(jnp.float32)
  x = jnp.asarray(
      [0.0, finfo.tiny, finfo.eps, 1 - finfo.epsneg, 1.0],
  )
  for y in e3x.ops.evaluate_derivatives(e3x.nn.smooth_switch, x, max_order=4):
    assert jnp.all(jnp.isfinite(y))


@pytest.mark.parametrize('x0, x1', [(1.0, 0.0), (0.5, 0.5)])
def test_smooth_switch_raises_with_invalid_x0_x1(x0: float, x1: float) -> None:
  with pytest.raises(ValueError, match='x1 must be larger than x0'):
    e3x.nn.smooth_switch(1, x0=x0, x1=x1)
