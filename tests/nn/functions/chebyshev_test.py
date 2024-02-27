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


def test__chebyshev() -> None:
  x = jnp.linspace(-1, 1, 11)
  expected_value = jnp.asarray([
      [1.0000000e00, -1.0000000e00, 1.0000000e00, -1.0000000e00],
      [1.0000000e00, -7.9999995e-01, 2.7999982e-01, 3.5200047e-01],
      [1.0000000e00, -6.0000008e-01, -2.7999982e-01, 9.3599981e-01],
      [1.0000000e00, -3.9999998e-01, -6.8000001e-01, 9.4400001e-01],
      [1.0000000e00, -2.0000008e-01, -9.1999996e-01, 5.6800020e-01],
      [1.0000000e00, -4.3711388e-08, -1.0000000e00, 1.1924881e-08],
      [1.0000000e00, 2.0000011e-01, -9.1999990e-01, -5.6800020e-01],
      [1.0000000e00, 4.0000001e-01, -6.8000001e-01, -9.4399995e-01],
      [1.0000000e00, 6.0000002e-01, -2.7999997e-01, -9.3599999e-01],
      [1.0000000e00, 8.0000007e-01, 2.8000024e-01, -3.5199958e-01],
      [1.0000000e00, 1.0000000e00, 1.0000000e00, 1.0000000e00],
  ])
  expected_grad = jnp.asarray([
      [-0.0000000e00, 1.0000000e00, -4.0000000e00, 9.0000000e00],
      [0.0000000e00, 1.0000000e00, -3.1999996e00, 4.6799984e00],
      [0.0000000e00, 9.9999994e-01, -2.4000001e00, 1.3200018e00],
      [0.0000000e00, 9.9999994e-01, -1.5999999e00, -1.0800002e00],
      [0.0000000e00, 9.9999994e-01, -8.0000031e-01, -2.5199993e00],
      [0.0000000e00, 1.0000000e00, -1.7484555e-07, -3.0000000e00],
      [0.0000000e00, 9.9999994e-01, 8.0000043e-01, -2.5199995e00],
      [0.0000000e00, 1.0000000e00, 1.6000000e00, -1.0800003e00],
      [0.0000000e00, 1.0000000e00, 2.3999999e00, 1.3200001e00],
      [0.0000000e00, 1.0000000e00, 3.2000003e00, 4.6800017e00],
      [0.0000000e00, 1.0000000e00, 4.0000000e00, 9.0000000e00],
  ])
  value, grad = e3x.ops.evaluate_derivatives(
      lambda x: e3x.nn.functions.chebyshev._chebyshev(x, num=4), x, max_order=1
  )
  assert jnp.allclose(value, expected_value, atol=1e-5)
  assert jnp.allclose(grad, expected_grad, atol=1e-5)


@pytest.mark.parametrize('num', [1, 4, 1024])
def test__chebyshev_has_nan_safe_derivatives(num: int) -> None:
  finfo = jnp.finfo(jnp.float32)
  x = jnp.asarray(
      [-1.0, -(1.0 - finfo.epsneg), 0.0, 1.0 - finfo.epsneg, 1.0],
      dtype=jnp.float32,
  )
  for y in e3x.ops.evaluate_derivatives(
      lambda x: e3x.nn.functions.chebyshev._chebyshev(x, num=num),
      x,
      max_order=4,
  ):
    assert jnp.all(jnp.isfinite(y))


def test__chebyshev_raises_with_invalid_num() -> None:
  with pytest.raises(ValueError, match='num must be greater or equal to 1'):
    e3x.nn.functions.chebyshev._chebyshev(0, num=0)


def test_basic_chebyshev() -> None:
  x = jnp.linspace(-0.1, 1.6, 5)
  expected = jnp.asarray([
      [1.0, -1.0, 1.0],
      [1.0, -0.5666666, -0.35777786],
      [1.0, 0.0, -1.0],
      [1.0, 0.5666667, -0.35777768],
      [1.0, 1.0, 1.0],
  ])
  assert jnp.allclose(
      e3x.nn.basic_chebyshev(x, num=3, limit=1.5), expected, atol=1e-5
  )


@pytest.mark.parametrize(
    'kind, use_reciprocal_weighting, expected',
    [
        (
            'shifted',
            False,
            jnp.array([
                [1.0, 1.0, 1.0],
                [1.0, -0.42857146, -0.632653],
                [1.0, -0.6666667, -0.11111108],
                [1.0, -0.76470596, 0.1695504],
                [1.0, -0.8181818, 0.33884305],
            ]),
        ),
        (
            'shifted',
            True,
            jnp.array([
                [1.0, 1.0, 1.0],
                [0.2857143, -0.122449, -0.18075801],
                [0.16666667, -0.11111112, -0.01851851],
                [0.11764706, -0.08996541, 0.01994711],
                [0.09090909, -0.07438017, 0.03080392],
            ]),
        ),
        (
            'damped',
            False,
            jnp.array([
                [1.0, 1.0, 1.0],
                [1.0, -0.26566797, -0.85884106],
                [1.0, -0.60269517, -0.27351704],
                [1.0, -0.7334809, 0.07598842],
                [1.0, -0.80000913, 0.28002912],
            ]),
        ),
        (
            'damped',
            True,
            jnp.array([
                [1.0, 1.0, 1.0],
                [0.36716598, -0.09754425, -0.3153372],
                [0.19865242, -0.11972685, -0.05433482],
                [0.1332596, -0.09774336, 0.01012619],
                [0.09999546, -0.07999728, 0.02800164],
            ]),
        ),
        (
            'cuspless',
            False,
            jnp.array([
                [1.0, 1.0, 1.0],
                [1.0, -0.2254322, -0.89836067],
                [1.0, -0.6005382, -0.27870774],
                [1.0, -0.733353, 0.07561328],
                [1.0, -0.80000097, 0.28000304],
            ]),
        ),
        (
            'cuspless',
            True,
            jnp.array([
                [1.0, 1.0, 1.0],
                [0.38728392, -0.08730627, -0.34792066],
                [0.19973086, -0.11994601, -0.05566654],
                [0.1333235, -0.09777319, 0.01008103],
                [0.09999954, -0.07999973, 0.02800018],
            ]),
        ),
    ],
)
def test_reciprocal_chebyshev(
    kind: str, use_reciprocal_weighting: bool, expected: Float[Array, '5 3']
) -> None:
  x = jnp.linspace(0.0, 10.0, 5)
  assert jnp.allclose(
      e3x.nn.reciprocal_chebyshev(
          x,
          num=3,
          kind=kind,
          use_reciprocal_weighting=use_reciprocal_weighting,
      ),
      expected,
      atol=1e-5,
  )


@pytest.mark.parametrize(
    'cuspless, use_exponential_weighting, expected',
    [
        (
            False,
            False,
            jnp.array([
                [1.0, 1.0, 1.0],
                [1.0, -0.8358299, 0.39722332],
                [1.0, -0.9865241, 0.9464596],
                [1.0, -0.99889386, 0.9955779],
                [1.0, -0.9999092, 0.9996369],
            ]),
        ),
        (
            False,
            True,
            jnp.array([
                [1.0, 1.0, 1.0],
                [0.082085, -0.0686091, 0.03260608],
                [0.00673795, -0.00664715, 0.00637719],
                [0.00055308, -0.00055247, 0.00055064],
                [0.0000454, -0.0000454, 0.00004538],
            ]),
        ),
        (
            True,
            False,
            jnp.array([
                [1.0, 1.0, 1.0],
                [1.0, -0.5889078, -0.3063752],
                [1.0, -0.9636147, 0.85710657],
                [1.0, -0.9969948, 0.98799723],
                [1.0, -0.9997532, 0.9990128],
            ]),
        ),
        (
            True,
            True,
            jnp.array([
                [1.0, 1.0, 1.0],
                [0.20554611, -0.12104771, -0.06297423],
                [0.01819265, -0.0175307, 0.01559304],
                [0.00150261, -0.00149809, 0.00148457],
                [0.0001234, -0.00012337, 0.00012328],
            ]),
        ),
    ],
)
def test_exponential_chebyshev(
    cuspless: bool,
    use_exponential_weighting: bool,
    expected: Float[Array, '5 8'],
) -> None:
  x = jnp.linspace(0.0, 10.0, 5)
  assert jnp.allclose(
      e3x.nn.exponential_chebyshev(
          x,
          num=3,
          cuspless=cuspless,
          use_exponential_weighting=use_exponential_weighting,
      ),
      expected,
      atol=1e-5,
  )
