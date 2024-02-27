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


def test_sinc() -> None:
  x = jnp.linspace(0.0, 2.0, 11)
  expected = jnp.asarray([
      [
          1.00000000e00,
          1.00000000e00,
          1.00000000e00,
          1.00000000e00,
          1.00000000e00,
          1.00000000e00,
          1.00000000e00,
          1.00000000e00,
      ],
      [
          9.71012175e-01,
          8.87063742e-01,
          7.56826639e-01,
          5.93561471e-01,
          4.13496643e-01,
          2.33872190e-01,
          7.09073991e-02,
          -6.20441660e-02,
      ],
      [
          8.87063742e-01,
          5.93561471e-01,
          2.33872190e-01,
          -6.20441660e-02,
          -2.06748337e-01,
          -1.89206615e-01,
          -6.93579093e-02,
          6.06883466e-02,
      ],
      [
          7.56826699e-01,
          2.33872294e-01,
          -1.55914947e-01,
          -1.89206675e-01,
          2.78275341e-08,
          1.26137808e-01,
          6.68206811e-02,
          -5.84680997e-02,
      ],
      [
          5.93561471e-01,
          -6.20441660e-02,
          -1.89206615e-01,
          6.06883466e-02,
          1.03374146e-01,
          -5.84681742e-02,
          -6.33616224e-02,
          5.54415472e-02,
      ],
      [
          4.13496643e-01,
          -2.06748337e-01,
          2.78275341e-08,
          1.03374146e-01,
          -8.26993659e-02,
          2.78275341e-08,
          5.90708852e-02,
          -5.16870953e-02,
      ],
      [
          2.33872294e-01,
          -1.89206675e-01,
          1.26137808e-01,
          -5.84680997e-02,
          2.78275341e-08,
          3.89786400e-02,
          -5.40590622e-02,
          4.73016798e-02,
      ],
      [
          7.09074885e-02,
          -6.93579838e-02,
          6.68206811e-02,
          -6.33616820e-02,
          5.90709560e-02,
          -5.40590622e-02,
          4.84540015e-02,
          -4.23972532e-02,
      ],
      [
          0.00000000e00,
          0.00000000e00,
          0.00000000e00,
          0.00000000e00,
          0.00000000e00,
          0.00000000e00,
          0.00000000e00,
          0.00000000e00,
      ],
      [
          0.00000000e00,
          0.00000000e00,
          0.00000000e00,
          0.00000000e00,
          0.00000000e00,
          0.00000000e00,
          0.00000000e00,
          0.00000000e00,
      ],
      [
          0.00000000e00,
          0.00000000e00,
          0.00000000e00,
          0.00000000e00,
          0.00000000e00,
          0.00000000e00,
          0.00000000e00,
          0.00000000e00,
      ],
  ])
  assert jnp.allclose(e3x.nn.sinc(x, num=8, limit=1.5), expected, atol=1e-5)


def test__fourier() -> None:
  x = jnp.linspace(0, 1, 11)
  expected_value = jnp.asarray([
      [1.0000000e00, 1.0000000e00, 1.0000000e00, 1.0000000e00],
      [1.0000000e00, 9.5105654e-01, 8.0901700e-01, 5.8778524e-01],
      [1.0000000e00, 8.0901700e-01, 3.0901697e-01, -3.0901703e-01],
      [1.0000000e00, 5.8778518e-01, -3.0901715e-01, -9.5105660e-01],
      [1.0000000e00, 3.0901697e-01, -8.0901706e-01, -8.0901694e-01],
      [1.0000000e00, -4.3711388e-08, -1.0000000e00, 1.1924881e-08],
      [1.0000000e00, -3.0901715e-01, -8.0901676e-01, 8.0901724e-01],
      [1.0000000e00, -5.8778518e-01, -3.0901709e-01, 9.5105660e-01],
      [1.0000000e00, -8.0901706e-01, 3.0901712e-01, 3.0901679e-01],
      [1.0000000e00, -9.5105660e-01, 8.0901724e-01, -5.8778572e-01],
      [1.0000000e00, -1.0000000e00, 1.0000000e00, -1.0000000e00],
  ])
  expected_grad = jnp.asarray(
      [
          [-0.0000000e00, -0.0000000e00, -0.0000000e00, -0.0000000e00],
          [-0.0000000e00, -9.7080559e-01, -3.6931636e00, -7.6248055e00],
          [-0.0000000e00, -1.8465818e00, -5.9756646e00, -8.9634962e00],
          [-0.0000000e00, -2.5416021e00, -5.9756641e00, -2.9124148e00],
          [-0.0000000e00, -2.9878323e00, -3.6931634e00, 5.5397468e00],
          [-0.0000000e00, -3.1415927e00, 5.4929353e-07, 9.4247780e00],
          [-0.0000000e00, -2.9878321e00, 3.6931655e00, 5.5397425e00],
          [-0.0000000e00, -2.5416019e00, 5.9756641e00, -2.9124150e00],
          [-0.0000000e00, -1.8465817e00, 5.9756641e00, -8.9634972e00],
          [-0.0000000e00, -9.7080493e-01, 3.6931617e00, -7.6248021e00],
          [-0.0000000e00, 2.7464677e-07, -1.0985871e-06, 2.2477870e-07],
      ],
  )
  value, grad = e3x.ops.evaluate_derivatives(
      lambda x: e3x.nn.functions.trigonometric._fourier(x, num=4),
      x,
      max_order=1,
  )
  assert jnp.allclose(value, expected_value, atol=1e-5)
  assert jnp.allclose(grad, expected_grad, atol=1e-5)


@pytest.mark.parametrize('num', [1, 4, 1024])
def test__fourier_has_nan_safe_derivatives(num: int) -> None:
  finfo = jnp.finfo(jnp.float32)
  x = jnp.asarray(
      [
          0.0,
          finfo.tiny,
          finfo.eps,
          0.5,
          1.0 - finfo.epsneg,
          1.0,
          1.0 + finfo.eps,
      ],
      dtype=jnp.float32,
  )
  for y in e3x.ops.evaluate_derivatives(
      lambda x: e3x.nn.functions.trigonometric._fourier(x, num=num),
      x,
      max_order=4,
  ):
    assert jnp.all(jnp.isfinite(y))


def test__fourier_raises_with_invalid_num() -> None:
  with pytest.raises(ValueError, match='num must be greater or equal to 1'):
    e3x.nn.functions.trigonometric._fourier(0, num=0)


def test_basic_fourier() -> None:
  x = jnp.linspace(0.0, 1.5, 5)
  expected = jnp.asarray([
      [1.0, 1.0, 1.0],
      [1.0, 0.70710677, 0.0],
      [1.0, 0.0, -1.0],
      [1.0, -0.70710677, 0.0],
      [1.0, -1.0, 1.0],
  ])
  assert jnp.allclose(
      e3x.nn.basic_fourier(x, num=3, limit=1.5), expected, atol=1e-5
  )


@pytest.mark.parametrize(
    'kind, use_reciprocal_weighting, expected',
    [
        (
            'shifted',
            False,
            jnp.array([
                [1.0, 1.0, 1.0],
                [1.0, -0.62349, -0.22252055],
                [1.0, -0.8660254, 0.4999999],
                [1.0, -0.9324723, 0.7390091],
                [1.0, -0.959493, 0.84125346],
            ]),
        ),
        (
            'shifted',
            True,
            jnp.array([
                [1.0, 1.0, 1.0],
                [0.2857143, -0.17814, -0.0635773],
                [0.16666667, -0.14433756, 0.08333332],
                [0.11764706, -0.10970262, 0.08694225],
                [0.09090909, -0.08722664, 0.07647759],
            ]),
        ),
        (
            'damped',
            False,
            jnp.array([
                [1.0, 1.0, 1.0],
                [1.0, -0.40530315, -0.6714587],
                [1.0, -0.8114982, 0.31705874],
                [1.0, -0.9136397, 0.6694751],
                [1.0, -0.951061, 0.80903405],
            ]),
        ),
        (
            'damped',
            True,
            jnp.array([
                [1.0, 1.0, 1.0],
                [0.36716598, -0.14881353, -0.2465368],
                [0.19865242, -0.16120608, 0.06298449],
                [0.1332596, -0.12175126, 0.08921397],
                [0.09999546, -0.09510178, 0.08089973],
            ]),
        ),
        (
            'cuspless',
            False,
            jnp.array([
                [1.0, 1.0, 1.0],
                [1.0, -0.34675387, -0.7595235],
                [1.0, -0.8095137, 0.31062478],
                [1.0, -0.91355807, 0.66917676],
                [1.0, -0.95105696, 0.8090186],
            ]),
        ),
        (
            'cuspless',
            True,
            jnp.array([
                [1.0, 1.0, 1.0],
                [0.38728392, -0.1342922, -0.29415125],
                [0.19973086, -0.16168487, 0.06204135],
                [0.1333235, -0.12179876, 0.08921699],
                [0.09999954, -0.09510526, 0.08090149],
            ]),
        ),
    ],
)
def test_reciprocal_fourier(
    kind: str, use_reciprocal_weighting: bool, expected: Float[Array, '5 3']
) -> None:
  x = jnp.linspace(0.0, 10.0, 5)
  assert jnp.allclose(
      e3x.nn.reciprocal_fourier(
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
                [1.0, -0.9669334, 0.8699206],
                [1.0, -0.99977595, 0.99910396],
                [1.0, -0.9999985, 0.999994],
                [1.0, -1.0, 0.99999994],
            ]),
        ),
        (
            False,
            True,
            jnp.array([
                [1.0, 1.0, 1.0],
                [0.082085, -0.07937073, 0.07140743],
                [0.00673795, -0.00673644, 0.00673191],
                [0.00055308, -0.00055308, 0.00055308],
                [0.0000454, -0.0000454, 0.0000454],
            ]),
        ),
        (
            True,
            False,
            jnp.array([
                [1.0, 1.0, 1.0],
                [1.0, -0.7986534, 0.27569452],
                [1.0, -0.9983672, 0.993474],
                [1.0, -0.99998885, 0.9999554],
                [1.0, -0.99999994, 0.9999997],
            ]),
        ),
        (
            True,
            True,
            jnp.array([
                [1.0, 1.0, 1.0],
                [0.20554611, -0.1641601, 0.05666794],
                [0.01819265, -0.01816294, 0.01807392],
                [0.00150261, -0.00150259, 0.00150254],
                [0.0001234, -0.0001234, 0.0001234],
            ]),
        ),
    ],
)
def test_exponential_fourier(
    cuspless: bool,
    use_exponential_weighting: bool,
    expected: Float[Array, '5 8'],
) -> None:
  x = jnp.linspace(0.0, 10.0, 5)
  assert jnp.allclose(
      e3x.nn.exponential_fourier(
          x,
          num=3,
          cuspless=cuspless,
          use_exponential_weighting=use_exponential_weighting,
      ),
      expected,
      atol=1e-5,
  )
