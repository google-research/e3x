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


def test__bernstein() -> None:
  x = jnp.linspace(0, 1, 11)
  expected_value = jnp.asarray([
      [1.0000000e00, 0.0000000e00, 0.0000000e00, 0.0000000e00],
      [7.2899973e-01, 2.4300018e-01, 2.7000017e-02, 9.9999947e-04],
      [5.1199985e-01, 3.8400027e-01, 9.6000060e-02, 7.9999957e-03],
      [3.4299982e-01, 4.4100028e-01, 1.8900014e-01, 2.6999986e-02],
      [2.1599992e-01, 4.3200034e-01, 2.8800023e-01, 6.3999996e-02],
      [1.2499994e-01, 3.7500027e-01, 3.7500027e-01, 1.2499994e-01],
      [6.3999966e-02, 2.8800020e-01, 4.3200034e-01, 2.1599996e-01],
      [2.6999986e-02, 1.8900014e-01, 4.4100028e-01, 3.4299982e-01],
      [7.9999957e-03, 9.6000060e-02, 3.8400030e-01, 5.1199985e-01],
      [9.9999900e-04, 2.7000006e-02, 2.4300012e-01, 7.2899985e-01],
      [0.0000000e00, 0.0000000e00, 0.0000000e00, 1.0000000e00],
  ])
  expected_grad = jnp.asarray([
      [-3.0, 3.0, 0.0, 0.0],
      [-2.4300003, 1.8899996, 0.51000047, 0.03],
      [-1.9200003, 0.95999956, 0.8400008, 0.12],
      [-1.4700001, 0.20999908, 0.990001, 0.26999998],
      [-1.08, -0.36000118, 0.9600011, 0.48000008],
      [-0.7500001, -0.7500012, 0.7500012, 0.7500001],
      [-0.48000002, -0.96000123, 0.360001, 1.0800002],
      [-0.26999998, -0.990001, -0.20999908, 1.4700001],
      [-0.12, -0.84000087, -0.95999944, 1.9200003],
      [-0.02999998, -0.51000035, -1.8900003, 2.4300005],
      [0.0, 0.0, -3.0, 3.0],
  ])
  value, grad = e3x.ops.evaluate_derivatives(
      lambda x: e3x.nn.functions.bernstein._bernstein(x, num=4), x, max_order=1
  )
  assert jnp.allclose(value, expected_value, atol=1e-5)
  assert jnp.allclose(grad, expected_grad, atol=1e-5)


@pytest.mark.parametrize('num', [1, 4, 1024])
def test__bernstein_has_nan_safe_derivatives(num: int) -> None:
  finfo = jnp.finfo(jnp.float32)
  x = jnp.asarray(
      [0.0, finfo.tiny, finfo.epsneg, 1.0 - finfo.epsneg, 1.0],
      dtype=jnp.float32,
  )
  for y in e3x.ops.evaluate_derivatives(
      lambda x: e3x.nn.functions.bernstein._bernstein(x, num=num),
      x,
      max_order=4,
  ):
    assert jnp.all(jnp.isfinite(y))


def test__bernstein_raises_with_invalid_num() -> None:
  with pytest.raises(ValueError, match='num must be greater or equal to 1'):
    e3x.nn.functions.bernstein._bernstein(0, num=0)


def test_basic_bernstein() -> None:
  x = jnp.linspace(-1.0, 2.0, 5)
  expected = jnp.asarray([
      [1.0, 0.0, 0.0],
      [1.0, 0.0, 0.0],
      [0.44444448, 0.4444448, 0.1111111],
      [0.02777779, 0.27777803, 0.6944445],
      [0.0, 0.0, 1.0],
  ])
  assert jnp.allclose(
      e3x.nn.basic_bernstein(x, num=3, limit=1.5), expected, atol=1e-5
  )


@pytest.mark.parametrize(
    'kind, use_reciprocal_weighting, expected',
    [
        (
            'shifted',
            False,
            jnp.array([
                [1.0, 0.0, 0.0],
                [0.08163264, 0.4081636, 0.5102042],
                [0.02777779, 0.27777803, 0.6944445],
                [0.01384083, 0.20761262, 0.7785468],
                [0.00826447, 0.16528946, 0.8264463],
            ]),
        ),
        (
            'shifted',
            True,
            jnp.array([
                [1.0, 0.0, 0.0],
                [0.02332361, 0.11661818, 0.14577264],
                [0.00462963, 0.04629634, 0.11574075],
                [0.00162833, 0.02442501, 0.09159374],
                [0.00075132, 0.01502632, 0.07513148],
            ]),
        ),
        (
            'damped',
            False,
            jnp.array([
                [1.0, 0.0, 0.0],
                [0.13481088, 0.46471068, 0.40047896],
                [0.03946277, 0.3183795, 0.6421581],
                [0.01775812, 0.23100315, 0.751239],
                [0.00999909, 0.17999287, 0.8100083],
            ]),
        ),
        (
            'damped',
            True,
            jnp.array([
                [1.0, 0.0, 0.0],
                [0.04949797, 0.17062595, 0.14704224],
                [0.00783938, 0.06324685, 0.12756626],
                [0.00236644, 0.03078339, 0.10010981],
                [0.00099986, 0.01799847, 0.08099715],
            ]),
        ),
        (
            'cuspless',
            False,
            jnp.array([
                [1.0, 0.0, 0.0],
                [0.14998886, 0.47459057, 0.37542105],
                [0.03989244, 0.3196772, 0.64043075],
                [0.01777516, 0.23109691, 0.75112826],
                [0.00999991, 0.17999941, 0.8100009],
            ]),
        ),
        (
            'cuspless',
            True,
            jnp.array([
                [1.0, 0.0, 0.0],
                [0.05808827, 0.1838013, 0.14539453],
                [0.00796775, 0.0638494, 0.12791379],
                [0.00236985, 0.03081065, 0.10014305],
                [0.00099999, 0.01799986, 0.08099972],
            ]),
        ),
    ],
)
def test_reciprocal_bernstein(
    kind: str, use_reciprocal_weighting: bool, expected: Float[Array, '5 3']
) -> None:
  x = jnp.linspace(0.0, 10.0, 5)
  assert jnp.allclose(
      e3x.nn.reciprocal_bernstein(
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
                [1.0, 0.0, 0.0],
                [0.00673795, 0.15069425, 0.84256804],
                [0.0000454, 0.0133851, 0.98656964],
                [0.00000031, 0.00110553, 0.9988943],
                [0.0, 0.00009083, 0.9999093],
            ]),
        ),
        (
            False,
            True,
            jnp.array([
                [1.0, 0.0, 0.0],
                [0.00055308, 0.01236974, 0.0691622],
                [0.00000031, 0.00009019, 0.00664745],
                [0.0, 0.00000061, 0.00055247],
                [0.0, 0.0, 0.0000454],
            ]),
        ),
        (
            True,
            False,
            jnp.array([
                [1.0, 0.0, 0.0],
                [0.04224923, 0.3265941, 0.63115704],
                [0.00033097, 0.03572338, 0.9639458],
                [0.00000226, 0.00300075, 0.9969971],
                [0.00000002, 0.00024673, 0.99975336],
            ]),
        ),
        (
            True,
            True,
            jnp.array([
                [1.0, 0.0, 0.0],
                [0.00868416, 0.06713015, 0.12973188],
                [0.00000602, 0.0006499, 0.01753673],
                [0.0, 0.00000451, 0.0014981],
                [0.0, 0.00000003, 0.00012337],
            ]),
        ),
    ],
)
def test_exponential_bernstein(
    cuspless: bool,
    use_exponential_weighting: bool,
    expected: Float[Array, '5 8'],
) -> None:
  x = jnp.linspace(0.0, 10.0, 5)
  assert jnp.allclose(
      e3x.nn.exponential_bernstein(
          x,
          num=3,
          cuspless=cuspless,
          use_exponential_weighting=use_exponential_weighting,
      ),
      expected,
      atol=1e-5,
  )
