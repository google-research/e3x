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


def test__gaussian() -> None:
  x = jnp.linspace(0, 1, 11)
  expected_value = jnp.asarray([
      [1.0000000e00, 3.8953209e-01, 2.3023585e-02, 2.0648536e-04],
      [9.1864747e-01, 6.3003719e-01, 6.5564685e-02, 1.0352867e-03],
      [7.1218950e-01, 8.5997713e-01, 1.5756679e-01, 4.3805540e-03],
      [4.6595076e-01, 9.9061620e-01, 3.1956363e-01, 1.5642131e-02],
      [2.5726593e-01, 9.6298993e-01, 5.4695004e-01, 4.7136799e-02],
      [1.1987326e-01, 7.9001588e-01, 7.9001582e-01, 1.1987326e-01],
      [4.7136799e-02, 5.4695004e-01, 9.6298993e-01, 2.5726599e-01],
      [1.5642131e-02, 3.1956366e-01, 9.9061620e-01, 4.6595076e-01],
      [4.3805540e-03, 1.5756682e-01, 8.5997719e-01, 7.1218956e-01],
      [1.0352852e-03, 6.5564655e-02, 6.3003719e-01, 9.1864753e-01],
      [2.0648536e-04, 2.3023602e-02, 3.8953215e-01, 1.0000000e00],
  ])
  expected_grad = jnp.asarray([
      [0.0, 2.2035263, 0.26048213, 0.00350417],
      [-1.5589963, 2.49482, 0.6305127, 0.01581246],
      [-2.417251, 1.9459062, 1.2478653, 0.05947237],
      [-2.372234, 0.5603771, 1.9884973, 0.18581903],
      [-1.7463789, -1.0894986, 2.4752135, 0.4799628],
      [-1.0171583, -2.234502, 2.2345023, 1.0171583],
      [-0.4799628, -2.4752135, 1.0894986, 1.7463791],
      [-0.18581903, -1.9884973, -0.5603766, 2.372234],
      [-0.05947237, -1.2478654, -1.9459062, 2.417251],
      [-0.01581243, -0.6305125, -2.49482, 1.558996],
      [-0.00350417, -0.26048228, -2.2035263, -0.0],
  ])
  value, grad = e3x.ops.evaluate_derivatives(
      lambda x: e3x.nn.functions.gaussian._gaussian(x, num=4), x, max_order=1
  )
  assert jnp.allclose(value, expected_value, atol=1e-5)
  assert jnp.allclose(grad, expected_grad, atol=1e-5)


@pytest.mark.parametrize('num', [1, 4, 1024])
def test__gaussian_has_nan_safe_derivatives(num: int) -> None:
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
      lambda x: e3x.nn.functions.gaussian._gaussian(x, num=num), x, max_order=4
  ):
    assert jnp.all(jnp.isfinite(y))


def test__gaussian_raises_with_invalid_num() -> None:
  with pytest.raises(ValueError, match='num must be greater or equal to 1'):
    e3x.nn.functions.gaussian._gaussian(0, num=0)


def test_basic_gaussian() -> None:
  x = jnp.linspace(-1.0, 2.0, 5)
  expected = jnp.asarray([
      [0.11334437, 0.00127087, 0.00000123],
      [0.87277037, 0.11334437, 0.00127087],
      [0.58022976, 0.8727704, 0.11334443],
      [0.03330429, 0.5802298, 0.87277037],
      [0.00016504, 0.03330427, 0.5802297],
  ])
  assert jnp.allclose(
      e3x.nn.basic_gaussian(x, num=3, limit=1.5), expected, atol=1e-5
  )


@pytest.mark.parametrize(
    'kind, use_reciprocal_weighting, expected',
    [
        (
            'shifted',
            False,
            jnp.array([
                [1.0, 0.29383266, 0.00745418],
                [0.08212775, 0.7985536, 0.67037594],
                [0.03330429, 0.5802298, 0.87277037],
                [0.02205789, 0.4886053, 0.9344418],
                [0.01744428, 0.44048873, 0.9603212],
            ]),
        ),
        (
            'shifted',
            True,
            jnp.array([
                [1.0, 0.29383266, 0.00745418],
                [0.02346507, 0.22815818, 0.191536],
                [0.00555072, 0.09670497, 0.14546174],
                [0.00259505, 0.05748298, 0.10993433],
                [0.00158584, 0.04004443, 0.08730193],
            ]),
        ),
        (
            'damped',
            False,
            jnp.array([
                [1.0, 0.29383266, 0.00745418],
                [0.14058569, 0.9171888, 0.5166262],
                [0.04302737, 0.64090234, 0.8242122],
                [0.0252153, 0.5174173, 0.91668016],
                [0.01890717, 0.45664245, 0.9521951],
            ]),
        ),
        (
            'damped',
            True,
            jnp.array([
                [1.0, 0.29383266, 0.00745418],
                [0.05161828, 0.33676052, 0.18968756],
                [0.00854749, 0.1273168, 0.16373174],
                [0.00336018, 0.06895082, 0.12215643],
                [0.00189063, 0.04566217, 0.09521519],
            ]),
        ),
        (
            'cuspless',
            False,
            jnp.array([
                [1.0, 0.29383266, 0.00745418],
                [0.15894775, 0.9396563, 0.47960508],
                [0.04339301, 0.6429428, 0.8224791],
                [0.02522899, 0.51753616, 0.9166037],
                [0.01890786, 0.45664975, 0.9521913],
            ]),
        ),
        (
            'cuspless',
            True,
            jnp.array([
                [1.0, 0.29383266, 0.00745418],
                [0.06155791, 0.36391377, 0.18574333],
                [0.00866692, 0.12841551, 0.16427447],
                [0.00336362, 0.06899974, 0.12220482],
                [0.00189078, 0.04566476, 0.09521869],
            ]),
        ),
    ],
)
def test_reciprocal_gaussian(
    kind: str, use_reciprocal_weighting: bool, expected: Float[Array, '5 3']
) -> None:
  x = jnp.linspace(0.0, 10.0, 5)
  assert jnp.allclose(
      e3x.nn.reciprocal_gaussian(
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
                [1.0, 0.29383266, 0.00745418],
                [0.01611954, 0.42501912, 0.9675298],
                [0.00796113, 0.3036261, 0.9997776],
                [0.00749468, 0.2946294, 0.9999985],
                [0.0074575, 0.29389805, 1.0],
            ]),
        ),
        (
            False,
            True,
            jnp.array([
                [1.0, 0.29383266, 0.00745418],
                [0.00132317, 0.03488769, 0.07941968],
                [0.00005364, 0.00204582, 0.00673645],
                [0.00000415, 0.00016295, 0.00055308],
                [0.00000034, 0.00001334, 0.0000454],
            ]),
        ),
        (
            True,
            False,
            jnp.array([
                [1.0, 0.29383266, 0.00745418],
                [0.0454099, 0.65392894, 0.8130375],
                [0.00889425, 0.32070267, 0.9983799],
                [0.00756466, 0.29600036, 0.9999889],
                [0.0074632, 0.29401028, 0.99999994],
            ]),
        ),
        (
            True,
            True,
            jnp.array([
                [1.0, 0.29383266, 0.00745418],
                [0.00933383, 0.13441256, 0.1671167],
                [0.00016181, 0.00583443, 0.01816317],
                [0.00001137, 0.00044477, 0.00150259],
                [0.00000092, 0.00003628, 0.0001234],
            ]),
        ),
    ],
)
def test_exponential_gaussian(
    cuspless: bool,
    use_exponential_weighting: bool,
    expected: Float[Array, '5 8'],
) -> None:
  x = jnp.linspace(0.0, 10.0, 5)
  assert jnp.allclose(
      e3x.nn.exponential_gaussian(
          x,
          num=3,
          cuspless=cuspless,
          use_exponential_weighting=use_exponential_weighting,
      ),
      expected,
      atol=1e-5,
  )
