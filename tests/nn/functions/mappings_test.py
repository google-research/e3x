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
    'kind, expected',
    [
        (
            'shifted',
            jnp.asarray([
                1.00000000,
                0.50000000,
                0.33333334,
                0.25000000,
                0.20000000,
                0.16666667,
                0.14285715,
                0.12500000,
                0.11111111,
                0.10000000,
                0.09090909,
            ]),
        ),
        (
            'damped',
            jnp.asarray([
                1.00000000,
                0.63212055,
                0.43233237,
                0.31673765,
                0.24542110,
                0.19865242,
                0.16625354,
                0.14272687,
                0.12495807,
                0.11109740,
                0.09999546,
            ]),
        ),
        (
            'cuspless',
            jnp.asarray([
                1.00000000,
                0.73105860,
                0.46831053,
                0.32789174,
                0.24886048,
                0.19973086,
                0.16659784,
                0.14283854,
                0.12499475,
                0.11110959,
                0.09999954,
            ]),
        ),
    ],
)
def test_reciprocal(
    kind: e3x.nn.functions.mappings.ReciprocalMapping,
    expected: Float[Array, '...'],
) -> None:
  x = jnp.linspace(0.0, 10.0, 11)
  assert jnp.allclose(
      e3x.nn.reciprocal_mapping(x, kind=kind), expected, atol=1e-5
  )


@pytest.mark.parametrize(
    'kind', e3x.nn.functions.mappings._valid_reciprocal_mappings
)
def test_reciprocal_mapping_has_nan_safe_derivatives(
    kind: e3x.nn.functions.mappings.ReciprocalMapping,
) -> None:
  finfo = jnp.finfo(jnp.float32)
  x = jnp.asarray([0.0, finfo.tiny, finfo.eps], dtype=jnp.float32)
  for y in e3x.ops.evaluate_derivatives(
      lambda x: e3x.nn.reciprocal_mapping(x, kind=kind), x, max_order=4
  ):
    assert jnp.all(jnp.isfinite(y))


@pytest.mark.parametrize(
    'gamma, cuspless, expected',
    [
        (
            1.0,
            True,
            jnp.asarray([
                1.0000000e00,
                6.9220060e-01,
                3.2131439e-01,
                1.2876232e-01,
                4.8883490e-02,
                1.8192649e-02,
                6.7212670e-03,
                2.4764934e-03,
                9.1157592e-04,
                3.3542135e-04,
                1.2340416e-04,
            ]),
        ),
        (
            1.0,
            False,
            jnp.asarray([
                1.0000000e00,
                3.6787945e-01,
                1.3533528e-01,
                4.9787067e-02,
                1.8315639e-02,
                6.7379470e-03,
                2.4787523e-03,
                9.1188197e-04,
                3.3546262e-04,
                1.2340980e-04,
                4.5399931e-05,
            ]),
        ),
        (
            0.5,
            True,
            jnp.asarray([
                1.00000000,
                0.83198595,
                0.56684600,
                0.35883468,
                0.22109611,
                0.13488013,
                0.08198334,
                0.04976438,
                0.03019232,
                0.01831451,
                0.01110874,
            ]),
        ),
        (
            0.5,
            False,
            jnp.asarray([
                1.00000000,
                0.60653067,
                0.36787945,
                0.22313017,
                0.13533528,
                0.08208500,
                0.04978707,
                0.03019738,
                0.01831564,
                0.01110900,
                0.00673795,
            ]),
        ),
    ],
)
def test_exponential_mapping(
    gamma: float,
    cuspless: bool,
    expected: Float[Array, '...'],
) -> None:
  x = jnp.linspace(0.0, 10.0, 11)
  assert jnp.allclose(
      e3x.nn.exponential_mapping(x, gamma=gamma, cuspless=cuspless),
      expected,
      atol=1e-5,
  )
