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

r"""Functions for mapping the interval :math:`[0,\infty)` to :math:`[0,1)`."""

from typing import Literal, Union
import jax.numpy as jnp
import jaxtyping

Array = jaxtyping.Array
Float = jaxtyping.Float

_valid_reciprocal_mappings = (
    'shifted',
    'damped',
    'cuspless',
)
ReciprocalMapping = Literal[_valid_reciprocal_mappings]


def reciprocal_mapping(
    x: Float[Array, '...'],
    kind: ReciprocalMapping = 'shifted',
) -> Float[Array, '...']:
  r"""Reciprocal mapping function.

  Computes the function (when ``kind = 'shifted'``)

  .. math::
    \mathrm{reciprocal\_mapping}(x) = \frac{1}{x+1}

  which is :math:`1` for :math:`x = 0` and :math:`0` for :math:`x \to \infty`,
  or (when ``kind = 'damped'``)

  .. math::
    \mathrm{reciprocal\_mapping}(x) = \frac{1-e^{-x}}{x}

  which is :math:`1` for :math:`x = 0` and :math:`\sim \frac{1}{x}` for
  :math:`x \gg 1`, or (when ``kind = 'cuspless'``)

  .. math::
    \mathrm{reciprocal\_mapping}(x) = \frac{1}{x+e^{-x}}

  which is similar to ``kind = 'damped'``, but has no cusp at :math:`x = 0`.

  .. jupyter-execute::
    :hide-code:

    import numpy as np, matplotlib.pyplot as plt
    import matplotlib_inline.backend_inline as inl
    from e3x.nn import reciprocal_mapping
    inl.set_matplotlib_formats('pdf', 'svg')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    x = np.linspace(0, 5, num=1001)
    y1 = reciprocal_mapping(x, 'shifted')
    y2 = reciprocal_mapping(x, 'damped')
    y3 = reciprocal_mapping(x, 'cuspless')
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{reciprocal\_mapping}(x)$')
    plt.plot(x, y1, lw=3, label='shifted');
    plt.plot(x, y2, lw=3, label='damped');
    plt.plot(x, y3, lw=3, label='cuspless');
    plt.legend()
    plt.grid()

  Args:
    x: Input array.
    kind: Which kind of mapping is used.

  Returns:
    The function value.
  """
  # Check that type is a valid value.
  if kind not in _valid_reciprocal_mappings:
    raise ValueError(
        f"kind must be in {_valid_reciprocal_mappings}, received '{kind}'"
    )

  if kind == 'shifted':
    return 1 / (x + 1)
  elif kind == 'damped':
    small = x < jnp.finfo(x.dtype).eps
    safe_x = jnp.where(small, 1, x)
    return jnp.where(small, 1 - x / 2 + x * x / 6, -jnp.expm1(-safe_x) / safe_x)
  elif kind == 'cuspless':
    return 1 / (x + jnp.exp(-x))
  else:  # Protection from potential bugs if other valid values are added.
    assert False, f"Missing implementation of kind '{kind}'!"


def exponential_mapping(
    x: Float[Array, '...'],
    gamma: Union[Float[Array, ''], float] = 1.0,
    cuspless: bool = False,
) -> Float[Array, '...']:
  r"""Exponential mapping function.

  Computes the function (when ``cuspless = False``)

  .. math::
    \mathrm{exponential\_mapping}(x) = \exp\left(-\gamma x\right)\,,

  or (when ``cuspless = True``)

  .. math::
    \mathrm{exponential\_mapping}(x) = \exp\left(-\gamma (x+e^{-x}-1)\right)\,,

  where :math:`\gamma` = ``gamma``.

  Plots for ``cuspless = False``:

  .. jupyter-execute::
    :hide-code:

    import numpy as np, matplotlib.pyplot as plt
    import matplotlib_inline.backend_inline as inl
    from e3x.nn import exponential_mapping
    inl.set_matplotlib_formats('pdf', 'svg')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    x = np.linspace(0, 3, num=1001)
    y1 = exponential_mapping(x, gamma=1.0, cuspless=False)
    y2 = exponential_mapping(x, gamma=2.0, cuspless=False)
    y3 = exponential_mapping(x, gamma=0.5, cuspless=False)
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{exponential\_mapping}(x)$')
    plt.plot(x, y1, lw=3, ls='-',  label='gamma = 1.0')
    plt.plot(x, y2, lw=3, ls='--', label='gamma = 2.0')
    plt.plot(x, y3, lw=3, ls=':',  label='gamma = 0.5')
    plt.legend(); plt.grid()

  Plots for ``cuspless = True``:

  .. jupyter-execute::
    :hide-code:

    inl.set_matplotlib_formats('pdf', 'svg')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    x = np.linspace(0, 3, num=1001)
    y1 = exponential_mapping(x, gamma=1.0, cuspless=True)
    y2 = exponential_mapping(x, gamma=2.0, cuspless=True)
    y3 = exponential_mapping(x, gamma=0.5, cuspless=True)
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{exponential\_mapping}(x)$')
    plt.plot(x, y1, lw=3, ls='-',  label='gamma = 1.0')
    plt.plot(x, y2, lw=3, ls='--', label='gamma = 2.0')
    plt.plot(x, y3, lw=3, ls=':',  label='gamma = 0.5')
    plt.legend(); plt.grid()

  Args:
    x: Input array.
    gamma: Exponential decay constant.
    cuspless: If this is ``True``, a cuspless exponential mapping is returned.

  Returns:
    The function value.
  """
  if cuspless:
    x = x + jnp.exp(-x) - 1
  return jnp.exp(-gamma * x)
