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

r"""Damping functions.

Functions that go to :math:`0` for small :math:`x` and are close to :math:`1`
for large :math:`x` (assuming :math:`x>0`).
"""

from typing import Union
import jax.numpy as jnp
import jaxtyping

Array = jaxtyping.Array
Float = jaxtyping.Float


def smooth_damping(
    x: Float[Array, '...'], gamma: Union[Float[Array, ''], float] = 1.0
) -> Float[Array, '...']:
  r"""Smooth damping function.

  Computes the function

  .. math::
    \mathrm{smooth\_damping}(x) =
      \exp\left(-\frac{1}{\exp(\gamma x) - 1} \right)

  where :math:`\gamma` = ``gamma``. This function (and all its derivatives) are
  :math:`0` at :math:`x=0`. Further, the function quickly approaches :math:`1`
  for :math:`x > 0`.

  .. jupyter-execute::
    :hide-code:

    import numpy as np, matplotlib.pyplot as plt
    import matplotlib_inline.backend_inline as inl
    from e3x.nn import smooth_damping
    inl.set_matplotlib_formats('pdf', 'svg')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    x = np.linspace(0, 2, num=1001)
    y1 = smooth_damping(x, gamma=1.0)
    y2 = smooth_damping(x, gamma=5.0)
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{smooth\_zero\_damping}(x)$')
    plt.plot(x, y1, lw=3, ls='-',  label='gamma = 1.0')
    plt.plot(x, y2, lw=3, ls='--', label='gamma = 5.0')
    plt.legend(); plt.grid()

  Args:
    x: Input array.
    gamma: Exponential decay constant.

  Returns:
    The function value.
  """
  exp = jnp.exp(-gamma * x)
  div = 1 - exp
  mask = div * div > 0
  return jnp.where(mask, jnp.exp(-exp / jnp.where(mask, div, 1)), 0)
