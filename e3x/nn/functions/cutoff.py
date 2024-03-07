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

r"""Cutoff functions.

Functions that gradually go from :math:`1` (at :math:`x=0`) to :math:`0` (at
:math:`x=x_{\rm cutoff}`).
"""

from typing import Union
import jax.numpy as jnp
import jaxtyping

Array = jaxtyping.Array
Float = jaxtyping.Float


def smooth_cutoff(
    x: Float[Array, '...'],
    cutoff: Union[Float[Array, ''], float] = 1.0,
) -> Float[Array, '...']:
  r"""Smooth cutoff function.

  Computes the function

  .. math::
    \mathrm{smooth\_cutoff}(x) = \begin{cases}
      \exp\left(1-\frac{1}{1-\left(\frac{x}{c}\right)^2}\right),
      & x < c\\
      0, & x \ge c
    \end{cases}

  which is zero beyond  :math:`c` = ``cutoff`` and is infinitely differentiable.

  .. jupyter-execute::
    :hide-code:

    import numpy as np, matplotlib.pyplot as plt
    import matplotlib_inline.backend_inline as inl
    import e3x
    inl.set_matplotlib_formats('pdf', 'svg')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    x = np.linspace(0, 2, num=1001)
    y1 = e3x.nn.smooth_cutoff(x, cutoff=1.0)
    y2 = e3x.nn.smooth_cutoff(x, cutoff=2.0)
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{smooth\_cutoff}(x)$')
    plt.plot(x, y1, lw=3, ls='-',  label='cutoff = 1.0')
    plt.plot(x, y2, lw=3, ls='--', label='cutoff = 2.0')
    plt.legend(); plt.grid()

  Args:
    x: Input array.
    cutoff: Cutoff value (must be larger than 0).

  Returns:
    The function value.

  Raises:
    ValueError: If ``cutoff`` is smaller or equal to 0.
  """
  if cutoff <= 0.0:
    raise ValueError(f'cutoff must be larger than 0, received {cutoff}')
  if cutoff != 1.0:  # Rescale x if necessary.
    x = x / cutoff
  inside_cutoff = x < 1
  safe_x = jnp.where(inside_cutoff, x, 0)
  return jnp.where(inside_cutoff, jnp.exp(1 - 1 / (1 - safe_x * safe_x)), 0)


def cosine_cutoff(
    x: Float[Array, '...'],
    cutoff: Union[Float[Array, ''], float] = 1.0,
) -> Float[Array, '...']:
  r"""Cosine cutoff function.

  Computes the function

  .. math::
    \mathrm{cosine\_cutoff}(x) = \begin{cases}
      \frac{1}{2}\cos\left(\pi\frac{x}{c}\right)+\frac{1}{2},
      & x < c\\
      0, & x \ge c
    \end{cases}

  which is zero beyond  :math:`c` = ``cutoff``.

  .. jupyter-execute::
    :hide-code:

    import numpy as np, matplotlib.pyplot as plt
    import matplotlib_inline.backend_inline as inl
    import e3x
    inl.set_matplotlib_formats('pdf', 'svg')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    x = np.linspace(0, 2, num=1001)
    y1 = e3x.nn.cosine_cutoff(x, cutoff=1.0)
    y2 = e3x.nn.cosine_cutoff(x, cutoff=2.0)
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{cosine\_cutoff}(x)$')
    plt.plot(x, y1, lw=3, ls='-',  label='cutoff = 1.0')
    plt.plot(x, y2, lw=3, ls='--', label='cutoff = 2.0')
    plt.legend(); plt.grid()

  Args:
    x: Input array.
    cutoff: Cutoff value (must be larger than 0).

  Returns:
    The function value.

  Raises:
    ValueError: If ``cutoff`` is smaller or equal to 0.
  """
  if cutoff <= 0.0:
    raise ValueError(f'cutoff must be larger than 0, received {cutoff}')
  if cutoff != 1.0:  # Rescale x if necessary.
    x = x / cutoff
  return jnp.where(x < 1, 0.5 * jnp.cos(jnp.pi * x) + 0.5, 0)
