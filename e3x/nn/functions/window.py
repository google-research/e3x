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

"""Window functions."""

from typing import Union
from e3x.nn.functions import smooth_switch
import jax
import jax.numpy as jnp
import jaxtyping

Array = jaxtyping.Array
Float = jaxtyping.Float


def rectangular_window(
    x: Float[Array, '...'],
    num: int,
    limit: Union[Float[Array, ''], float] = 1.0,
) -> Float[Array, '... num']:
  r"""Rectangular window basis functions.

  Computes the basis functions

  .. math::
    \mathrm{rectangular\_window}_k(x) = \begin{cases}
      0, & x < \frac{k l}{K}\\
      1, & \frac{k l}{K+1} \le x \le \frac{(k+1) l}{K} \\
      0, & x > \frac{(k+1) l}{K}
    \end{cases}

  where :math:`k=0 \dots K-1` with :math:`K` = ``num`` and
  :math:`l` = ``limit``. Plot for :math:`K = 5` and :math:`l = 1`:

  .. jupyter-execute::
    :hide-code:

    import numpy as np, matplotlib.pyplot as plt
    import matplotlib_inline.backend_inline as inl
    from e3x.nn import rectangular_window
    inl.set_matplotlib_formats('pdf', 'svg')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    x = np.linspace(0, 1.0, num=1001); K = 5; l = 1.0
    y = rectangular_window(x, num=K, limit=l)
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{rectangular\_window}_k(x)$')
    for k in range(K):
      plt.plot(x, y[:,k], lw=3, label=r'$k$'+f'={k}')
    plt.legend(); plt.grid()

  Args:
    x: Input array.
    num: Number of basis functions :math:`K`.
    limit: Basis functions are distributed between 0 and ``limit``.

  Returns:
    Value of all basis functions for all values in ``x``. The output shape
    follows the input, with an additional dimension of size ``num`` appended.
  """
  with jax.ensure_compile_time_eval():
    index = jnp.arange(num)
    bound = jnp.linspace(0.0, limit, num=num + 1)
    lower = bound[index]
    upper = bound[index + 1]
  x_1 = jnp.expand_dims(x, -1)
  return jnp.where(
      x_1 < lower,
      jnp.zeros_like(x_1),
      jnp.where(x_1 > upper, jnp.zeros_like(x_1), jnp.ones_like(x_1)),
  )


def triangular_window(
    x: Float[Array, '...'],
    num: int,
    limit: Union[Float[Array, ''], float] = 1.0,
) -> Float[Array, '... num']:
  r"""Triangular window basis functions.

  Computes the basis functions

  .. math::
    \mathrm{triangular\_window}_k(x) = \max\left(
      \min\left(\frac{K}{l}x - k - 1, \frac{K}{l}x + k + 1\right),0\right)

  where :math:`k=0 \dots K-1` with :math:`K` = ``num`` and
  :math:`l` = ``limit``. Plot for :math:`K = 5` and :math:`l = 1`:

  .. jupyter-execute::
    :hide-code:

    import numpy as np, matplotlib.pyplot as plt
    import matplotlib_inline.backend_inline as inl
    from e3x.nn import triangular_window
    inl.set_matplotlib_formats('pdf', 'svg')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    x = np.linspace(0, 1.0, num=1001); K = 5; l = 1.0
    y = triangular_window(x, num=K, limit=l)
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{triangular\_window}_k(x)$')
    for k in range(K):
      plt.plot(x, y[:,k], lw=3, label=r'$k$'+f'={k}')
    plt.legend(); plt.grid()

  Args:
    x: Input array.
    num: Number of basis functions :math:`K`.
    limit: Basis functions are distributed between 0 and ``limit``.

  Returns:
    Value of all basis functions for all values in ``x``. The output shape
    follows the input, with an additional dimension of size ``num`` appended.
  """
  with jax.ensure_compile_time_eval():
    width = limit / num
    center = jnp.linspace(0.0, limit, num=num + 1)[:-1]
    lower = center - width
    upper = center + width
  x_1 = jnp.expand_dims(x, -1)
  return jnp.maximum(
      jnp.minimum((x_1 - lower) / width, -(x_1 - upper) / width), 0
  )


def smooth_window(
    x: Float[Array, '...'],
    num: int,
    limit: Union[Float[Array, ''], float] = 1.0,
) -> Float[Array, '... num']:
  r"""Smooth window basis functions.

  Computes the basis functions (see
  :func:`smooth_switch <e3x.nn.functions.smooth_switch>`)

  .. math::
    \mathrm{smooth\_window}_k(x) =
    \mathrm{smooth\_switch}\left(\frac{K}{l}x - \frac{kl}{K}\right) -
    \mathrm{smooth\_switch}\left(1 -\frac{K}{l}x + \frac{kl}{K}\right)

  where :math:`k=0 \dots K-1` with :math:`K` = ``num`` and
  :math:`l` = ``limit``. Plot for :math:`K = 5` and :math:`l = 1`:

  .. jupyter-execute::
    :hide-code:

    import numpy as np, matplotlib.pyplot as plt
    import matplotlib_inline.backend_inline as inl
    from e3x.nn import smooth_window
    inl.set_matplotlib_formats('pdf', 'svg')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    x = np.linspace(0, 1.0, num=1001); K = 5; l = 1.0
    y = smooth_window(x, num=K, limit=l)
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{smooth\_window}_k(x)$')
    for k in range(K):
      plt.plot(x, y[:,k], lw=3, label=r'$k$'+f'={k}')
    plt.legend(); plt.grid()

  Args:
    x: Input array.
    num: Number of basis functions :math:`K`.
    limit: Basis functions are distributed between 0 and ``limit``.

  Returns:
    Value of all basis functions for all values in ``x``. The output shape
    follows the input, with an additional dimension of size ``num`` appended.
  """
  with jax.ensure_compile_time_eval():
    center = jnp.linspace(0.0, limit, num=num + 1)[:-1]
  x_1 = num / limit * (jnp.expand_dims(x, -1) - center)
  return smooth_switch(x_1 + 1) - smooth_switch(x_1)
