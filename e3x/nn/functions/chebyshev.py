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

"""Functions based on Chebyshev polynomials."""

from typing import Union
from e3x.nn.functions import mappings
import jax
import jax.numpy as jnp
import jaxtyping

Array = jaxtyping.Array
Float = jaxtyping.Float


def _chebyshev(
    x: Float[Array, '...'],
    num: int,
) -> Float[Array, '... num']:
  """Helper function to evaluate the first num Chebyshev polynomials.

  Note: This implementation is only correct if x is in the interval [-1, 1].

  Args:
    x: Input array.
    num: Number of Chebyshev polynomials (maximum degree is num-1).

  Returns:
    The first num Chebyshev polynomials evaluated at x.
  """
  x = jnp.expand_dims(x, axis=-1)
  if num < 1:
    raise ValueError(f'num must be greater or equal to 1, received {num}')
  elif num == 1:
    return jnp.ones_like(x)
  with jax.ensure_compile_time_eval():
    n = jnp.arange(num)
    n2 = n * n
  eps = jnp.finfo(x.dtype).epsneg
  one_minus_eps = 1 - eps
  small = x < -one_minus_eps
  large = x > one_minus_eps
  safe_x = jnp.where(jnp.logical_or(small, large), 0, x)
  # Taylor expansions with O(x**3) error for small/large x.
  xm1 = x - 1
  xp1 = x + 1
  limit_small = (-1) ** n * (1 - n2 * xp1 + 1 / 6 * n2 * (n2 - 1) * xp1 * xp1)
  limit_large = 1 + n2 * xm1 + 1 / 6 * n2 * (n2 - 1) * xm1 * xm1
  return jnp.where(
      small,
      limit_small,
      jnp.where(large, limit_large, jnp.cos(n * jnp.arccos(safe_x))),
  )


def basic_chebyshev(
    x: Float[Array, '...'],
    num: int,
    limit: Union[Float[Array, ''], float] = 1.0,
) -> Float[Array, '... num']:
  r"""Basic Chebyshev polynomial basis functions.

  Computes the basis functions

  .. math::
    \mathrm{chebyshev}_k(x) = \begin{cases}
    0 & x < 0 \\
    \cos\left(k \arccos\left(2\frac{x}{l}-1
    \right)\right), & 0 \le x \le l \\
    0 & x > l \,,
    \end{cases}

  where :math:`k=0 \dots K-1` with :math:`K` = ``num`` and
  :math:`l` = ``limit``.

  Plot for :math:`K = 5` and :math:`l = 1`:

  .. jupyter-execute::
    :hide-code:

    import numpy as np, matplotlib.pyplot as plt
    import matplotlib_inline.backend_inline as inl
    import e3x
    inl.set_matplotlib_formats('pdf', 'svg')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    x = np.linspace(0, 1.0, num=1001); K = 5; l = 1.0
    y = e3x.nn.basic_chebyshev(x, num=K, limit=l)
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{chebyshev}_k(x)$')
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

  def _function(x: Float[Array, '...'], num: int) -> Float[Array, '... num']:
    """Small wrapper for _chebyshev which prevents discontinuities."""
    x_1 = jnp.expand_dims(x, axis=-1)
    return jnp.where(
        jnp.abs(x_1) <= 1.0,
        _chebyshev(x, num),
        jnp.sign(x_1) ** jnp.arange(num),
    )

  return _function(2 * x / limit - 1, num=num)


def reciprocal_chebyshev(
    x: Float[Array, '...'],
    num: int,
    kind: mappings.ReciprocalMapping = 'shifted',
    use_reciprocal_weighting: bool = False,
) -> Float[Array, '... num']:
  r"""Reciprocal Chebyshev polynomial basis functions.

  Computes the basis functions (see
  :func:`basic_chebyshev <e3x.nn.functions.chebyshev.basic_chebyshev>` and
  :func:`reciprocal_mapping <e3x.nn.functions.mappings.reciprocal_mapping>`)

  .. math::
    \mathrm{reciprocal\_chebyshev}_k(x) =
      \mathrm{chebyshev}_k(2\cdot\mathrm{reciprocal\_mapping}(x)-1)

  where :math:`k=0 \dots K-1` with :math:`K` = ``num``.

  Plot for :math:`K = 5` (``kind = 'shifted'``, ``use_reciprocal_weighting =
  False``):

  .. jupyter-execute::
    :hide-code:

    import numpy as np, matplotlib.pyplot as plt
    import matplotlib_inline.backend_inline as inl
    import e3x
    inl.set_matplotlib_formats('pdf', 'svg')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    x = np.linspace(0, 10.0, num=1001); K = 5; l = 10.0
    y = e3x.nn.reciprocal_chebyshev(x, num=K, kind='shifted',
    use_reciprocal_weighting=False)
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{reciprocal\_chebyshev}_k(x)$')
    for k in range(K):
      plt.plot(x, y[:,k], lw=3, label=r'$k$'+f'={k}')
    plt.legend(); plt.grid()

  Plot for :math:`K = 5` (``kind = 'shifted'``, ``use_reciprocal_weighting =
  True``):

  .. jupyter-execute::
    :hide-code:

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    y = e3x.nn.reciprocal_chebyshev(x, num=K, kind='shifted',
    use_reciprocal_weighting=True)
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{reciprocal\_chebyshev}_k(x)$')
    for k in range(K):
      plt.plot(x, y[:,k], lw=3, label=r'$k$'+f'={k}')
    plt.legend(); plt.grid()

  Plot for :math:`K = 5` (``kind = 'damped'``, ``use_reciprocal_weighting =
  False``):

  .. jupyter-execute::
    :hide-code:

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    y = e3x.nn.reciprocal_chebyshev(x, num=K, kind='damped',
    use_reciprocal_weighting=False)
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{reciprocal\_chebyshev}_k(x)$')
    for k in range(K):
      plt.plot(x, y[:,k], lw=3, label=r'$k$'+f'={k}')
    plt.legend(); plt.grid()

  Plot for :math:`K = 5` (``kind = 'damped'``, ``use_reciprocal_weighting =
  True``):

  .. jupyter-execute::
    :hide-code:

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    y = e3x.nn.reciprocal_chebyshev(x, num=K, kind='damped',
    use_reciprocal_weighting=True)
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{reciprocal\_chebyshev}_k(x)$')
    for k in range(K):
      plt.plot(x, y[:,k], lw=3, label=r'$k$'+f'={k}')
    plt.legend(); plt.grid()

  Plot for :math:`K = 5` (``kind = 'cuspless'``, ``use_reciprocal_weighting =
  False``):

  .. jupyter-execute::
    :hide-code:

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    y = e3x.nn.reciprocal_chebyshev(x, num=K, kind='cuspless',
    use_reciprocal_weighting=False)
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{reciprocal\_chebyshev}_k(x)$')
    for k in range(K):
      plt.plot(x, y[:,k], lw=3, label=r'$k$'+f'={k}')
    plt.legend(); plt.grid()

  Plot for :math:`K = 5` (``kind = 'cuspless'``, ``use_reciprocal_weighting =
  True``):

  .. jupyter-execute::
    :hide-code:

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    y = e3x.nn.reciprocal_chebyshev(x, num=K, kind='cuspless',
    use_reciprocal_weighting=True)
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{reciprocal\_chebyshev}_k(x)$')
    for k in range(K):
      plt.plot(x, y[:,k], lw=3, label=r'$k$'+f'={k}')
    plt.legend(); plt.grid()

  Args:
    x: Input array.
    num: Number of basis functions :math:`K`.
    kind: Which kind of reciprocal mapping is used.
    use_reciprocal_weighting: If ``True``, the functions are weighted by the
      value of the reciprocal mapping.

  Returns:
    Value of all basis functions for all values in ``x``. The output shape
    follows the input, with an additional dimension of size ``num`` appended.
  """
  mapping = mappings.reciprocal_mapping(x, kind=kind)
  chebyshev = _chebyshev(2 * mapping - 1, num=num)
  if use_reciprocal_weighting:
    chebyshev *= jnp.expand_dims(mapping, axis=-1)
  return chebyshev


def exponential_chebyshev(
    x: Float[Array, '...'],
    num: int,
    gamma: Union[Float[Array, ''], float] = 1.0,
    cuspless: bool = False,
    use_exponential_weighting: bool = False,
) -> Float[Array, '... num']:
  r"""Exponential Chebyshev polynomial basis functions.

  Computes the basis functions (see
  :func:`basic_chebyshev <e3x.nn.functions.chebyshev.basic_chebyshev>` and
  :func:`exponential_mapping <e3x.nn.functions.mappings.exponential_mapping>`)

  .. math::
    \mathrm{exponential\_chebyshev}_k(x) =
      \mathrm{chebyshev}_k(2\cdot\mathrm{exponential\_mapping}(x)-1)

  or (if ``use_exponential_weighting = True``)

  .. math::
    \mathrm{exponential\_chebyshev}_k(x) = \mathrm{exponential\_mapping}(x)
      \cdot \mathrm{chebyshev}_k(2\cdot\mathrm{exponential\_mapping}(x)-1)

  where :math:`k=0 \dots K-1` with :math:`K` = ``num``.

  Plot for :math:`K = 5` and :math:`\gamma = 1` (``cuspless = False``,
  ``use_exponential_weighting = False``):

  .. jupyter-execute::
    :hide-code:

    import numpy as np, matplotlib.pyplot as plt
    import matplotlib_inline.backend_inline as inl
    import e3x
    inl.set_matplotlib_formats('pdf', 'svg')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    x = np.linspace(0, 5.0, num=1001); K = 5; l = 5.0
    y = e3x.nn.exponential_chebyshev(x, num=K, gamma=1, cuspless=False,
    use_exponential_weighting=False)
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{exponential\_chebyshev}_k(x)$')
    for k in range(K):
      plt.plot(x, y[:,k], lw=3, label=r'$k$'+f'={k}')
    plt.legend(); plt.grid()

  Plot for :math:`K = 5` and :math:`\gamma = 1` (``cuspless = False``,
  ``use_exponential_weighting = True``):

  .. jupyter-execute::
    :hide-code:

    inl.set_matplotlib_formats('pdf', 'svg')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    x = np.linspace(0, 5.0, num=1001); K = 5; l = 5.0
    y = e3x.nn.exponential_chebyshev(x, num=K, gamma=1, cuspless=False,
    use_exponential_weighting=True)
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{exponential\_chebyshev}_k(x)$')
    for k in range(K):
      plt.plot(x, y[:,k], lw=3, label=r'$k$'+f'={k}')
    plt.legend(); plt.grid()

  Plot for :math:`K = 5` and :math:`\gamma = 1` (``cuspless = True``,
  ``use_exponential_weighting = False``):

  .. jupyter-execute::
    :hide-code:

    inl.set_matplotlib_formats('pdf', 'svg')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    x = np.linspace(0, 5.0, num=1001); K = 5; l = 5.0
    y = e3x.nn.exponential_chebyshev(x, num=K, gamma=1, cuspless=True,
    use_exponential_weighting=False)
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{exponential\_chebyshev}_k(x)$')
    for k in range(K):
      plt.plot(x, y[:,k], lw=3, label=r'$k$'+f'={k}')
    plt.legend(); plt.grid()

  Plot for :math:`K = 5` and :math:`\gamma = 1` (``cuspless = True``,
  ``use_exponential_weighting = True``):

  .. jupyter-execute::
    :hide-code:

    inl.set_matplotlib_formats('pdf', 'svg')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    x = np.linspace(0, 5.0, num=1001); K = 5; l = 5.0
    y = e3x.nn.exponential_chebyshev(x, num=K, gamma=1, cuspless=True,
    use_exponential_weighting=True)
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{exponential\_chebyshev}_k(x)$')
    for k in range(K):
      plt.plot(x, y[:,k], lw=3, label=r'$k$'+f'={k}')
    plt.legend(); plt.grid()

  Args:
    x: Input array.
    num: Number of basis functions :math:`K`.
    gamma: Exponential decay constant for the exponential mapping.
    cuspless: If ``True``, the returned functions are cuspless.
    use_exponential_weighting: If ``True``, the functions are weighted by the
      value of the exponential mapping.

  Returns:
    Value of all basis functions for all values in ``x``. The output shape
    follows the input, with an additional dimension of size ``num`` appended.
  """
  mapping = mappings.exponential_mapping(x, gamma, cuspless=cuspless)
  chebyshev = _chebyshev(2 * mapping - 1, num=num)
  if use_exponential_weighting:
    chebyshev *= jnp.expand_dims(mapping, axis=-1)
  return chebyshev
