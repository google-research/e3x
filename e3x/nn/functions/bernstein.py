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

"""Functions based on Bernstein polynomials."""

import functools
from typing import Any, Tuple, Union
from e3x.nn.functions import mappings
import jax
import jax.numpy as jnp
import jaxtyping

Array = jaxtyping.Array
Float = jaxtyping.Float


@functools.partial(jax.custom_jvp, nondiff_argnums=(1,))
def _bernstein(
    x: Float[Array, '...'],
    num: int,
) -> Float[Array, '... num']:
  """Helper function to evaluate the first num Bernstein polynomials.

  Note: Bernstein polynomials are only defined for x in the interval [0, 1].

  Args:
    x: Input array.
    num: Number of Bernstein polynomials (maximum degree is num-1).

  Returns:
    The first num Bernstein polynomials evaluated at x.
  """
  x = jnp.expand_dims(x, axis=-1)
  if num < 1:
    raise ValueError(f'num must be greater or equal to 1, received {num}')
  elif num == 1:
    return jnp.ones_like(x)
  with jax.ensure_compile_time_eval():
    n = num - 1
    v = jnp.arange(n + 1)
    binomln = -jax.scipy.special.betaln(1 + n - v, 1 + v) - jnp.log(n + 1)
  eps = jnp.finfo(x.dtype).epsneg
  mask0 = x < eps
  mask1 = x > 1 - eps
  mask = jnp.logical_or(mask0, mask1)
  safe_x = jnp.where(mask, 0.5, x)
  y = jnp.where(
      mask,
      0,
      jnp.exp(binomln + v * jnp.log(safe_x) + (n - v) * jnp.log1p(-safe_x)),
  )
  y = jnp.where(jnp.logical_and(mask0, v == 0), 1, y)  # Entries for x = 0.
  y = jnp.where(jnp.logical_and(mask1, v == n), 1, y)  # Entries for x = 1.
  return y


@_bernstein.defjvp
def _bernstein_jvp_impl(
    num: int, primals: Any, tangents: Any
) -> Tuple[Any, Any]:
  """JVP implementation for the Bernstein polynomials."""
  (x,) = primals
  (x_dot,) = tangents
  x_dot = jnp.expand_dims(x_dot, axis=-1)
  primal_out = _bernstein(x, num)
  if num > 1:
    lower = _bernstein(x, num - 1)
    lower = jnp.concatenate([lower, jnp.zeros_like(lower[..., 0:1])], axis=-1)
    shift = jnp.roll(lower, shift=1, axis=-1)
    shift = shift.at[..., 0].set(0)
    tangent_out = x_dot * (num - 1) * (shift - lower)
  else:
    tangent_out = 0 * x_dot
  return primal_out, tangent_out


def basic_bernstein(
    x: Float[Array, '...'],
    num: int,
    limit: Union[Float[Array, ''], float] = 1.0,
) -> Float[Array, '... num']:
  r"""Basic Bernstein polynomial basis functions.

  .. math::
    \mathrm{bernstein}_k(x) = \begin{cases}
    0 & x < 0\\
    \binom{K-1}{k} \left(\frac{x}{l}\right)^{k}
    \left(1-\frac{x}{l}\right)^{K-1-k}, & 0 \le x \le l \\
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
    y = e3x.nn.basic_bernstein(x, num=K, limit=l)
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{bernstein}_k(x)$')
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
  return _bernstein(x / limit, num=num)


def reciprocal_bernstein(
    x: Float[Array, '...'],
    num: int,
    kind: mappings.ReciprocalMapping = 'shifted',
    use_reciprocal_weighting: bool = False,
) -> Float[Array, '... num']:
  r"""Reciprocal Bernstein polynomial basis functions.

  Computes the basis functions (see
  :func:`basic_bernstein <e3x.nn.functions.bernstein.basic_bernstein>` and
  :func:`reciprocal_mapping <e3x.nn.functions.mappings.reciprocal_mapping>`)

  .. math::
    \mathrm{reciprocal\_bernstein}_k(x) =
      \mathrm{bernstein}_k(1-\mathrm{reciprocal\_mapping}(x))

  or (if ``use_reciprocal_weighting = True``)

  .. math::
    \mathrm{reciprocal\_bernstein}_k(x) = \mathrm{reciprocal\_mapping}(x)
      \cot \mathrm{bernstein}_k(1-\mathrm{reciprocal\_mapping}(x))


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
    y = e3x.nn.reciprocal_bernstein(x, num=K, kind='shifted',
    use_reciprocal_weighting=False)
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{reciprocal\_bernstein}_k(x)$')
    for k in range(K):
      plt.plot(x, y[:,k], lw=3, label=r'$k$'+f'={k}')
    plt.legend(); plt.grid()

  Plot for :math:`K = 5` (``kind = 'shifted'``, ``use_reciprocal_weighting =
  True``):

  .. jupyter-execute::
    :hide-code:

    inl.set_matplotlib_formats('pdf', 'svg')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    y = e3x.nn.reciprocal_bernstein(x, num=K, kind='shifted',
    use_reciprocal_weighting=True)
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{reciprocal\_bernstein}_k(x)$')
    for k in range(K):
      plt.plot(x, y[:,k], lw=3, label=r'$k$'+f'={k}')
    plt.legend(); plt.grid()

  Plot for :math:`K = 5` (``kind = 'damped'``, ``use_reciprocal_weighting =
  False``):

  .. jupyter-execute::
    :hide-code:

    inl.set_matplotlib_formats('pdf', 'svg')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    y = e3x.nn.reciprocal_bernstein(x, num=K, kind='damped',
    use_reciprocal_weighting=False)
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{reciprocal\_bernstein}_k(x)$')
    for k in range(K):
      plt.plot(x, y[:,k], lw=3, label=r'$k$'+f'={k}')
    plt.legend(); plt.grid()

  Plot for :math:`K = 5` (``kind = 'damped'``, ``use_reciprocal_weighting =
  True``):

  .. jupyter-execute::
    :hide-code:

    inl.set_matplotlib_formats('pdf', 'svg')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    y = e3x.nn.reciprocal_bernstein(x, num=K, kind='damped',
    use_reciprocal_weighting=True)
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{reciprocal\_bernstein}_k(x)$')
    for k in range(K):
      plt.plot(x, y[:,k], lw=3, label=r'$k$'+f'={k}')
    plt.legend(); plt.grid()

  Plot for :math:`K = 5` (``kind = 'cuspless'``, ``use_reciprocal_weighting =
  False``):

  .. jupyter-execute::
    :hide-code:

    inl.set_matplotlib_formats('pdf', 'svg')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    y = e3x.nn.reciprocal_bernstein(x, num=K, kind='cuspless',
    use_reciprocal_weighting=False)
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{reciprocal\_bernstein}_k(x)$')
    for k in range(K):
      plt.plot(x, y[:,k], lw=3, label=r'$k$'+f'={k}')
    plt.legend(); plt.grid()

  Plot for :math:`K = 5` (``kind = 'cuspless'``, ``use_reciprocal_weighting =
  True``):

  .. jupyter-execute::
    :hide-code:

    inl.set_matplotlib_formats('pdf', 'svg')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    y = e3x.nn.reciprocal_bernstein(x, num=K, kind='cuspless',
    use_reciprocal_weighting=True)
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{reciprocal\_bernstein}_k(x)$')
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
  bernstein = _bernstein(1 - mapping, num=num)
  if use_reciprocal_weighting:
    bernstein *= jnp.expand_dims(mapping, axis=-1)
  return bernstein


def exponential_bernstein(
    x: Float[Array, '...'],
    num: int,
    gamma: Union[Float[Array, ''], float] = 1.0,
    cuspless: bool = False,
    use_exponential_weighting: bool = False,
) -> Float[Array, '... num']:
  r"""Exponential Bernstein polynomial basis functions.

  Computes the basis functions (see
  :func:`basic_bernstein <e3x.nn.functions.bernstein.basic_bernstein>` and
  :func:`exponential_mapping <e3x.nn.functions.mappings.exponential_mapping>`)

  .. math::
    \mathrm{exponential\_bernstein}_k(x) =
      \mathrm{bernstein}_k(1-\mathrm{exponential\_mapping}(x))

  or (if ``use_exponential_weighting = True``)

  .. math::
    \mathrm{exponential\_bernstein}_k(x) = \mathrm{exponential\_mapping}(x)
      \cdot \mathrm{bernstein}_k(1-\mathrm{exponential\_mapping}(x))

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
    y = e3x.nn.exponential_bernstein(x, num=K, gamma=1, cuspless=False,
    use_exponential_weighting=False)
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{exponential\_bernstein}_k(x)$')
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
    y = e3x.nn.exponential_bernstein(x, num=K, gamma=1, cuspless=False,
    use_exponential_weighting=True)
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{exponential\_bernstein}_k(x)$')
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
    y = e3x.nn.exponential_bernstein(x, num=K, gamma=1, cuspless=True,
    use_exponential_weighting=False)
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{exponential\_bernstein}_k(x)$')
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
    y = e3x.nn.exponential_bernstein(x, num=K, gamma=1, cuspless=True,
    use_exponential_weighting=True)
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{exponential\_bernstein}_k(x)$')
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
  bernstein = _bernstein(1 - mapping, num=num)
  if use_exponential_weighting:
    bernstein *= jnp.expand_dims(mapping, axis=-1)
  return bernstein
