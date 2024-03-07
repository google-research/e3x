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

"""Functions based on trigonometric functions."""

from typing import Union
from e3x.nn.functions import mappings
import jax
import jax.numpy as jnp
import jaxtyping


Array = jaxtyping.Array
Float = jaxtyping.Float


def sinc(
    x: Float[Array, '...'],
    num: int,
    limit: Union[Float[Array, ''], float] = 1.0,
) -> Float[Array, '... num']:
  r"""Sinc basis functions.

  Computes the basis functions

  .. math::
    \mathrm{sinc}_k(x) = \begin{cases}
      \frac{\mathrm{sin}((k+1)\pi x)}{\pi x}, & x \le l \\
      0, & x > l
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
    y = e3x.nn.sinc(x, num=K, limit=l)
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{sinc}_k(x)$')
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
  x = jnp.expand_dims(x / limit, axis=-1)
  return jnp.where(x <= 1.0, jnp.sinc(jnp.arange(1, num + 1) * x), 0)


def _fourier(
    x: Float[Array, '...'],
    num: int,
) -> Float[Array, '... num']:
  """Helper function to evaluate the first num Fourier basis functions.

  Note: Assumes x has been scaled already.

  Args:
    x: Input array.
    num: Number of basis functions.

  Returns:
    The first num Fourier basis functions evaluated at x.
  """
  if num < 1:
    raise ValueError(f'num must be greater or equal to 1, received {num}')
  with jax.ensure_compile_time_eval():
    frequency = jnp.pi * jnp.arange(0, num)
  return jnp.cos(frequency * jnp.expand_dims(x, axis=-1))


def basic_fourier(
    x: Float[Array, '...'],
    num: int,
    limit: Union[Float[Array, ''], float] = 1.0,
) -> Float[Array, '... num']:
  r"""Fourier basis functions.

  Computes the basis functions

  .. math::
    \mathrm{fourier}_k(x) = \mathrm{cos}\left(k\cdot\frac{\pi}{l}\cdot x\right)

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
    y = e3x.nn.basic_fourier(x, num=K, limit=l)
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{fourier}_k(x)$')
    for k in range(K):
      plt.plot(x, y[:,k], lw=3, label=r'$k$'+f'={k}')
    plt.legend(); plt.grid()

  Args:
    x: Input array.
    num: Number of basis functions :math:`K`.
    limit: Basis functions most expressive between 0 and ``limit``, see
      definition above.

  Returns:
    Value of all basis functions for all values in ``x``. The output shape
    follows the input, with an additional dimension of size ``num`` appended.
  """
  return _fourier(x / limit, num=num)


def reciprocal_fourier(
    x: Float[Array, '...'],
    num: int,
    kind: mappings.ReciprocalMapping = 'shifted',
    use_reciprocal_weighting: bool = False,
) -> Float[Array, '... num']:
  r"""Reciprocal Fourier basis functions.

  Computes the basis functions (see
  :func:`basic_fourier <e3x.nn.functions.trigonometric.basic_fourier>` and
  :func:`reciprocal_mapping <e3x.nn.functions.mappings.reciprocal_mapping>`)

  .. math::
    \mathrm{reciprocal\_fourier}_k(x) =
      \mathrm{fourier}_k(1-\mathrm{reciprocal\_mapping}(x))

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
    y = e3x.nn.reciprocal_fourier(x, num=K, kind='shifted',
    use_reciprocal_weighting=False)
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{reciprocal\_fourier}_k(x)$')
    for k in range(K):
      plt.plot(x, y[:,k], lw=3, label=r'$k$'+f'={k}')
    plt.legend(); plt.grid()

  Plot for :math:`K = 5` (``kind = 'shifted'``, ``use_reciprocal_weighting =
  True``):

  .. jupyter-execute::
    :hide-code:

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    y = e3x.nn.reciprocal_fourier(x, num=K, kind='shifted',
    use_reciprocal_weighting=True)
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{reciprocal\_fourier}_k(x)$')
    for k in range(K):
      plt.plot(x, y[:,k], lw=3, label=r'$k$'+f'={k}')
    plt.legend(); plt.grid()

  Plot for :math:`K = 5` (``kind = 'damped'``, ``use_reciprocal_weighting =
  False``):

  .. jupyter-execute::
    :hide-code:

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    y = e3x.nn.reciprocal_fourier(x, num=K, kind='damped',
    use_reciprocal_weighting=False)
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{reciprocal\_fourier}_k(x)$')
    for k in range(K):
      plt.plot(x, y[:,k], lw=3, label=r'$k$'+f'={k}')
    plt.legend(); plt.grid()

  Plot for :math:`K = 5` (``kind = 'damped'``, ``use_reciprocal_weighting =
  True``):

  .. jupyter-execute::
    :hide-code:

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    y = e3x.nn.reciprocal_fourier(x, num=K, kind='damped',
    use_reciprocal_weighting=True)
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{reciprocal\_fourier}_k(x)$')
    for k in range(K):
      plt.plot(x, y[:,k], lw=3, label=r'$k$'+f'={k}')
    plt.legend(); plt.grid()

  Plot for :math:`K = 5` (``kind = 'cuspless'``, ``use_reciprocal_weighting =
  False``):

  .. jupyter-execute::
    :hide-code:

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    y = e3x.nn.reciprocal_fourier(x, num=K, kind='cuspless',
    use_reciprocal_weighting=False)
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{reciprocal\_fourier}_k(x)$')
    for k in range(K):
      plt.plot(x, y[:,k], lw=3, label=r'$k$'+f'={k}')
    plt.legend(); plt.grid()

  Plot for :math:`K = 5` (``kind = 'cuspless'``, ``use_reciprocal_weighting =
  True``):

  .. jupyter-execute::
    :hide-code:

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    y = e3x.nn.reciprocal_fourier(x, num=K, kind='cuspless',
    use_reciprocal_weighting=True)
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{reciprocal\_fourier}_k(x)$')
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
  fourier = _fourier(1 - mapping, num=num)
  if use_reciprocal_weighting:
    fourier *= jnp.expand_dims(mapping, axis=-1)
  return fourier


def exponential_fourier(
    x: Float[Array, '...'],
    num: int,
    gamma: Union[Float[Array, ''], float] = 1.0,
    cuspless: bool = False,
    use_exponential_weighting: bool = False,
) -> Float[Array, '... num']:
  r"""Exponential Fourier basis functions.

  Computes the basis functions (see
  :func:`basic_fourier <e3x.nn.functions.trigonometric.basic_fourier>` and
  :func:`exponential_mapping <e3x.nn.functions.mappings.exponential_mapping>`)

  .. math::
    \mathrm{exponential\_fourier}_k(x) =
      \mathrm{fourier}_k(1-\mathrm{exponential\_mapping}(x))

  or (if ``use_exponential_weighting = True``)

  .. math::
    \mathrm{exponential\_fourier}_k(x) = \mathrm{exponential\_mapping}(x)
      \cdot \mathrm{fourier}_k(1-\mathrm{exponential\_mapping}(x))

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
    y = e3x.nn.exponential_fourier(x, num=K, gamma=1, cuspless=False,
    use_exponential_weighting=False)
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{exponential\_fourier}_k(x)$')
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
    y = e3x.nn.exponential_fourier(x, num=K, gamma=1, cuspless=False,
    use_exponential_weighting=True)
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{exponential\_fourier}_k(x)$')
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
    y = e3x.nn.exponential_fourier(x, num=K, gamma=1, cuspless=True,
    use_exponential_weighting=False)
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{exponential\_fourier}_k(x)$')
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
    y = e3x.nn.exponential_fourier(x, num=K, gamma=1, cuspless=True,
    use_exponential_weighting=True)
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{exponential\_fourier}_k(x)$')
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
  fourier = _fourier(1 - mapping, num=num)
  if use_exponential_weighting:
    fourier *= jnp.expand_dims(mapping, axis=-1)
  return fourier
