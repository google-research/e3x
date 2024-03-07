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

"""Functions based on Gaussians."""

from typing import Union
from e3x.nn.functions import mappings
import jax
import jax.numpy as jnp
import jaxtyping

Array = jaxtyping.Array
Float = jaxtyping.Float


def _gaussian(x: Float[Array, '...'], num: int) -> Float[Array, '... num']:
  """Helper function for evaluating num evenly spaced Gaussians."""
  if num < 1:
    raise ValueError(f'num must be greater or equal to 1, received {num}')
  with jax.ensure_compile_time_eval():
    mu = jnp.linspace(0, 1, num)
    gamma = jnp.sqrt(2 * num) * (num - 1)  # sqrt(2*num)/d with d = 1/(num-1)
  x = jnp.expand_dims(x, axis=-1)
  return jnp.exp(-gamma * (x - mu) ** 2)


def basic_gaussian(
    x: Float[Array, '...'],
    num: int,
    limit: Union[Float[Array, ''], float] = 1.0,
) -> Float[Array, '... num']:
  r"""Basic Gaussian basis functions.

  Computes the basis functions

  .. math::
    \mathrm{gaussian}_k(x) = \exp\left(-\frac{\sqrt{2K}(K-1)}{l}
      \left(x-\frac{kl}{K-1}\right)^2\right)

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
    x = np.linspace(0, 1.75, num=1001); K = 5; l = 1.0
    y = e3x.nn.basic_gaussian(x, num=K, limit=l)
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{gaussian}_k(x)$')
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
  return _gaussian(x / limit, num=num)


def reciprocal_gaussian(
    x: Float[Array, '...'],
    num: int,
    kind: mappings.ReciprocalMapping = 'shifted',
    use_reciprocal_weighting: bool = False,
) -> Float[Array, '... num']:
  r"""Reciprocal Gaussian basis functions.

  Computes the basis functions (see
  :func:`basic_gaussian <e3x.nn.functions.gaussian.basic_gaussian>` and
  :func:`reciprocal_mapping <e3x.nn.functions.mappings.reciprocal_mapping>`)

  .. math::
    \mathrm{reciprocal\_gaussian}_k(x) =
      \mathrm{gaussian}_k(1-\mathrm{reciprocal\_mapping}(x))


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
    y = e3x.nn.reciprocal_gaussian(x, num=K, kind='shifted',
    use_reciprocal_weighting=False)
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{reciprocal\_gaussian}_k(x)$')
    for k in range(K):
      plt.plot(x, y[:,k], lw=3, label=r'$k$'+f'={k}')
    plt.legend(); plt.grid()

  Plot for :math:`K = 5` (``kind = 'shifted'``, ``use_reciprocal_weighting =
  False``):

  .. jupyter-execute::
    :hide-code:

    inl.set_matplotlib_formats('pdf', 'svg')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    y = e3x.nn.reciprocal_gaussian(x, num=K, kind='shifted',
    use_reciprocal_weighting=True)
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{reciprocal\_gaussian}_k(x)$')
    for k in range(K):
      plt.plot(x, y[:,k], lw=3, label=r'$k$'+f'={k}')
    plt.legend(); plt.grid()


  Plot for :math:`K = 5` (``kind = 'damped'``, ``use_reciprocal_weighting =
  False``):

  .. jupyter-execute::
    :hide-code:

    inl.set_matplotlib_formats('pdf', 'svg')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    y = e3x.nn.reciprocal_gaussian(x, num=K, kind='damped',
    use_reciprocal_weighting=False)
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{reciprocal\_gaussian}_k(x)$')
    for k in range(K):
      plt.plot(x, y[:,k], lw=3, label=r'$k$'+f'={k}')
    plt.legend(); plt.grid()

  Plot for :math:`K = 5` (``kind = 'damped'``, ``use_reciprocal_weighting =
  True``):

  .. jupyter-execute::
    :hide-code:

    inl.set_matplotlib_formats('pdf', 'svg')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    y = e3x.nn.reciprocal_gaussian(x, num=K, kind='damped',
    use_reciprocal_weighting=True)
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{reciprocal\_gaussian}_k(x)$')
    for k in range(K):
      plt.plot(x, y[:,k], lw=3, label=r'$k$'+f'={k}')
    plt.legend(); plt.grid()

  Plot for :math:`K = 5` (``kind = 'cuspless'``, ``use_reciprocal_weighting =
  False``):

  .. jupyter-execute::
    :hide-code:

    inl.set_matplotlib_formats('pdf', 'svg')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    y = e3x.nn.reciprocal_gaussian(x, num=K, kind='cuspless',
    use_reciprocal_weighting=False)
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{reciprocal\_gaussian}_k(x)$')
    for k in range(K):
      plt.plot(x, y[:,k], lw=3, label=r'$k$'+f'={k}')
    plt.legend(); plt.grid()

  Plot for :math:`K = 5` (``kind = 'cuspless'``, ``use_reciprocal_weighting =
  True``):

  .. jupyter-execute::
    :hide-code:

    inl.set_matplotlib_formats('pdf', 'svg')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    y = e3x.nn.reciprocal_gaussian(x, num=K, kind='cuspless',
    use_reciprocal_weighting=True)
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{reciprocal\_gaussian}_k(x)$')
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
  gaussian = _gaussian(1 - mapping, num=num)
  if use_reciprocal_weighting:
    gaussian *= jnp.expand_dims(mapping, axis=-1)
  return gaussian


def exponential_gaussian(
    x: Float[Array, '...'],
    num: int,
    gamma: Union[Float[Array, ''], float] = 1.0,
    cuspless: bool = False,
    use_exponential_weighting: bool = False,
) -> Float[Array, '... num']:
  r"""Exponential Gaussian basis functions.

  Computes the basis functions (see
  :func:`basic_gaussian <e3x.nn.functions.gaussian.basic_gaussian>` and
  :func:`exponential_mapping <e3x.nn.functions.mappings.exponential_mapping>`)

  .. math::
    \mathrm{exponential\_gaussian}_k(x) =
      \mathrm{gaussian}_k(1-\mathrm{exponential\_mapping}(x))

  or (if ``use_exponential_weighting = True``)

  .. math::
    \mathrm{exponential\_gaussian}_k(x) = \mathrm{exponential\_mapping}(x)
      \cdot \mathrm{gaussian}_k(1-\mathrm{exponential\_mapping}(x))


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
    y = e3x.nn.exponential_gaussian(x, num=K, gamma=1, cuspless=False,
    use_exponential_weighting=False)
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{exponential\_gaussian}_k(x)$')
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
    y = e3x.nn.exponential_gaussian(x, num=K, gamma=1, cuspless=False,
    use_exponential_weighting=True)
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{exponential\_gaussian}_k(x)$')
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
    y = e3x.nn.exponential_gaussian(x, num=K, gamma=1, cuspless=True,
    use_exponential_weighting=False)
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{exponential\_gaussian}_k(x)$')
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
    y = e3x.nn.exponential_gaussian(x, num=K, gamma=1, cuspless=True,
    use_exponential_weighting=True)
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{exponential\_gaussian}_k(x)$')
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
  gaussian = _gaussian(1 - mapping, num=num)
  if use_exponential_weighting:
    gaussian *= jnp.expand_dims(mapping, axis=-1)
  return gaussian
