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

r"""Equivariant implementation of common activation functions.

.. _ActivationFunctions:

When working with equivariant features, any nonlinear operation may only be
applied to the scalar channel, else the features lose their equivariance.
Consequently, standard activation functions may not be applied naively to
equivariant features. However, it is possible to apply gated linear activations

.. math::
  \mathrm{gated\_linear}(\mathbf{x}) =
  \mathrm{gate}\left(\mathbf{x}^{(0_+)}\right) \circ \mathbf{x}

where the nonlinear :math:`\mathrm{gate}` function is applied element-wise on
the scalar channel :math:`\mathbf{x}^{(0_+)}` (the element-wise product
':math:`\circ`' implies broadcasting over dimensions). Many common activation
functions can be written as (equivariant) gated linear activations, such that
for scalar inputs, they are equivalent to their non-equivariant counterpart.
See also :ref:`here <Irreps>` for more details on the notation and
:ref:`here <IrrepFeatures>` for features in e3x in general.

.. _ImplementedActivationFunctions:
"""

import math
from typing import Callable, Union
import jax
import jax.numpy as jnp
import jaxtyping

Array = jaxtyping.Array
Float = jaxtyping.Float


def _gated_linear(
    g: Callable[
        [Float[Array, '... 1 1 num_features']],
        Float[Array, '... 1 1 num_features'],
    ],
    x: Union[
        Float[Array, '... 1 (max_degree+1)**2 num_features'],
        Float[Array, '... 2 (max_degree+1)**2 num_features'],
    ],
) -> Union[
    Float[Array, '... 1 (max_degree+1)**2 num_features'],
    Float[Array, '... 2 (max_degree+1)**2 num_features'],
]:
  """General gated linear activation."""
  if len(x.shape) < 3:
    raise ValueError(
        'shape of x must have at least three dimensions, received shape '
        f'{x.shape}'
    )
  return g(x[..., 0:1, 0:1, :]) * x


def relu(
    x: Union[
        Float[Array, '... 1 (max_degree+1)**2 num_features'],
        Float[Array, '... 2 (max_degree+1)**2 num_features'],
    ]
) -> Union[
    Float[Array, '... 1 (max_degree+1)**2 num_features'],
    Float[Array, '... 2 (max_degree+1)**2 num_features'],
]:
  r"""Rectified linear unit activation function.

  Computes the gated linear activation with:

  .. math::
    \mathrm{gate}(x) = \max(\mathrm{sgn}\ x, 0)

  For scalar inputs, this is equivalent to:

  .. math::
    \mathrm{relu}(x) = \max(x, 0)

  .. jupyter-execute::
    :hide-code:

    import numpy as np, matplotlib.pyplot as plt
    import matplotlib_inline.backend_inline as inl
    from e3x.nn import relu
    inl.set_matplotlib_formats('pdf', 'svg')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    x = np.linspace(-3, 3, num=1001)
    y = np.squeeze(relu(x.reshape(1,1,-1)))
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{relu}(x)$')
    plt.plot(x, y, lw=3); plt.grid()

  Args:
    x: Input features to which the nonlinearity is applied.

  Returns:
    The result of applying the nonlinearity to the input features.
  """
  g = lambda x: jnp.maximum(jnp.sign(x), 0)
  return _gated_linear(g, x)


def leaky_relu(
    x: Union[
        Float[Array, '... 1 (max_degree+1)**2 num_features'],
        Float[Array, '... 2 (max_degree+1)**2 num_features'],
    ],
    negative_slope: Union[Float[Array, ''], float] = 1e-2,
) -> Union[
    Float[Array, '... 1 (max_degree+1)**2 num_features'],
    Float[Array, '... 2 (max_degree+1)**2 num_features'],
]:
  r"""Leaky rectified linear unit activation function.

  Computes the gated linear activation with:

  .. math::
    \mathrm{gate}(x) = \begin{cases}
      1, & x \ge 0\\
      \alpha, & x < 0
    \end{cases}

  For scalar inputs, this is equivalent to:

  .. math::
    \mathrm{leaky\_relu}(x) = \begin{cases}
      x, & x \ge 0\\
      \alpha x, & x < 0
    \end{cases}

  where :math:`\alpha` = ``negative_slope``.

  .. jupyter-execute::
    :hide-code:

    import numpy as np, matplotlib.pyplot as plt
    import matplotlib_inline.backend_inline as inl
    from e3x.nn import leaky_relu
    inl.set_matplotlib_formats('pdf', 'svg')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    x = np.linspace(-3, 3, num=1001)
    y1 = np.squeeze(leaky_relu(x.reshape(1,1,-1), negative_slope=0.01))
    y2 = np.squeeze(leaky_relu(x.reshape(1,1,-1), negative_slope=0.5))
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{leaky\_relu}(x)$')
    plt.plot(x, y1, lw=3, ls='-',  label='negative_slope=0.01')
    plt.plot(x, y2, lw=3, ls='--', label='negative_slope=0.5')
    plt.legend(); plt.grid()

  Args:
    x: Input features to which the nonlinearity is applied.
    negative_slope: Value that specifies the negative slope.

  Returns:
    The result of applying the nonlinearity to the input features.
  """
  g = lambda x: jnp.where(x >= 0, 1, negative_slope)
  return _gated_linear(g, x)


def elu(
    x: Union[
        Float[Array, '... 1 (max_degree+1)**2 num_features'],
        Float[Array, '... 2 (max_degree+1)**2 num_features'],
    ],
    alpha: Union[Float[Array, ''], float] = 1.0,
) -> Union[
    Float[Array, '... 1 (max_degree+1)**2 num_features'],
    Float[Array, '... 2 (max_degree+1)**2 num_features'],
]:
  r"""Exponential linear unit activation function.

  Computes the gated linear activation with:

  .. math::
    \mathrm{gate}(x) = \begin{cases}
      1, & x > 0\\
      \frac{\alpha}{x} \left(\exp(x) - 1\right), & x \le 0
    \end{cases}

  For scalar inputs, this is equivalent to:

  .. math::
    \mathrm{elu}(x) = \begin{cases}
      x, & x > 0\\
      \alpha \left(\exp(x) - 1\right), & x \le 0
    \end{cases}

  .. jupyter-execute::
    :hide-code:

    import numpy as np, matplotlib.pyplot as plt
    import matplotlib_inline.backend_inline as inl
    from e3x.nn import elu
    inl.set_matplotlib_formats('pdf', 'svg')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    x = np.linspace(-3, 3, num=1001)
    y1 = np.squeeze(elu(x.reshape(1,1,-1), alpha=1.0))
    y2 = np.squeeze(elu(x.reshape(1,1,-1), alpha=2.0))
    y3 = np.squeeze(elu(x.reshape(1,1,-1), alpha=0.5))
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{elu}(x)$')
    plt.plot(x, y1, lw=3, ls='-',  label=r'$\alpha=1.0$')
    plt.plot(x, y2, lw=3, ls='--', label=r'$\alpha=2.0$')
    plt.plot(x, y3, lw=3, ls=':',  label=r'$\alpha=0.5$')
    plt.legend(); plt.grid()

  Args:
    x: Input features to which the nonlinearity is applied.
    alpha: The :math:`\alpha` value (see definition of :math:`\mathrm{elu}`).

  Returns:
    The result of applying the nonlinearity to the input features.
  """

  def g(
      x: Float[Array, '... 1 1 num_features']
  ) -> Float[Array, '... 1 1 num_features']:
    not_tiny = jnp.abs(x) > jnp.finfo(x.dtype).eps
    expm1_safe_x = jnp.where(x > 0, 0.0, x)
    div_safe_x = jnp.where(not_tiny, x, 1)
    return jnp.where(
        x > 0,
        1,
        jnp.where(
            not_tiny,
            alpha * jnp.expm1(expm1_safe_x) / div_safe_x,
            alpha * (x * x / 6 + x / 2 + 1),
        ),  # + O(x^3) Taylor series around x=0
    )

  return _gated_linear(g, x)


def selu(
    x: Union[
        Float[Array, '... 1 (max_degree+1)**2 num_features'],
        Float[Array, '... 2 (max_degree+1)**2 num_features'],
    ]
) -> Union[
    Float[Array, '... 1 (max_degree+1)**2 num_features'],
    Float[Array, '... 2 (max_degree+1)**2 num_features'],
]:
  r"""Scaled exponential linear unit activation.

  Computes the gated linear activation with:

  .. math::
    \mathrm{gate}(x) = \lambda \begin{cases}
      1, & x > 0\\
      \frac{\alpha}{x} \left(\exp(x) - 1\right), & x \le 0
    \end{cases}

  where :math:`\lambda = 1.0507009873554804934193349852946` and
  :math:`\alpha = 1.6732632423543772848170429916717`.

  For scalar inputs, this is equivalent to:

  .. math::
    \mathrm{selu}(x) = \lambda \begin{cases}
      x, & x > 0\\
      \alpha \left(\exp(x) - 1\right), & x \le 0
    \end{cases}

  For more information, see
  `Self-Normalizing Neural Networks
  <https://papers.nips.cc/paper/6698-self-normalizing-neural-networks.pdf>`_.

  .. jupyter-execute::
    :hide-code:

    import numpy as np, matplotlib.pyplot as plt
    import matplotlib_inline.backend_inline as inl
    from e3x.nn import selu
    inl.set_matplotlib_formats('pdf', 'svg')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    x = np.linspace(-3, 3, num=1001)
    y = np.squeeze(selu(x.reshape(1,1,-1)))
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{selu}(x)$')
    plt.plot(x, y, lw=3); plt.grid()

  Args:
    x: Input features to which the nonlinearity is applied.

  Returns:
    The result of applying the nonlinearity to the input features.
  """
  alpha = 1.6732632423543772848170429916717
  scale = 1.0507009873554804934193349852946
  return scale * elu(x, alpha)


def celu(
    x: Union[
        Float[Array, '... 1 (max_degree+1)**2 num_features'],
        Float[Array, '... 2 (max_degree+1)**2 num_features'],
    ],
    alpha: Union[Float[Array, ''], float] = 1.0,
) -> Union[
    Float[Array, '... 1 (max_degree+1)**2 num_features'],
    Float[Array, '... 2 (max_degree+1)**2 num_features'],
]:
  r"""Continuously-differentiable exponential linear unit activation.

  Computes the gated linear activation with:

  .. math::
    \mathrm{gate}(x) = \begin{cases}
      1, & x > 0\\
      \frac{\alpha}{x} \left(\exp(\frac{x}{\alpha}) - 1\right), & x \le 0
    \end{cases}

  For scalar inputs, this is equivalent to:

  .. math::
    \mathrm{celu}(x) = \begin{cases}
      x, & x > 0\\
      \alpha \left(\exp(\frac{x}{\alpha}) - 1\right), & x \le 0
    \end{cases}

  For more information, see
  `Continuously Differentiable Exponential Linear Units
  <https://arxiv.org/pdf/1704.07483.pdf>`_.

  .. jupyter-execute::
    :hide-code:

    import numpy as np, matplotlib.pyplot as plt
    import matplotlib_inline.backend_inline as inl
    from e3x.nn import celu
    inl.set_matplotlib_formats('pdf', 'svg')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    x = np.linspace(-3, 3, num=1001)
    y1 = np.squeeze(celu(x.reshape(1,1,-1), alpha=1.0))
    y2 = np.squeeze(celu(x.reshape(1,1,-1), alpha=2.0))
    y3 = np.squeeze(celu(x.reshape(1,1,-1), alpha=0.5))
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{celu}(x)$')
    plt.plot(x, y1, lw=3, ls='-',  label=r'$\alpha=1.0$')
    plt.plot(x, y2, lw=3, ls='--', label=r'$\alpha=2.0$')
    plt.plot(x, y3, lw=3, ls=':',  label=r'$\alpha=0.5$')
    plt.legend(); plt.grid()

  Args:
    x: Input features to which the nonlinearity is applied.
    alpha: The :math:`\alpha` value (see definition of :math:`\mathrm{celu}`).

  Returns:
    The result of applying the nonlinearity to the input features.
  """

  def g(
      x: Float[Array, '... 1 1 num_features']
  ) -> Float[Array, '... 1 1 num_features']:
    not_tiny = jnp.abs(x) > jnp.finfo(x.dtype).eps
    expm1_safe_x = jnp.where(x > 0, 0.0, x)
    div_safe_x = jnp.where(not_tiny, x, 1)
    return jnp.where(
        x > 0,
        1,
        jnp.where(
            not_tiny,
            alpha * jnp.expm1(expm1_safe_x / alpha) / div_safe_x,
            x * x / (6 * alpha * alpha) + x / (2 * alpha) + 1,
        ),  # + O(x^3) Taylor series around x=0
    )

  return _gated_linear(g, x)


def silu(
    x: Union[
        Float[Array, '... 1 (max_degree+1)**2 num_features'],
        Float[Array, '... 2 (max_degree+1)**2 num_features'],
    ]
) -> Union[
    Float[Array, '... 1 (max_degree+1)**2 num_features'],
    Float[Array, '... 2 (max_degree+1)**2 num_features'],
]:
  r"""SiLU/Swish activation function.

  Computes the gated linear activation with:

  .. math::
    \mathrm{gate}(x) = \mathrm{sigmoid}(x)

  For scalar inputs, this is equivalent to:

  .. math::
    \mathrm{silu}(x) = x \cdot \mathrm{sigmoid}(x) = \frac{x}{1 + e^{-x}}

  .. jupyter-execute::
    :hide-code:

    import numpy as np, matplotlib.pyplot as plt
    import matplotlib_inline.backend_inline as inl
    from e3x.nn import silu
    inl.set_matplotlib_formats('pdf', 'svg')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    x = np.linspace(-3, 3, num=1001)
    y = np.squeeze(silu(x.reshape(1,1,-1)))
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{silu}(x)$')
    plt.plot(x, y, lw=3); plt.grid()

  Args:
    x: Input features to which the nonlinearity is applied.

  Returns:
    The result of applying the nonlinearity to the input features.
  """
  return _gated_linear(jax.scipy.special.expit, x)


swish = silu


def gelu(
    x: Union[
        Float[Array, '... 1 (max_degree+1)**2 num_features'],
        Float[Array, '... 2 (max_degree+1)**2 num_features'],
    ],
    approximate: bool = True,
) -> Union[
    Float[Array, '... 1 (max_degree+1)**2 num_features'],
    Float[Array, '... 2 (max_degree+1)**2 num_features'],
]:
  r"""Gaussian error linear unit activation function.

  Computes the gated linear activation. If ``approximate=False``, the
  :math:`\mathrm{gate}` function is given by:

  .. math::
    \mathrm{gate}(x) = \frac{1}{2} \left(1 + \mathrm{erf} \left(
      \frac{x}{\sqrt{2}} \right) \right)

  For scalar inputs, this is equivalent to:

  .. math::
    \mathrm{gelu}(x) = \frac{x}{2} \left(1 + \mathrm{erf} \left(
      \frac{x}{\sqrt{2}} \right) \right)

  If ``approximate=True``, the :math:`\mathrm{gate}` function is
  approximated as:

  .. math::
    \mathrm{gate}(x) = \frac{1}{2} \left(1 + \mathrm{tanh} \left(
      \sqrt{\frac{2}{\pi}} \left(x + 0.044715 x^3 \right) \right) \right)

  For scalar inputs, this is equivalent to:

  .. math::
    \mathrm{gelu}(x) = \frac{x}{2} \left(1 + \mathrm{tanh} \left(
      \sqrt{\frac{2}{\pi}} \left(x + 0.044715 x^3 \right) \right) \right)

  For more information, see `Gaussian Error Linear Units (GELUs)
  <https://arxiv.org/abs/1606.08415>`_, section 2.

  .. jupyter-execute::
    :hide-code:

    import numpy as np, matplotlib.pyplot as plt
    import matplotlib_inline.backend_inline as inl
    from e3x.nn import gelu
    inl.set_matplotlib_formats('pdf', 'svg')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    x = np.linspace(-3, 3, num=1001)
    y1 = np.squeeze(gelu(x.reshape(1,1,-1), approximate=True))
    y2 = np.squeeze(gelu(x.reshape(1,1,-1), approximate=False))
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{gelu}(x)$')
    plt.plot(x, y1, lw=3, ls='-',  label='approximate=True')
    plt.plot(x, y2, lw=3, ls='--', label='approximate=False')
    plt.legend(); plt.grid()

  Args:
    x: Input features to which the nonlinearity is applied.
    approximate: Whether to use the approximate or exact formulation.

  Returns:
    The result of applying the nonlinearity to the input features.
  """

  def g(
      x: Float[Array, '... 1 1 num_features']
  ) -> Float[Array, '... 1 1 num_features']:
    if approximate:
      sqrt_2_over_pi = math.sqrt(2 / math.pi)
      cdf = 0.5 * (1.0 + jnp.tanh(sqrt_2_over_pi * (x + 0.044715 * (x**3))))
      return cdf
    else:
      return jnp.array((jax.lax.erf(x / math.sqrt(2)) + 1) / 2, dtype=x.dtype)

  return _gated_linear(g, x)


def mish(
    x: Union[
        Float[Array, '... 1 (max_degree+1)**2 num_features'],
        Float[Array, '... 2 (max_degree+1)**2 num_features'],
    ],
    fast: bool = True,
) -> Union[
    Float[Array, '... 1 (max_degree+1)**2 num_features'],
    Float[Array, '... 2 (max_degree+1)**2 num_features'],
]:
  r"""Mish activation function.

  Computes the gated linear activation with:

  .. math::
    \mathrm{gate}(x) = \mathrm{tanh}(\mathrm{softplus}(x))

  For scalar inputs, this is equivalent to:

  .. math::
    \mathrm{mish}(x) = x \cdot \mathrm{tanh}(\mathrm{softplus}(x))

  For more information, see `Mish: A Self Regularized Non-Monotonic Activation
  Function <https://arxiv.org/pdf/1908.08681>`_.

  .. jupyter-execute::
    :hide-code:

    import numpy as np, matplotlib.pyplot as plt
    import matplotlib_inline.backend_inline as inl
    from e3x.nn import mish
    inl.set_matplotlib_formats('pdf', 'svg')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    x = np.linspace(-3, 3, num=1001)
    y = np.squeeze(mish(x.reshape(1,1,-1)))
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{mish}(x)$')
    plt.plot(x, y, lw=3); plt.grid()

  Args:
    x: Input features to which the nonlinearity is applied.
    fast: Whether to use a faster formulation with higher memory cost.

  Returns:
    The result of applying the nonlinearity to the input features.
  """

  def g(
      x: Float[Array, '... 1 1 num_features']
  ) -> Float[Array, '... 1 1 num_features']:
    if fast:  # Faster, but requires extra memory.
      y = jnp.exp(-x)
      z = 2 * y + 1
      return z / (z + 2 * y * y)
    else:
      return jnp.tanh(jax.nn.softplus(x))

  return _gated_linear(g, x)


def serf(
    x: Union[
        Float[Array, '... 1 (max_degree+1)**2 num_features'],
        Float[Array, '... 2 (max_degree+1)**2 num_features'],
    ]
) -> Union[
    Float[Array, '... 1 (max_degree+1)**2 num_features'],
    Float[Array, '... 2 (max_degree+1)**2 num_features'],
]:
  r"""Serf activation function.

  Computes the gated linear activation with:

  .. math::
    \mathrm{gate}(x) = \mathrm{erf}(\mathrm{softplus}(x))

  For scalar inputs, this is equivalent to:

  .. math::
    \mathrm{serf}(x) = x \cdot \mathrm{erf}(\mathrm{softplus}(x))

  For more information, see `SERF: Towards better training of deep neural
  networks using log-Softplus Error activation Function
  <https://arxiv.org/pdf/2108.09598>`_.

  .. jupyter-execute::
    :hide-code:

    import numpy as np, matplotlib.pyplot as plt
    import matplotlib_inline.backend_inline as inl
    from e3x.nn import serf
    inl.set_matplotlib_formats('pdf', 'svg')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    x = np.linspace(-3, 3, num=1001)
    y = np.squeeze(serf(x.reshape(1,1,-1)))
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{serf}(x)$')
    plt.plot(x, y, lw=3); plt.grid()

  Args:
    x: Input features to which the nonlinearity is applied.

  Returns:
    The result of applying the nonlinearity to the input features.
  """
  g = lambda x: jax.lax.erf(jax.nn.softplus(x))
  return _gated_linear(g, x)


def shifted_softplus(
    x: Union[
        Float[Array, '... 1 (max_degree+1)**2 num_features'],
        Float[Array, '... 2 (max_degree+1)**2 num_features'],
    ]
) -> Union[
    Float[Array, '... 1 (max_degree+1)**2 num_features'],
    Float[Array, '... 2 (max_degree+1)**2 num_features'],
]:
  r"""Shifted-softplus activation function.

  Computes the gated linear activation with:

  .. math::
    \mathrm{gate}(x) = \frac{\log\left(\frac{1}{2}(1 + e^x)\right)}{x}

  For scalar inputs, this is equivalent to:

  .. math::
    \mathrm{shifted\_softplus}(x) = \log\left(\frac{1}{2}(1 + e^x)\right)

  .. jupyter-execute::
    :hide-code:

    import numpy as np, matplotlib.pyplot as plt
    import matplotlib_inline.backend_inline as inl
    from e3x.nn import shifted_softplus
    inl.set_matplotlib_formats('pdf', 'svg')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    x = np.linspace(-3, 3, num=1001)
    y = np.squeeze(shifted_softplus(x.reshape(1,1,-1)))
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{shifted\_softplus}(x)$')
    plt.plot(x, y, lw=3); plt.grid()

  Args:
    x: Input features to which the nonlinearity is applied.

  Returns:
    The result of applying the nonlinearity to the input features.
  """

  def g(
      x: Float[Array, '... 1 1 num_features']
  ) -> Float[Array, '... 1 1 num_features']:
    not_tiny = jnp.abs(x) > jnp.finfo(x.dtype).eps
    safe_x = jnp.where(not_tiny, x, 1)  # division safe x
    return jnp.where(
        not_tiny,
        (jnp.logaddexp(x, 0) - math.log(2)) / safe_x,
        x / 8 + 1 / 2,  # + O(x^3) Taylor series around x=0
    )

  return _gated_linear(g, x)


def hard_tanh(
    x: Union[
        Float[Array, '... 1 (max_degree+1)**2 num_features'],
        Float[Array, '... 2 (max_degree+1)**2 num_features'],
    ]
) -> Union[
    Float[Array, '... 1 (max_degree+1)**2 num_features'],
    Float[Array, '... 2 (max_degree+1)**2 num_features'],
]:
  r"""Hard :math:`\mathrm{tanh}` activation function.

  Computes the gated linear activation with:

  .. math::
    \mathrm{gate}(x) =  \begin{cases}
      -\frac{1}{x}, & x < -1\\
      1, & -1 \le x \le 1\\
      \frac{1}{x}, & 1 < x
    \end{cases}

  For scalar inputs, this is equivalent to:

  .. math::
    \mathrm{hard\_tanh}(x) = \begin{cases}
      -1, & x < -1\\
      x, & -1 \le x \le 1\\
      1, & 1 < x
    \end{cases}

  .. jupyter-execute::
    :hide-code:

    import numpy as np, matplotlib.pyplot as plt
    import matplotlib_inline.backend_inline as inl
    from e3x.nn import hard_tanh
    inl.set_matplotlib_formats('pdf', 'svg')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    x = np.linspace(-3, 3, num=1001)
    y = np.squeeze(hard_tanh(x.reshape(1,1,-1)))
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{hard\_tanh}(x)$')
    plt.plot(x, y, lw=3); plt.grid()

  Args:
    x: Input features to which the nonlinearity is applied.

  Returns:
    The result of applying the nonlinearity to the input features.
  """

  def g(
      x: Float[Array, '... 1 1 num_features']
  ) -> Float[Array, '... 1 1 num_features']:
    safe_x = jnp.where(jnp.abs(x) > 0.1, x, 1)  # division safe x
    return jnp.where(x > 1, 1 / safe_x, jnp.where(x < -1, -1 / safe_x, 1))

  return _gated_linear(g, x)


def soft_sign(
    x: Union[
        Float[Array, '... 1 (max_degree+1)**2 num_features'],
        Float[Array, '... 2 (max_degree+1)**2 num_features'],
    ]
) -> Union[
    Float[Array, '... 1 (max_degree+1)**2 num_features'],
    Float[Array, '... 2 (max_degree+1)**2 num_features'],
]:
  r"""Soft-sign activation function.

  Computes the gated linear activation with:

  .. math::
    \mathrm{gate}(x) =  \frac{1}{|x| + 1}

  For scalar inputs, this is equivalent to:

  .. math::
    \mathrm{soft\_sign}(x) = \frac{x}{|x| + 1}

  .. jupyter-execute::
    :hide-code:

    import numpy as np, matplotlib.pyplot as plt
    import matplotlib_inline.backend_inline as inl
    from e3x.nn import soft_sign
    inl.set_matplotlib_formats('pdf', 'svg')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    x = np.linspace(-3, 3, num=1001)
    y = np.squeeze(soft_sign(x.reshape(1,1,-1)))
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{soft\_sign}(x)$')
    plt.plot(x, y, lw=3); plt.grid()

  Args:
    x: Input features to which the nonlinearity is applied.

  Returns:
    The result of applying the nonlinearity to the input features.
  """
  g = lambda x: 1 / (jnp.abs(x) + 1)
  return _gated_linear(g, x)


def relu6(
    x: Union[
        Float[Array, '... 1 (max_degree+1)**2 num_features'],
        Float[Array, '... 2 (max_degree+1)**2 num_features'],
    ]
) -> Union[
    Float[Array, '... 1 (max_degree+1)**2 num_features'],
    Float[Array, '... 2 (max_degree+1)**2 num_features'],
]:
  r"""Rectified Linear Unit 6 activation function.

  Computes the gated linear activation with:

  .. math::
    \mathrm{gate}(x) =  \min\left(\max(\mathrm{sgn}\ x, 0), \frac{6}{x}\right)

  For scalar inputs, this is equivalent to:

  .. math::
    \mathrm{relu6}(x) = \min(\max(x, 0), 6)

  .. jupyter-execute::
    :hide-code:

    import numpy as np, matplotlib.pyplot as plt
    import matplotlib_inline.backend_inline as inl
    from e3x.nn import relu6
    inl.set_matplotlib_formats('pdf', 'svg')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    x = np.linspace(-9, 9, num=1001)
    y = np.squeeze(relu6(x.reshape(1,1,-1)))
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{relu6}(x)$')
    plt.plot(x, y, lw=3); plt.grid()

  Args:
    x: Input features to which the nonlinearity is applied.

  Returns:
    The result of applying the nonlinearity to the input features.
  """

  def g(
      x: Float[Array, '... 1 1 num_features']
  ) -> Float[Array, '... 1 1 num_features']:
    safe_x = jnp.where(x > 1, x, 1)
    return jnp.minimum(jnp.maximum(jnp.sign(x), 0), 6.0 / safe_x)

  return _gated_linear(g, x)


def hard_silu(
    x: Union[
        Float[Array, '... 1 (max_degree+1)**2 num_features'],
        Float[Array, '... 2 (max_degree+1)**2 num_features'],
    ]
) -> Union[
    Float[Array, '... 1 (max_degree+1)**2 num_features'],
    Float[Array, '... 2 (max_degree+1)**2 num_features'],
]:
  r"""Hard SiLU/Swish activation function.

  Computes the element-wise function (see
  :func:`hard_sigmoid <jax.nn.hard_sigmoid>`)

  .. math::
    \mathrm{hard\_silu}(x) = x \cdot \mathrm{hard\_sigmoid}(x)

  .. jupyter-execute::
    :hide-code:

    import numpy as np, matplotlib.pyplot as plt
    import matplotlib_inline.backend_inline as inl
    from e3x.nn import hard_silu
    inl.set_matplotlib_formats('pdf', 'svg')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    x = np.linspace(-3, 3, num=1001)
    y = np.squeeze(hard_silu(x.reshape(1,1,-1)))
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{hard\_silu}(x)$')
    plt.plot(x, y, lw=3); plt.grid()

  Args:
    x: Input features to which the nonlinearity is applied.

  Returns:
    The result of applying the nonlinearity to the input features.
  """
  return _gated_linear(jax.nn.hard_sigmoid, x)


hard_swish = hard_silu


def bent_identity(
    x: Union[
        Float[Array, '... 1 (max_degree+1)**2 num_features'],
        Float[Array, '... 2 (max_degree+1)**2 num_features'],
    ]
) -> Union[
    Float[Array, '... 1 (max_degree+1)**2 num_features'],
    Float[Array, '... 2 (max_degree+1)**2 num_features'],
]:
  r"""Bent identity activation function.

  Computes the gated linear activation with:

  .. math::
    \mathrm{gate}(x) = \frac{\sqrt{x^2 + 1}-1}{2x} + 1

  For scalar inputs, this is equivalent to:

  .. math::
    \mathrm{bent\_identity}(x) = \frac{\sqrt{x^2 + 1}-1}{2} + x

  .. jupyter-execute::
    :hide-code:

    import numpy as np, matplotlib.pyplot as plt
    import matplotlib_inline.backend_inline as inl
    from e3x.nn import bent_identity
    inl.set_matplotlib_formats('pdf', 'svg')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    x = np.linspace(-3, 3, num=1001)
    y = np.squeeze(bent_identity(x.reshape(1,1,-1)))
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{bent\_identity}(x)$')
    plt.plot(x, y, lw=3); plt.grid()

  Args:
    x: Input features to which the nonlinearity is applied.

  Returns:
    The result of applying the nonlinearity to the input features.
  """

  def g(
      x: Float[Array, '... 1 1 num_features']
  ) -> Float[Array, '... 1 1 num_features']:
    not_tiny = jnp.abs(x) > jnp.finfo(x.dtype).eps
    safe_x = jnp.where(not_tiny, x, 1)
    return jnp.where(
        not_tiny,
        (jnp.sqrt(x * x + 1) - 1) / (2 * safe_x) + 1,
        x / 4 + 1,  # + O(x^3) (Taylor series around x=0).
    )

  return _gated_linear(g, x)
