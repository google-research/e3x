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

r"""Convenience wrappers to simplify usage of other functions."""

import functools
from typing import Any, Callable, Optional, Protocol, Tuple, Union
from e3x import config
from e3x import ops
from e3x import so3
from flax import linen as nn
import jax
import jax.numpy as jnp
import jaxtyping

Array = jaxtyping.Array
Float = jaxtyping.Float
Dtype = Any  # This could be a real type if support for that is added.


_default_angular_fn = functools.partial(
    so3.spherical_harmonics, r_is_normalized=True, normalization='racah'
)


class AngularFn(Protocol):
  """Protocol for angular functions."""

  def __call__(
      self, r: Float[Array, '... 3'], max_degree: int, cartesian_order: bool
  ) -> Float[Array, '... (max_degree+1)**2']:
    ...


def basis(
    r: Float[Array, '... 3'],
    *,
    max_degree: int,
    num: int,
    radial_fn: Callable[[Float[Array, '...'], int], Float[Array, '... num']],
    angular_fn: AngularFn = _default_angular_fn,
    cutoff_fn: Optional[
        Callable[[Float[Array, '...']], Float[Array, '...']]
    ] = None,
    return_cutoff: bool = False,
    return_norm: bool = False,
    damping_fn: Optional[
        Callable[[Float[Array, '...']], Float[Array, '...']]
    ] = None,
    cartesian_order: bool = config.Config.cartesian_order,
) -> Union[
    Float[Array, '... 1 (max_degree+1)**2 num'],
    Tuple[Float[Array, '... 1 (max_degree+1)**2 num'], Float[Array, '...']],
    Tuple[
        Float[Array, '... 1 (max_degree+1)**2 num'],
        Float[Array, '...'],
        Float[Array, '...'],
    ],
]:
  r"""Convenience wrapper for computing radial-angular basis functions.

  This function can be used to compute radial-angular basis functions of the
  form

  .. math::
    \mathrm{B}_{n\ell m}(\vec{r}) = R_{n\ell}\left(\lVert\vec{r}\rVert\right)
    A_\ell^m\left(\frac{\vec{r}}{\lVert\vec{r}\rVert}\right)

  for input vectors :math:`\vec{r}=[x\ y\ z]^\intercal \in \mathbb{R}^3`. Here,
  :math:`R_{n\ell}` is the radial component and :math:`A_\ell^m` are
  angular components (given by ``angular_fn``). In the most simple case, the
  radial component is independent of :math:`\ell` and given by

  .. math::
    R_{n\ell}(r) = g_n(r)\,,

  where :math:`g_n(r)` is one of the outputs of ``radial_fn``. However, since
  angular functions such as the spherical harmonics :math:`Y_\ell^m` for
  :math:`\ell > 0` are undefined when :math:`\vec{r}=[0\ 0\ 0]^\intercal`,
  depending on whether zero vectors are expected as inputs, it can be desirable
  to damp the radial component for small input vectors. When ``damping_fn`` is
  not ``None``, the radial component is instead given by

  .. math::
    R_{n\ell}(r) =
    \begin{cases}
    g_n(r) & l = 0 \\
    g_n(r) \cdot d(r) & l > 0 \\
    \end{cases}\,,

  where :math:`d(r)` is the output of ``damping_fn``. Similarly, it is often
  useful to combine the radial function with a cutoff function, such that the
  radial component is zero beyond a certain cutoff radius. When ``cutoff_fn``
  is not ``None``, the radial component is given by

  .. math::
    R_{n\ell}(r) = g_n(r) \cdot c(r)\,,

  where :math:`c(r)` is the output of ``cutoff_fn``. It is also possible to
  combine ``damping_fn`` and ``cutoff_fn``, in which case the radial component
  is given by

  .. math::
    R_{n\ell}(r) =
    \begin{cases}
    g_n(r) \cdot c(r) & l = 0 \\
    g_n(r) \cdot c(r) \cdot d(r) & l > 0\,. \\
    \end{cases}

  Example:
    >>> import jax.numpy as jnp
    >>> import e3x
    >>> r = jnp.asarray([[0.5, 1.2, -0.1], [-0.4, 0.5, 1.2]])
    >>> basis = e3x.nn.basis(
    ...   r=r,
    ...   max_degree=1,
    ...   num=8,
    ...   radial_fn=e3x.nn.sinc,
    ...   cutoff_fn=e3x.nn.smooth_cutoff,
    ...   damping_fn=e3x.nn.smooth_damping,
    ... )
    >>> basis.shape
    (2, 1, 4, 8)

  Args:
    r: Input array of shape ``(..., 3)`` containing Cartesian vectors.
    max_degree: Maximum degree of the spherical harmonics.
    num: Number of radial basis functions.
    radial_fn: Callable for computing radial basis functions. This function
      should take an array of shape ``(...)`` (the norm of ``r``) and an integer
      (the number of radial basis functions ``num``) as input and return an
      array of shape ``(..., num)`` (the values of ``num`` radial basis
      functions :math:`g_n`).
    angular_fn: Callable for computing angular basis functions. This function
      should take an array of shape ``(..., 3)`` (Cartesian vectors ``r``), an
      integer (the maximum degree ``max_degree``), and a boolean (which ordering
      convention to use, see ``cartesian_order``) as input and return an array
      of shape ``(..., (max_degree+1)**2)`` (the values of the angular basis
      functions :math:`A_\ell^m`). By default, spherical harmonics with Racah's
      normalization are used.
    cutoff_fn: Optional Callable for computing cutoff values. This function
      should take an array of shape ``(...)`` (the norm of ``r``) as input and
      return an array of shape ``(...)`` (the values of the cutoff function).
    return_cutoff: If ``True``, also return the values of the cutoff function.
    return_norm: If ``True``, also return the norm of the input vectors ``r``.
    damping_fn: Optional Callable for computing damping values. This function
      should take an array of shape ``(...)`` (the norm of ``r``) as input and
      return an array of shape ``(...)`` (the values of the damping function).
      If present, basis functions with spherical harmonics degree
      :math:`\ell > 0` are multiplied with the values of the damping function.
    cartesian_order: If ``True``, spherical harmonics are in Cartesian order.

  Returns:
    Value of all basis functions for all values in ``r``. If the input has shape
    ``(..., 3)``, the output has shape ``(..., 1, (max_degree+1)**2, num)`` (the
    ``1`` in the shape is a parity axis added for compatibility with other
    methods). If ``return_cutoff_value=True``, or ``return_norm=True``, also
    returns the values of the cutoff function/vector norms  with shape ``(...)``
    (a tuple of basis function values and cutoff values and/or norms is
    returned).
  """
  # Check that r is a collection of 3-vectors.
  if r.shape[-1] != 3:
    raise ValueError(f'r must have shape (..., 3), received shape {r.shape}')

  # Check that cutoff_fn is specified when cutoff values are requested.
  if return_cutoff and cutoff_fn is None:
    raise ValueError('return_cutoff is True, but no cutoff_fn was specified')

  # Normalize input vectors.
  norm = ops.norm(r, axis=-1, keepdims=True)  # (..., 1)
  u = r / jnp.where(norm > 0, norm, 1)
  norm = norm.squeeze(-1)  # (...)

  # Evaluate radial basis functions.
  rbf = radial_fn(norm, num)  # (..., N)

  # Optionally: Apply cutoff function.
  if cutoff_fn is not None:
    cut = cutoff_fn(norm)  # (...)
    rbf = rbf * jnp.expand_dims(cut, axis=-1)  # (..., N) * (..., 1)
  else:
    cut = jnp.ones_like(norm)

  # Evaluate angular basis functions.
  ylm = angular_fn(  # (..., (L+1)**2)
      r=u,
      max_degree=max_degree,
      cartesian_order=cartesian_order,
  )

  # Combine radial and angular basis functions.
  ylm = jnp.expand_dims(ylm, axis=-1)  # (..., (L+1)**2, 1)
  rbf = jnp.expand_dims(rbf, axis=-2)  # (..., 1, N)
  out = ylm * rbf  # (..., (L+1)**2, N)

  # Optionally: Apply damping function.
  if damping_fn is not None:
    damping_values = damping_fn(norm)  # (...)
    damping_values = jnp.expand_dims(damping_values, axis=(-2, -1))
    out = out.at[..., 1:, :].multiply(damping_values)

  # Add parity axis.
  out = jnp.expand_dims(out, axis=-3)  # (..., 1, (L+1)**2, N)

  # Add optional return values.
  if (return_cutoff and cutoff_fn is not None) or return_norm:
    out = (out,)
    if return_cutoff and cutoff_fn is not None:
      out += (cut,)
    if return_norm:
      out += (norm,)
  return out


class ExponentialBasis(nn.Module):
  """Exponential basis module.

  This module wraps :func:`basis <e3x.nn.wrappers.basis>` and injects a
  learnable `gamma` parameter into the provided `radial_fn`. Only works with
  radial functions that accept a `gamma` keyword (all exponentially mapped
  radial functions).
  """

  initial_gamma: float = 1.0
  param_dtype: Dtype = jnp.float32

  @nn.compact
  def __call__(self, *args, **kwargs):
    gamma = jax.nn.softplus(
        self.param(
            'gamma',
            lambda _, dtype: ops.inverse_softplus(  # PRNGKey is unused.
                jnp.array(self.initial_gamma, dtype=dtype)
            ),
            self.param_dtype,
        )
    )
    kwargs['radial_fn'] = functools.partial(kwargs['radial_fn'], gamma=gamma)
    return basis(*args, **kwargs)
