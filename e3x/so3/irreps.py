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

r"""Functions related to irreducible representations of rotations."""

import itertools
from e3x import ops
from e3x.config import Config
import jax
import jax.numpy as jnp
import jaxtyping
import more_itertools

from ._clebsch_gordan_lut import _generate_clebsch_gordan_lookup_table
from ._common import _cartesian_permutation
from ._common import _cartesian_permutation_for_degree
from ._common import _check_degree_is_positive_or_zero
from ._common import _integer_powers
from ._normalization import Normalization
from ._normalization import normalization_constant
from ._spherical_harmonics_lut import _generate_spherical_harmonics_lookup_table
from ._tensor_conversion_lut import _generate_tensor_conversion_lookup_table
# pylint: enable=g-importing-member

Array = jaxtyping.Array
Float = jaxtyping.Float
Integer = jaxtyping.Integer


def spherical_harmonics(
    r: Float[Array, '... 3'],
    max_degree: int,
    r_is_normalized: bool = False,
    cartesian_order: bool = Config.cartesian_order,
    normalization: Normalization = Config.normalization,
) -> Float[Array, '... (max_degree+1)**2']:
  r"""Real Cartesian spherical harmonics :math:`Y_\ell^m(\vec{r})`.

  Evaluates :math:`Y_\ell^m(\vec{r})` for all :math:`\ell=0,\dots,L` and
  :math:`m=-\ell,\dots,\ell` with :math:`L` = ``max_degree``. The
  `spherical harmonics <https://en.wikipedia.org/wiki/Spherical_harmonics>`_ are
  basis functions for irreducible representations of :math:`\mathrm{SO}(3)`. In
  total, there are :math:`(L+1)^2` spherical harmonics for a given :math:`L`.
  For example, these are all spherical harmonics for :math:`L=3` (blue:
  positive, red: negative, arrows show the :red:`x`-, :green:`y`-, and
  :blue:`z`-axes, click & drag to rotate):

  .. raw:: html

   <iframe src="../_static/spherical_harmonics.html" width="670" height="670"
   frameBorder="0" scrolling="no">spherical harmonics up to degree 3</iframe>

  In general, the real Cartesian spherical harmonics are given by

  .. math::
    Y_{\ell}^{m}(\vec{r}) = \mathcal{N}\begin{cases}
    \sqrt{2}\cdot \Pi_\ell^{\lvert m\rvert}(z) \cdot A_{\lvert m \rvert}(x,y)
    & m < 0 \\
    \Pi_\ell^{0}(z)
    & m = 0 \\
    \sqrt{2}\cdot \Pi_\ell^{m}(z) \cdot B_m(x,y)
    & m > 0 \\
    \end{cases}
  .. math::
    A_{m}(x,y) = \sum_{k=0}^{ m}\binom{m}{k}x^{k} y^{m-k}
      \sin\left(\frac{\pi}{2}(m-k)\right)
  .. math::
    B_{m}(x,y) = \sum_{k=0}^{m}\binom{m}{k}x^{k} y^{m-k}
      \cos\left(\frac{\pi}{2}(m-k)\right)
  .. math::
    \Pi_{\ell}^{m}(z) = \sqrt{\frac{(\ell-m)!}{(\ell+m)!}}
      \sum_{k=0}^{\lfloor(\ell-m)/2\rfloor} \ \frac{(-1)^k}{2^\ell}
      \binom{\ell}{k} \binom{2\ell-2k}{\ell}\frac{(\ell-2k)!}{(\ell-2k-m)!}
      r^{2k-\ell}z^{\ell-2k-m}

  with :math:`\vec{r}=[x\ y\ z]^\intercal \in \mathbb{R}^3` and
  :math:`r = \lVert \vec{r} \rVert`. Here, :math:`\mathcal{N}` is a
  normalization constant that depends on the chosen normalization scheme. When
  ``normalization`` is ``'racah'`` or ``'schmidt'`` Racah's normalization (also
  known as Schmidt's semi-normalization) is used (the integral runs over the
  surface of the unit sphere :math:`\Omega`):

  .. math::
    \mathcal{N} = 1 \qquad
    \int_{\Omega} Y_\ell^m(\vec{r}) Y_{\ell'}^{m'}(\vec{r}) d\Omega =
      \frac{4\pi}{2\ell+1}\delta_{\ell\ell'}\delta_{mm'}\,,

  when ``normalization`` is ``'4pi'``:

  .. math::
    \mathcal{N} = \sqrt{2\ell+1} \qquad
    \int_{\Omega} Y_\ell^m(\vec{r}) Y_{\ell'}^{m'}(\vec{r}) d\Omega =
      4\pi\delta_{\ell\ell'}\delta_{mm'}\,,

  and when ``normalization`` is ``'orthonormal'``:

  .. math::
    \mathcal{N} = \sqrt{\frac{2\ell+1}{4\pi}} \qquad
    \int_{\Omega} Y_\ell^m(\vec{r}) Y_{\ell'}^{m'}(\vec{r}) d\Omega =
      \delta_{\ell\ell'}\delta_{mm'}\,.

  Args:
    r: Array of shape ``(..., 3)`` containing Cartesian vectors :math:`\vec{r}`.
    max_degree: Maximum degree :math:`L` of the spherical harmonics.
    r_is_normalized: If True, :math:`\vec{r}` is assumed to be already
      normalized.
    cartesian_order: If ``True``, spherical harmonics are returned in Cartesian
      order.
    normalization: Which normalization is used for the spherical harmonics.

  Returns:
    The values :math:`Y_\ell^m(\vec{r})` of all spherical harmonics up to
    :math:`\ell` = ``max_degree``. Values are returned in an Array of shape
    ``(..., (max_degree+1)**2)`` ordered
    :math:`[Y_{0}^{0}\ Y_{1}^{1}\ Y_{1}^{-1}\ Y_{1}^{0}\ Y_{2}^{2}\ \cdots]`.
    If ``cartesian_order = False``, values are ordered
    :math:`[Y_{0}^{0}\ Y_{1}^{-1}\ Y_{1}^{0}\ Y_{1}^{1}\ Y_{2}^{-2}\ \cdots]`
    instead.

  Raises:
    ValueError: If ``r`` has an invalid shape (not a 3-vector), ``max_degree``
    is not positive or zero, or ``normalization`` has an invalid value.
  """
  # Perform checks.
  if r.shape[-1] != 3:
    raise ValueError(f'r must have shape (..., 3), received shape {r.shape}')
  _check_degree_is_positive_or_zero(max_degree)

  with jax.ensure_compile_time_eval():
    # Load/Generate lookup table and convert to jax array.
    lookup_table = _generate_spherical_harmonics_lookup_table(max_degree)
    cm = jnp.asarray(lookup_table['cm'])
    ls = jnp.asarray(lookup_table['ls'])
    # Apply normalization constants.
    for l in range(max_degree + 1):
      cm = cm.at[:, l**2 : (l + 1) ** 2].multiply(
          normalization_constant(normalization, l)
      )
    # Optionally reorder spherical harmonics to Cartesian order.
    if cartesian_order:
      cm = cm[:, _cartesian_permutation(max_degree)]

  # Normalize r (if not already normalized).
  if not r_is_normalized:
    r = ops.normalize(r, axis=-1)

  # Calculate all relevant monomials in the (x, y, z)-coordinates.
  # Note: This is done via integer powers and indexing on purpose! Using
  # jnp.power or the "**"-operator for this operation leads to NaNs in the
  # gradients for some inputs (jnp.power is not NaN-safe).
  r_powers = _integer_powers(jnp.expand_dims(r, axis=-2), max_degree)
  monomials = (
      r_powers[..., 0][..., ls[:, 0]]  #   x**lx.
      * r_powers[..., 1][..., ls[:, 1]]  # y**ly.
      * r_powers[..., 2][..., ls[:, 2]]  # z**lz.
  )

  # Calculate and return spherical harmonics (linear combination of monomials).
  return jnp.matmul(monomials, cm)


def solid_harmonics(
    r: Float[Array, '... 3'],
    max_degree: int,
    cartesian_order: bool = Config.cartesian_order,
) -> Float[Array, '... (max_degree+1)**2']:
  r"""Real Cartesian (regular) solid harmonics :math:`R_\ell^m(\vec{r})`.

  The `solid harmonics <https://en.wikipedia.org/wiki/Solid_harmonics>`_ are
  defined as

  .. math::
    R_{\ell}^{m}(\vec{r}) = \lVert\vec{r}\rVert^\ell Y_{\ell}^{m}(\vec{r})\,,

  where :math:`Y_{\ell}^{m}(\vec{r})` are
  `spherical harmonics <https://en.wikipedia.org/wiki/Spherical_harmonics>`_
  using Racah's normalization. See also
  :func:`spherical_harmonics <e3x.so3.irreps.spherical_harmonics>` for more
  details.

  Args:
    r: Array of shape ``(..., 3)`` containing Cartesian vectors :math:`\vec{r}`.
    max_degree: Maximum degree :math:`L` of the solid harmonics.
    cartesian_order: If ``True``, solid harmonics are returned in Cartesian
      order.

  Returns:
    The values :math:`R_\ell^m(\vec{r})` of all solid harmonics up to
    :math:`\ell` = ``max_degree``. Values are returned in an Array of shape
    ``(..., (max_degree+1)**2)`` ordered
    :math:`[R_{0}^{0}\ R_{1}^{1}\ R_{1}^{-1}\ R_{1}^{0}\ R_{2}^{2}\ \cdots]`.
    If ``cartesian_order = False``, values are ordered
    :math:`[R_{0}^{0}\ R_{1}^{-1}\ R_{1}^{0}\ R_{1}^{1}\ R_{2}^{-2}\ \cdots]`
    instead.

  Raises:
    ValueError: If ``r`` has an invalid shape (not a 3-vector), or
    ``max_degree`` is not positive or zero.
  """
  return spherical_harmonics(
      r=r,
      max_degree=max_degree,
      cartesian_order=cartesian_order,
      normalization='racah',
      r_is_normalized=True,
  )


def clebsch_gordan(
    max_degree1: int,
    max_degree2: int,
    max_degree3: int,
    cartesian_order: bool = Config.cartesian_order,
) -> Float[Array, '(max_degree1+1)**2 (max_degree2+1)**2 (max_degree3+1)**2']:
  r"""Clebsch-Gordan coefficients for coupling all degrees at once.

  See the :ref:`corresponding section in the overview <CouplingIrreps>` for more
  details on coupling irreps.

  Args:
    max_degree1: Maximum degree of the first factor.
    max_degree2: Maximum degree of the second factor.
    max_degree3: Maximum degree of the tensor product.
    cartesian_order: If ``True``, Cartesian order is assumed.

  Returns:
    The values of all Clebsch-Gordan coefficients for coupling degrees up to the
    requested maximum degrees stored in an Array of shape
    ``((max_degree1+1)**2, (max_degree2+1)**2, (max_degree3+1)**2))``.

  Raises:
    ValueError: If ``max_degree1``, ``max_degree2``, or ``max_degree3`` are not
      positive or zero.
  """
  # Perform checks.
  _check_degree_is_positive_or_zero(max_degree1)
  _check_degree_is_positive_or_zero(max_degree2)
  _check_degree_is_positive_or_zero(max_degree3)

  with jax.ensure_compile_time_eval():
    # Load/Generate lookup table with Clebsch-Gordan coefficients.
    max_degree = max(max_degree1, max_degree2, max_degree3)
    lookup_table = _generate_clebsch_gordan_lookup_table(max_degree)
    # Extract relevant slices and convert to jax array.
    cg = jnp.asarray(
        lookup_table['cg'][
            : (max_degree1 + 1) ** 2,
            : (max_degree2 + 1) ** 2,
            : (max_degree3 + 1) ** 2,
        ]
    )
    # Optionally reorder spherical harmonics to Cartesian order.
    if cartesian_order:
      p1 = _cartesian_permutation(max_degree1)
      p2 = _cartesian_permutation(max_degree2)
      p3 = _cartesian_permutation(max_degree3)
      cg = cg[p1, :, :][:, p2, :][:, :, p3]

  return cg


def clebsch_gordan_for_degrees(
    degree1: int,
    degree2: int,
    degree3: int,
    cartesian_order: bool = Config.cartesian_order,
) -> Float[Array, '2*degree1+1 2*degree2+1 2*degree3+1']:
  r"""Clebsch-Gordan coefficients for coupling only specific degrees.

  See also :func:`clebsch_gordan <e3x.so3.clebsch_gordan>` fore more details.

  Args:
    degree1: Degree of the first factor.
    degree2: Degree of the second factor.
    degree3: Degree of the tensor product.
    cartesian_order: If ``True``, Cartesian order is assumed.

  Returns:
    The values of the Clebsch-Gordan coefficients for coupling the requested
    degrees stored in an Array of shape
    ``(2*degree1+1, 2*degree2+1, 2*degree3+1)``.

  Raises:
    ValueError: If ``degree1``, ``degree2``, or ``degree3`` are not positive or
      zero.
  """
  # Perform checks.
  _check_degree_is_positive_or_zero(degree1)
  _check_degree_is_positive_or_zero(degree2)
  _check_degree_is_positive_or_zero(degree3)

  with jax.ensure_compile_time_eval():
    # Load/Generate lookup table with Clebsch-Gordan coefficients.
    max_degree = max(degree1, degree2, degree3)
    lookup_table = _generate_clebsch_gordan_lookup_table(max_degree)
    # Extract relevant slices and convert to jax array.
    cg = jnp.asarray(
        lookup_table['cg'][
            degree1**2 : (degree1 + 1) ** 2,
            degree2**2 : (degree2 + 1) ** 2,
            degree3**2 : (degree3 + 1) ** 2,
        ]
    )
    # Optionally reorder spherical harmonics to Cartesian order.
    if cartesian_order:
      p1 = _cartesian_permutation_for_degree(degree1)
      p2 = _cartesian_permutation_for_degree(degree2)
      p3 = _cartesian_permutation_for_degree(degree3)
      cg = cg[p1, :, :][:, p2, :][:, :, p3]

  return cg


def _check_tensor_shape(x: Float[Array, '...'], degree: int) -> None:
  """Checks the shape of a tensor."""
  _check_degree_is_positive_or_zero(degree)
  if x.ndim == 0:
    raise ValueError('input must be a multi-dimensional array')
  if degree == 0:
    if x.shape[-1] != 1:
      raise ValueError(
          'input (assumed to be a tensor of degree 0) must have shape '
          f'(..., 1), received shape {x.shape}'
      )
  else:
    if x.shape[-degree:] != (3,) * degree:
      raise ValueError(
          f'input (assumed to be a tensor of degree {degree}) must have shape '
          f'(..., {", ".join(("3",)*degree)}), received shape {x.shape}'
      )


def is_traceless_symmetric(
    x: Float[Array, '...'], degree: int, rtol: float = 1e-5, atol: float = 1e-5
) -> bool:
  """Checks whether a given tensor is traceless and symmetric.

  Args:
    x: The input traceless symmetric tensor.
    degree: The degree of the input tensor.
    rtol: The relative tolerance parameter (see :func:`jax.numpy.allclose
      <jax.numpy.allclose>`).
    atol: The absolute tolerance parameter (see :func:`jax.numpy.allclose
      <jax.numpy.allclose>`).

  Returns:
    ``True`` if the tensor is traceless and symmetric, `False` otherwise.

  Raises:
    ValueError: If ``degree`` is not positive or zero or the shape of the tensor
      is inconsistent with ``degree``.
  """
  _check_degree_is_positive_or_zero(degree)
  _check_tensor_shape(x, degree)
  if degree < 2:  # Scalars and vectors are always traceless symmetric.
    return True
  # Check for symmetry.
  batch_dims = tuple(range(x.ndim - degree))
  for permutation in itertools.permutations(range(x.ndim - degree, x.ndim)):
    if not jnp.allclose(
        x,
        jnp.transpose(x, axes=(*batch_dims, *permutation)),
        rtol=rtol,
        atol=atol,
    ):
      return False
  # Check for tracelessness. Note: Checking only one trace is valid after
  # checking for symmetry, because for symmetric tensors, all traces are equal.
  return jnp.allclose(jnp.trace(x, axis1=-2, axis2=-1), 0, rtol=rtol, atol=atol)  # pytype: disable=bad-return-type  # jnp-type


def _tensor_compression_indices(
    degree: int,
) -> tuple[Integer[Array, '...'], ...]:
  """Returns indices for compressing a symmetric tensor.

  A Cartesian tensor of degree l has shape (3, 3, ..., 3), where the 3 is
  repeated l times, i.e. there are a total of 3**l entries. However, for totally
  symmetric tensors, only the (l+1)*(l+2)/2 entries in the "upper triangle" are
  independent. For this reason, it is more efficient to store symmetric tensors
  in a "compressed" format, where only the independent entries are stored in a
  flat array. This function returns a tuple of indices for compressing symmetric
  tensors via slicing. Note: for l=2, the output of this function is equivalent
  to jnp.triu_indices(3).

  Args:
    degree: degree of the tensor (also called order or rank).

  Returns:
    The indices for the "triangle". The returned tuple contains l arrays, each
    with the indices along one dimension of the tensor (can be used for
    slicing). A compressed tensor is obtained by slicing an uncompressed tensor
    with the returned indices.

  Raises:
    ValueError: If ``degree`` is not positive or zero.
  """
  _check_degree_is_positive_or_zero(degree)
  with jax.ensure_compile_time_eval():
    if degree == 0:
      indices = tuple()
    else:
      combinations = jnp.asarray(
          list(itertools.combinations_with_replacement((0, 1, 2), r=degree))
      )
      indices = tuple(
          jnp.squeeze(i, axis=-1) for i in jnp.hsplit(combinations, degree)
      )
  return indices


def tensor_to_irreps(
    x: Float[Array, '...'],  # Last degree dimensions must all have size 3.
    degree: int,
    cartesian_order: bool = Config.cartesian_order,
    normalization: Normalization = Config.normalization,
):
  """Converts a traceless symmetric tensor to irreducible representations.

  This function does not check whether the input tensor is actually symmetric
  and traceless, because doing so is incompatible with ``jax.jit``. Using a
  non-symmetric or traced tensor as input will lead to meaningless results.
  In case of doubt, the input should be checked manually with
  :func:`is_traceless_symmetric` prior to calling this function.

  Args:
    x: The input traceless symmetric tensor.
    degree: The degree of the input tensor.
    cartesian_order: If ``True``, irreps are returned in Cartesian order.
    normalization: Which normalization is used for the irreps.

  Returns:
    The corresponding irreducible representation. See
    :func:`spherical_harmonics <e3x.so3.irreps.spherical_harmonics>` for more
    information about the meaning of `cartesian_order` and `normalization`.

  Raises:
    ValueError: If ``degree`` is not positive or zero, the shape of ``x`` is
    inconsistent with ``degree`` or ``normalization`` has an invalid value.
  """
  # Perform checks.
  _check_degree_is_positive_or_zero(degree)
  _check_tensor_shape(x, degree)

  with jax.ensure_compile_time_eval():
    # Load/Generate lookup table and convert to jax array.
    lookup_table = _generate_tensor_conversion_lookup_table(degree)
    t2s = jnp.asarray(lookup_table['t2s'][degree])
    # Apply normalization constant.
    t2s *= normalization_constant(normalization, degree)
    # Optionally reorder irreps to Cartesian order.
    if cartesian_order:
      t2s = t2s[:, _cartesian_permutation_for_degree(degree)]

  return jnp.matmul(x[(..., *_tensor_compression_indices(degree))], t2s)


def _tensor_expansion_indices(degree) -> Integer[Array, '...']:
  """Returns indices for uncompressing a symmetric tensor.

  This can be used to invert the effect of slicing a tensor with the output of
  _tensor_compression_indices: The compressed tensor needs to be sliced with
  the output of this function (and reshaped, because otherwise a flattened
  version is obtained).

  Args:
    degree: degree of the tensor (also called order or rank).

  Returns:
    The indices for expanding the compressed tensor. An uncompressed tensor is
    obtained by slicing a compressed tensor with the returned indices (and
    reshaping the result).

  Raises:
    ValueError: If ``degree`` is not positive or zero.
  """
  _check_degree_is_positive_or_zero(degree)
  with jax.ensure_compile_time_eval():
    # Maps multi-index tuple for indexing a tensor with the given degree to the
    # corresponding index in a flattened array.
    flat_index_map = {}
    for i, index_tuple in enumerate(
        itertools.product((0, 1, 2), repeat=degree)
    ):
      flat_index_map[index_tuple] = i

    # Indices for expanding the (l+1)*(l+2)/2 independent components stored in
    # a compressed tensor to the 3**l components in an uncompressed tensor.
    indices = jnp.empty(shape=len(flat_index_map), dtype=int)
    for i, index_tuple in enumerate(
        itertools.combinations_with_replacement((0, 1, 2), r=degree)
    ):
      for permutation in more_itertools.distinct_permutations(index_tuple):
        indices = indices.at[flat_index_map[permutation]].set(i)
  return indices


def irreps_to_tensor(
    x: Float[Array, '... 2*degree+1'],
    degree: int,
    cartesian_order: bool = Config.cartesian_order,
    normalization: Normalization = Config.normalization,
):
  """Converts irreducible representations to a traceless symmetric tensor.

  Args:
    x: The input irreducible representation.
    degree: The degree of the input irreducible representation.
    cartesian_order: If ``True``, irreps are assumed to be in Cartesian order.
    normalization: Which normalization is used.

  Returns:
    The corresponding traceless symmetric tensor. See
    :func:`spherical_harmonics <e3x.so3.irreps.spherical_harmonics>` for more
    information about the meaning of `cartesian_order` and `normalization`.

  Raises:
    ValueError: If ``degree`` is not positive or zero, the shape of ``x`` is
      inconsistent with ``degree`` or ``normalization`` has an invalid value.
  """
  # Perform checks.
  _check_degree_is_positive_or_zero(degree)
  if x.shape[-1] != 2 * degree + 1:
    raise ValueError(
        f'input (assumed to correspond to irreps of degree {degree}) must have'
        f' shape (..., {2*degree + 1}), received shape {x.shape}'
    )

  with jax.ensure_compile_time_eval():
    # Load/Generate lookup table and convert to jax array.
    lookup_table = _generate_tensor_conversion_lookup_table(degree)
    s2t = jnp.asarray(lookup_table['s2t'][degree])
    # Apply normalization constant.
    s2t /= normalization_constant(normalization, degree)
    # Optionally reorder irreps to Cartesian order.
    if cartesian_order:
      s2t = s2t[_cartesian_permutation_for_degree(degree), :]

  y = jnp.matmul(x, s2t)  # First convert to the compressed tensor.
  y = y[..., _tensor_expansion_indices(degree)]  # Expand tensor.
  # Return in correct shape.
  shape = (1,) if degree == 0 else (3,) * degree
  return y.reshape((*y.shape[:-1], *shape))
