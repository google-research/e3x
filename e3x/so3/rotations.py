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

"""Functions for generating rotation matrices."""

import functools
from typing import Tuple, Union
from e3x import ops
from e3x.config import Config
import jax
import jax.numpy as jnp
import jaxtyping

from ._common import _cartesian_permutation_wigner_d_entries
from ._common import _integer_powers
from ._wigner_d_lut import _generate_wigner_d_lookup_table
from .irreps import clebsch_gordan
# pylint: enable=g-importing-member

Array = jaxtyping.Array
Float = jaxtyping.Float
UInt32 = jaxtyping.UInt32
PRNGKey = UInt32[Array, '2']


def _check_rotation_matrix_shape(rot: Float[Array, '...']) -> None:
  """Helper function to check the shape of a rotation matrix.

  Args:
    rot: Array that should be checked for the correct shape.

  Raises:
    ValueError: If the shape is invalid for a rotation matrix.
  """
  if rot.shape[-2:] != (3, 3):
    raise (
        ValueError(
            'rotation matrices must have shape (..., 3, 3), received '
            f'shape {rot.shape}'
        )
    )


def rotation(
    axis: Float[Array, '... 3'], angle: Float[Array, '...']
) -> Float[Array, '... 3 3']:
  r""":math:`3\times3` rotation matrix specified by an axis and an angle.

  Calculates the :math:`3\times3` rotation matrix associated with the
  counterclockwise rotation about the given ``axis`` by ``angle`` using the
  Euler-Rodrigues formula.

  Example:
    >>> import jax.numpy as jnp
    >>> import e3x
    >>> r = jnp.asarray([1., 0., 0.])  # Unit vector in x-direction.
    >>> axis = jnp.asarray([0., 1., 0.])  # Rotate around y-axis.
    >>> angle = jnp.pi  # Rotate by 180°.
    >>> jnp.round(r @ e3x.so3.rotation(axis, angle), decimals=3)
    Array([-1.,  0.,  0.], dtype=float32)

  Args:
    axis: Rotation axis.
    angle: Rotation angle in radians.

  Returns:
    The :math:`3\times3` rotation matrix.
  """
  axis = ops.normalize(axis, axis=-1)
  a = jnp.cos(0.5 * angle)
  tmp = -axis * jnp.expand_dims(jnp.sin(0.5 * angle), -1)
  b = tmp[..., 0]
  c = tmp[..., 1]
  d = tmp[..., 2]
  a2, b2, c2, d2 = a * a, b * b, c * c, d * d
  ab, ac, ad, bc, bd, cd = a * b, a * c, a * d, b * c, b * d, c * d
  row1 = jnp.stack((a2 + b2 - c2 - d2, 2 * (bc + ad), 2 * (bd - ac)), axis=-1)
  row2 = jnp.stack((2 * (bc - ad), a2 + c2 - b2 - d2, 2 * (cd + ab)), axis=-1)
  row3 = jnp.stack((2 * (bd + ac), 2 * (cd - ab), a2 + d2 - b2 - c2), axis=-1)
  return jnp.stack((row1, row2, row3), axis=-1)


def alignment_rotation(
    u: Float[Array, '... 3'], v: Float[Array, '... 3']
) -> Float[Array, '... 3 3']:
  r"""Rotation matrix that aligns :math:`\vec{u}` with :math:`\vec{v}`.

  Calculates the :math:`3\times3` rotation matrix that aligns the vector
  :math:`\vec{u}` with the vector :math:`\vec{v}` using the shortest possible
  arc. When :math:`\vec{u}` and :math:`\vec{v}` are exactly antiparallel, there
  are infinitely many paths with the same arc length and one of them is chosen
  at random.

  Example:
    >>> import jax.numpy as jnp
    >>> import e3x
    >>> u = jnp.asarray([1., 0., 0.])  # Unit vector in x-direction.
    >>> v = jnp.asarray([0., 1., 0.])  # Unit vector in y-direction.
    >>> jnp.round(u @ e3x.so3.alignment_rotation(u, v), decimals=3)
    Array([0., 1., 0.], dtype=float32)

  Args:
    u: Vector :math:`\vec{u}`.
    v: Vector :math:`\vec{v}`.

  Returns:
    The :math:`3\times3` rotation matrix.
  """
  # Normalize the input vectors.
  u = ops.normalize(u, axis=-1)
  v = ops.normalize(v, axis=-1)

  # Create the skew-symmetric cross product matrix.
  x = jnp.cross(u, v)  # Cross product vector.
  zeros = jnp.zeros_like(x[..., 0:1])
  col1 = jnp.concatenate((zeros, -x[..., 2:3], x[..., 1:2]), axis=-1)
  col2 = jnp.concatenate((x[..., 2:3], zeros, -x[..., 0:1]), axis=-1)
  col3 = jnp.concatenate((-x[..., 1:2], x[..., 0:1], zeros), axis=-1)
  skew = jnp.stack((col1, col2, col3), axis=-1)

  # Calculate rotation matrix (does not handle antiparallel case correctly).
  div = 1 + jnp.sum(u * v, axis=-1)
  mask = div > 0
  safe_div = jnp.where(mask, div, 1)
  rot = skew + skew @ skew * (1 / safe_div)[..., None, None] + jnp.eye(3)

  # Handle antiparallel case.
  w = ops.normalize(u + jnp.roll(u, shift=1, axis=-1))
  axis = jnp.cross(u, w)
  rot = jnp.where(mask[..., None, None], rot, rotation(axis, jnp.pi))

  return rot


def rotation_euler(
    a: Float[Array, '...'],
    b: Float[Array, '...'],
    c: Float[Array, '...'],
) -> Float[Array, '... 3 3']:
  r""":math:`3\times3` rotation matrix specified by Euler angles.

  Calculates the :math:`3\times3` rotation matrix associated with the
  counterclockwise rotation about the :math:`x`-axis by ``a``, followed by
  counterclockwise rotation about the :math:`y`-axis by ``b``, followed by
  counterclockwise rotation about the :math:`z`-axis by ``c``. Note that the
  order of rotations matters for the result.

  Example:
    >>> import jax.numpy as jnp
    >>> import e3x
    >>> r = jnp.asarray([1., 0., 0.])  # Unit vector in x-direction.
    >>> # Rotate by 180° about y-axis.
    >>> jnp.round(r @ e3x.so3.rotation_euler(0.0, jnp.pi, 0.0), decimals=3)
    Array([-1.,  0.,  0.], dtype=float32)

  Args:
    a: Rotation angle about :math:`x`-axis in radians.
    b: Rotation angle about :math:`y`-axis in radians.
    c: Rotation angle about :math:`z`-axis in radians.

  Returns:
    The :math:`3\times3` rotation matrix.
  """
  sin_a = jnp.sin(a)
  cos_a = jnp.cos(a)
  sin_b = jnp.sin(b)
  cos_b = jnp.cos(b)
  sin_c = jnp.sin(c)
  cos_c = jnp.cos(c)
  row1 = jnp.stack(
      (
          cos_b * cos_c,
          sin_a * sin_b * cos_c - cos_a * sin_c,
          cos_a * sin_b * cos_c + sin_a * sin_c,
      ),
      axis=-1,
  )
  row2 = jnp.stack(
      (
          cos_b * sin_c,
          sin_a * sin_b * sin_c + cos_a * cos_c,
          cos_a * sin_b * sin_c - sin_a * cos_c,
      ),
      axis=-1,
  )
  row3 = jnp.stack(
      (-sin_b, sin_a * cos_b, cos_a * cos_b),
      axis=-1,
  )
  return jnp.stack((row1, row2, row3), axis=-1)


def euler_angles_from_rotation(
    rot: Float[Array, '... 3 3']
) -> Tuple[Float[Array, '...'], Float[Array, '...'], Float[Array, '...']]:
  r"""Extracts Euler angles from a rotation matrix.

  This function returns three values a, b, and c, corresponding to angles (in
  radians) for counterclockwise rotation about the x-, y-, and z-axes. The
  rotation about Euler angles is assumed to be performed in x, y, z order. See
  also :func:`rotation_euler <e3x.so3.rotations.rotation_euler>` for
  constructing rotation matrices from Euler angles and the conventions used.

  Args:
    rot: An Array of shape `(..., 3, 3)` representing :math:`3\times3` rotation
      matrices.

  Returns:
    A tuple `(a, b, c)` representing the three Euler angles. Each entry in the
    tuple has shape `(...)`.

  Raises:
    ValueError: If ``rot`` does not have shape `(..., 3, 3)`.
  """
  # Check shape, raises if shape is not (..., 3, 3).
  _check_rotation_matrix_shape(rot)

  # Normal case: rot[..., 0, 2] is neither 1 nor -1.
  b = -jnp.arcsin(rot[..., 0, 2])
  cos_b = jnp.cos(b)
  safe_cos_b = jnp.where(cos_b != 0.0, cos_b, 1.0)  # Make safe for division.
  a = jnp.arctan2(rot[..., 1, 2] / safe_cos_b, rot[..., 2, 2] / safe_cos_b)
  c = jnp.arctan2(rot[..., 0, 1] / safe_cos_b, rot[..., 0, 0] / safe_cos_b)

  # Special case: rot[..., 0, 2] is 1 or -1.
  almost1 = 1.0 - jnp.finfo(rot.dtype).epsneg
  mask_pos = rot[..., 0, 2] > almost1  # rot[..., 0, 2] is 1.
  mask_neg = rot[..., 0, 2] < -almost1  # rot[..., 0, 2] is -1.
  c = jnp.where(jnp.logical_or(mask_pos, mask_neg), 0, c)  # c is arbitrary.
  # Handle rot[..., 0, 2] = 1.
  a = jnp.where(mask_pos, jnp.arctan2(-rot[..., 1, 0], -rot[..., 2, 0]), a)
  b = jnp.where(mask_pos, -jnp.pi / 2, b)
  # Handle rot[..., 0, 2] = -1.
  a = jnp.where(mask_neg, jnp.arctan2(rot[..., 1, 0], rot[..., 2, 0]), a)
  b = jnp.where(mask_neg, jnp.pi / 2, b)

  return a, b, c


def random_rotation(
    key: PRNGKey,
    perturbation: float = 1.0,
    num: int = 1,  # When num=1, leading dimension is automatically squeezed.
) -> Union[Float[Array, '3 3'], Float[Array, 'num 3 3']]:
  r"""Samples a random :math:`3\times3` rotation matrix.

  Samples random :math:`3\times3` rotation matrices from :math:`\mathrm{SO(3)}`.
  The ``perturbation`` parameter controls how strongly random points on a sphere
  centered on the origin are perturbed by the rotation. For
  ``perturbation=1.0``, any point on the sphere is rotated to any other point on
  the sphere with equal probability. If ``perturbation<1.0``, returned rotation
  matrices are biased to identity matrices. For example, with
  ``perturbation=0.5``, a point on the sphere is rotated to any other point on
  the same hemisphere with equal probability.

  Example:
    >>> import jax
    >>> import e3x
    >>> e3x.so3.random_rotation(jax.random.PRNGKey(0), perturbation=1.0)
    Array([[-0.93064284, -0.11807037,  0.34635717],
           [ 0.33270139,  0.1210826 ,  0.9352266 ],
           [-0.15236041,  0.9855955 , -0.07340252]], dtype=float32)
    >>> e3x.so3.random_rotation(jax.random.PRNGKey(0), perturbation=0.0)
    Array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]], dtype=float32)

  Args:
    key: A PRNG key used as the random key.
    perturbation: A value between 0.0 and 1.0 that determines the perturbation.
    num: Number of returned rotation matrices.

  Returns:
    An Array of shape :math:`(\mathrm{num}, 3, 3)` or :math:`(3, 3)` (if num =
    1) representing random :math:`3\times3` rotation matrices.
  """
  # Check that perturbation is a meaningful value.
  if not 0.0 <= perturbation <= 1.0:
    raise ValueError(
        f'perturbation must be between 0.0 and 1.0, received {perturbation}'
    )
  # Draw random numbers and transform them.
  twopi = 2 * jnp.pi
  u = jax.random.uniform(key, shape=(num, 3))
  sqrt1 = jnp.sqrt(1 - u[..., 0])
  sqrt2 = jnp.sqrt(u[..., 0])
  angl1 = twopi * u[..., 1]
  angl2 = twopi * u[..., 2]
  # Construct random quaternion.
  r = sqrt1 * jnp.sin(angl1)
  i = sqrt1 * jnp.cos(angl1)
  j = sqrt2 * jnp.sin(angl2)
  k = sqrt2 * jnp.cos(angl2)
  # Perturbation (Slerp starting from identity quaternion).
  flip = r < 0  # Flip sign if r < 0 (always take the shorter route).
  r = jnp.where(flip, -r, r)
  i = jnp.where(flip, -i, i)
  j = jnp.where(flip, -j, j)
  k = jnp.where(flip, -k, k)
  phi = jnp.arccos(r)
  sinphi = jnp.sin(phi)
  # Prevent division by zero.
  zeromask = jnp.abs(sinphi) < 1e-9
  f1 = jnp.where(
      zeromask, 1 - perturbation, jnp.sin((1 - perturbation) * phi) / sinphi
  )
  f2 = jnp.where(zeromask, perturbation, jnp.sin(perturbation * phi) / sinphi)
  r, i, j, k = f1 + f2 * r, f2 * i, f2 * j, f2 * k
  # Construct rotation matrix.
  i2, j2, k2 = i * i, j * j, k * k
  ij, ik, jk, ir, jr, kr = i * j, i * k, j * k, i * r, j * r, k * r
  row1 = jnp.stack((1 - 2 * (j2 + k2), 2 * (ij - kr), 2 * (ik + jr)), axis=-1)
  row2 = jnp.stack((2 * (ij + kr), 1 - 2 * (i2 + k2), 2 * (jk - ir)), axis=-1)
  row3 = jnp.stack((2 * (ik - jr), 2 * (jk + ir), 1 - 2 * (i2 + j2)), axis=-1)
  rot = jnp.squeeze(jnp.stack((row1, row2, row3), axis=-1))
  return rot


@functools.partial(jnp.vectorize, excluded={3, 4}, signature='(),(),()->(n,n)')
def _wigner_d_euler(
    a: Float[Array, ''],
    b: Float[Array, ''],
    c: Float[Array, ''],
    max_degree: int,
    cartesian_order: bool,
) -> Float[Array, '(max_degree+1)**2 (max_degree+1)**2']:
  """Wigner-D matrix from Euler angles a, b, c."""
  with jax.ensure_compile_time_eval():
    # Calculate scaled Clebsch-Gordan coefficients.
    degrees = jnp.arange(max_degree + 1)
    factors = jnp.sqrt(degrees * (degrees + 1))
    factors = jnp.repeat(
        factors,
        repeats=2 * degrees + 1,
        total_repeat_length=(max_degree + 1) ** 2,
    )
    cg = (
        factors
        * clebsch_gordan(
            1, max_degree, max_degree, cartesian_order=cartesian_order
        )[1:]
    )
    # Determine which entries correspond to x-, y-, and z-axes.
    if cartesian_order:
      x, y, z = 0, 1, 2
    else:
      x, y, z = 2, 0, 1
    # Initialize Wigner-D matrix.
    dmat = jnp.zeros_like(
        a, shape=((max_degree + 1) ** 2, (max_degree + 1) ** 2)
    )
    dmat = dmat.at[0, 0].set(1.0)
  # Calculate entries of Wigner-D matrix.
  for l in range(1, max_degree + 1):
    i = l**2
    j = (l + 1) ** 2
    dx = jax.scipy.linalg.expm(a * cg[x, i:j, i:j])
    dy = jax.scipy.linalg.expm(b * cg[y, i:j, i:j])
    dz = jax.scipy.linalg.expm(c * cg[z, i:j, i:j])
    dmat = dmat.at[i:j, i:j].set(dx @ dy @ dz)
  return dmat


def wigner_d(
    rot: Float[Array, '... 3 3'],
    max_degree: int,
    cartesian_order: bool = Config.cartesian_order,
) -> Float[Array, '... (max_degree+1)**2 (max_degree+1)**2']:
  r"""Wigner-D matrix corresponding to a given :math:`3\times3` rotation matrix.

  Transform :math:`3\times3` rotation matrices to
  :math:`(\mathrm{max\_degree}+1)^2 \times (\mathrm{max\_degree}+1)^2` Wigner-D
  matrices that can be used to rotate irreducible representations of
  :math:`\mathrm{SO}(3)`.

  Example:
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import e3x
    >>> r = jnp.asarray([0.3, -1.4, 0.7])
    >>> rot = e3x.so3.random_rotation(jax.random.PRNGKey(0))
    >>> wigner_d = e3x.so3.wigner_d(rot, max_degree=2)
    >>> jnp.allclose(
    ... e3x.so3.spherical_harmonics(r, max_degree=2) @ wigner_d,
    ... e3x.so3.spherical_harmonics(r @ rot, max_degree=2),
    ... atol=1e-5)
    Array(True, dtype=bool)

  Args:
    rot: An Array of shape :math:`(\dots, 3, 3)` representing :math:`3\times3`
      rotation matrices.
    max_degree: Maximum degree of the irreducible representations.
    cartesian_order: If True, Cartesian order is assumed.

  Returns:
    An Array of shape
    :math:`(\dots, (\mathrm{max\_degree}+1)^2,(\mathrm{max\_degree}+1)^2)`
    representing Wigner-D matrices corresponding to the input rotations.

  Raises:
    ValueError: If ``rot`` does not have shape `(..., 3, 3)`.
  """
  _check_rotation_matrix_shape(rot)  # Raise if shape is not (..., 3, 3).

  with jax.ensure_compile_time_eval():
    # Load/Generate lookup table and convert to jax arrays.
    lookup_table = _generate_wigner_d_lookup_table(max_degree)
    cm = jnp.asarray(lookup_table['cm'])
    ls = jnp.asarray(lookup_table['ls'])
    # Optionally reorder to Cartesian order.
    if cartesian_order:
      cm = cm[:, _cartesian_permutation_wigner_d_entries(max_degree)]

  # Calculate all relevant monomials of the rotation matrix entries.
  # Note: This is done via integer powers and indexing on purpose! Using
  # jnp.power or the "**"-operator for this operation leads to NaNs in the
  # gradients for some inputs (jnp.power is not NaN-safe).
  rot_powers = _integer_powers(rot.reshape(*rot.shape[:-2], 1, -1), max_degree)
  monomials = (
      rot_powers[..., 0][..., ls[:, 0]]  #   R_00**l_00.
      * rot_powers[..., 1][..., ls[:, 1]]  # R_01**l_01.
      * rot_powers[..., 2][..., ls[:, 2]]  # R_02**l_02.
      * rot_powers[..., 3][..., ls[:, 3]]  # R_10**l_10.
      * rot_powers[..., 4][..., ls[:, 4]]  # R_11**l_11.
      * rot_powers[..., 5][..., ls[:, 5]]  # R_12**l_12.
      * rot_powers[..., 6][..., ls[:, 6]]  # R_20**l_20.
      * rot_powers[..., 7][..., ls[:, 7]]  # R_21**l_21.
      * rot_powers[..., 8][..., ls[:, 8]]  # R_22**l_22.
  )

  # Entries of the Wigner-D matrix are linear combinations of the monomials.
  dmat_entries = jnp.matmul(monomials, cm)

  # Assemble Wigner-D matrix.
  dmat = jnp.zeros_like(  # Initialize Wigner-D matrix to zeros.
      rot, shape=(*rot.shape[:-2], (max_degree + 1) ** 2, (max_degree + 1) ** 2)
  )
  for l in range(max_degree + 1):  # Set entries of non-zero blocks on diagonal.
    i = l**2  # Start index Wigner-D slice.
    j = (l + 1) ** 2  # Stop index Wigner-D slice.
    b = ((l + 1) * (2 * l + 1) * (2 * l + 3)) // 3  # Start index entries.
    a = b - (2 * l + 1) ** 2  # Stop index entries.
    num = 2 * l + 1  # Matrix block has shape (..., 2*l+1, 2*l+1).
    dmat = dmat.at[..., i:j, i:j].set(
        dmat_entries[..., a:b].reshape((*rot.shape[:-2], num, num))
    )
  return dmat
