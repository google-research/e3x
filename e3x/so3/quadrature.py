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

r"""Quadrature rules for approximating surface integrals of the unit sphere."""

import io
import pkgutil
from typing import Literal, Optional
import zipfile
from absl import logging
import jax.numpy as jnp
import jaxtyping
import numpy as np

Array = jaxtyping.Array
Float = jaxtyping.Float


def _load_grid(
    kind: Literal['Lebedev', 'Delley'],
    precision: Optional[int] = None,
    num: Optional[int] = None,
) -> tuple[Float[Array, 'num_points 3'], Float[Array, 'num_points']]:
  """Loads a quadrature grid from disk."""
  if kind not in ('Lebedev', 'Delley'):
    raise ValueError(
        f"Quadrature grid with {kind=} does not exist, choose from ('Lebedev',"
        " 'Delley')."
    )
  if (precision is None) == (num is None):
    raise ValueError(
        f'Exactly one of {precision=} or {num=} must be specified.'
    )

  try:
    f = io.BytesIO(pkgutil.get_data(__name__, f'_{kind.lower()}_grids.npz'))
    with np.load(f) as quadrature_grids:
      if precision is not None:
        available_precisions = quadrature_grids['precision']
        if precision > np.max(available_precisions):
          i = available_precisions.size - 1
          logging.warning(
              (
                  'A %s rule with precision=%s is not available, returning'
                  ' %s rule with highest available precision=%s instead.'
              ),
              kind,
              precision,
              kind,
              available_precisions[i],
          )
        else:
          precision_difference = available_precisions - max(precision, 0)
          i = np.nanargmin(
              np.where(precision_difference < 0, np.nan, precision_difference)
          )
      else:  # num is not None.
        available_nums = quadrature_grids['num']
        if num < np.min(available_nums):
          i = 0
          logging.warning(
              (
                  'A %s rule with num=%s points is not available,'
                  ' returning %s rule with the lowest available number of'
                  ' points num=%s instead.'
              ),
              kind,
              num,
              kind,
              available_nums[i],
          )
        else:
          num_difference = max(num, 0) - available_nums
          i = np.nanargmin(np.where(num_difference < 0, np.nan, num_difference))
      return (
          jnp.asarray(quadrature_grids[f'r{i}']),
          jnp.asarray(quadrature_grids[f'w{i}']),
      )
  except (zipfile.BadZipFile, OSError, IOError, KeyError, ValueError) as exc:
    raise RuntimeError(
        f'failed to load {kind} quadrature grids included with the  {__name__}'
        ' package (data may be corrupted), consider re-installing the package'
        ' to fix this problem'
    ) from exc


def lebedev_quadrature(
    precision: Optional[int] = None, num: Optional[int] = None
) -> tuple[Float[Array, 'num_points 3'], Float[Array, 'num_points']]:
  r"""Returns a Lebedev quadrature grid.

  A Lebedev quadrature is a numerical approximation to the surface integral of
  a function :math:`f` over the unit sphere

  .. math::
    \int_{\Omega} f(\Omega) d\Omega \approx 4\pi \sum_{i=1}^N w_i f(\vec{r}_i)

  where :math:`\vec{r}_i` and :math:`w_i` are grid points and weights,
  respectively. A Lebedev rule of precision :math:`p` can be used to correctly
  integrate any polynomial for which the highest degree term :math:`x^iy^jz^k`
  satisfies :math:`i+j+k \leq p`.

  Args:
    precision: The desired minimum precision :math:`p`. The returned Lebedev
      rule will be the smallest grid that has the desired precision (if this
      value is not ``None``, ``num`` must be set to ``None``).
    num: The desired maximum number of points :math:`N`. The returned Lebedev
      rule will be the highest-precision grid that has not more than ``num``
      points (if this value is not ``None``, ``precision`` must be set to
      ``None``).

  Returns:
    A tuple of two arrays representing the grid points :math:`\vec{r}_i` and
    weights :math:`w_i`.

  Raises:
    ValueError:
      If both ``precision`` and ``num`` are ``None`` (or both are specified at
      the same time).
    RuntimeError:
      If the Lebedev quadrature rules cannot be loaded from disk.
  """
  return _load_grid(kind='Lebedev', precision=precision, num=num)


def delley_quadrature(
    precision: Optional[int] = None, num: Optional[int] = None
) -> tuple[Float[Array, 'num_points 3'], Float[Array, 'num_points']]:
  r"""Returns a Delley quadrature grid.

  A Delley quadrature is a numerical approximation to the surface integral of
  a function :math:`f` over the unit sphere

  .. math::
    \int_{\Omega} f(\Omega) d\Omega \approx 4\pi \sum_{i=1}^N w_i f(\vec{r}_i)

  where :math:`\vec{r}_i` and :math:`w_i` are grid points and weights,
  respectively. A Delley rule of precision :math:`p` can be used to correctly
  integrate any polynomial for which the highest degree term :math:`x^iy^jz^k`
  satisfies :math:`i+j+k \leq p`. The Delley rules are an optimized version of
  the Lebedev rules with improved numerical precision, for details, see
  Delley, Bernard. "High order integration schemes on the unit sphere."
  Journal of Computational Chemistry 17.9 (1996): 1152-1155.

  Args:
    precision: The desired minimum precision :math:`p`. The returned Delley rule
      will be the smallest grid that has the desired precision (if this value is
      not ``None``, ``num`` must be set to ``None``).
    num: The desired maximum number of points :math:`N`. The returned Delley
      rule will be the highest-precision grid that has not more than ``num``
      points (if this value is not ``None``, ``precision`` must be set to
      ``None``).

  Returns:
    A tuple of two arrays representing the grid points :math:`\vec{r}_i` and
    weights :math:`w_i`.

  Raises:
    ValueError:
      If both ``precision`` and ``num`` are ``None`` (or both are specified at
      the same time).
    RuntimeError:
      If the Delley quadrature rules cannot be loaded from disk.
  """
  return _load_grid(kind='Delley', precision=precision, num=num)
