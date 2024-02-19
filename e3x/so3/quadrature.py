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
from typing import Optional
import zipfile
from absl import logging
import jax.numpy as jnp
import jaxtyping
import numpy as np

Array = jaxtyping.Array
Float = jaxtyping.Float


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
  if (precision is None) == (num is None):
    raise ValueError(
        f'Exactly one of {precision=} or {num=} must be specified.'
    )

  try:
    f = io.BytesIO(pkgutil.get_data(__name__, '_lebedev_grids.npz'))
    with np.load(f) as lebedev_grids:
      if precision is not None:
        available_precisions = lebedev_grids['precision']
        if precision > np.max(available_precisions):
          i = available_precisions.size - 1
          logging.warning(
              (
                  'A Lebedev rule with precision=%s is not available, returning'
                  ' Lebedev rule with highest available precision=%s instead.'
              ),
              precision,
              available_precisions[i],
          )
        else:
          precision_difference = available_precisions - max(precision, 0)
          i = np.nanargmin(
              np.where(precision_difference < 0, np.nan, precision_difference)
          )
      else:  # num is not None.
        available_nums = lebedev_grids['num']
        if num < np.min(available_nums):
          i = 0
          logging.warning(
              (
                  'A Lebedev rule with num=%s points is not available,'
                  ' returning Lebedev rule with the lowest available number of'
                  ' points num=%s instead.'
              ),
              num,
              available_nums[i],
          )
        else:
          num_difference = max(num, 0) - available_nums
          i = np.nanargmin(np.where(num_difference < 0, np.nan, num_difference))
      r, w = lebedev_grids[f'r{i}'], lebedev_grids[f'w{i}']
  except (zipfile.BadZipFile, OSError, IOError, KeyError, ValueError) as exc:
    raise RuntimeError(
        f'failed to load Lebedev quadrature grids included with the  {__name__}'
        ' package (data may be corrupted), consider re-installing the package'
        ' to fix this problem'
    ) from exc

  return jnp.asarray(r), jnp.asarray(w)
