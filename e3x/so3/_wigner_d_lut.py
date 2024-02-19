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

"""Code for generating Wigner-D matrix lookup tables."""

import argparse
import multiprocessing as mp
from typing import Any, Dict, IO, Tuple, TypedDict, cast
from absl import logging
from etils import epath
import jaxtyping
import numpy as np
import sympy as sp

from ..config import Config
from ._common import _check_degree_is_positive_or_zero
from ._common import _number_of_rotation_matrix_monomials_of_degree
from ._common import _number_of_wigner_d_entries_of_degree
from ._common import _rotation_matrix_powers_of_degree
from ._common import _total_number_of_rotation_matrix_monomials
from ._common import _total_number_of_wigner_d_entries
from ._lookup_table_generation_utility import _load_lookup_table_from_disk
from ._lookup_table_generation_utility import _print_cache_usage_information
from ._lookup_table_generation_utility import _save_lookup_table_to_disk
from ._symbolic import _polynomial_dot_product
from ._symbolic import _rotate_xyz_polynomial
from ._symbolic import _spherical_harmonics
# pylint: enable=g-importing-member

Array = jaxtyping.Array
Float = jaxtyping.Float
Integer = jaxtyping.Integer


_wigner_d_lut_name = 'Wigner-D matrix'
_wigner_d_lut_path = '_wigner_d_lut.npz'


class WignerDLookupTable(TypedDict):
  """A lookup table with coefficients for computing Wigner-D matrices.

  Attributes:
    max_degree: Maximum degree of Wigner-D matrices for which coefficients are
      stored in the table.
    ls: Vector containing the powers for the rotation matrix entry monomials.
    cm: Coefficient matrix for computing the Wigner-D matrix entries by matrix
      multiplication with a vector containing rotation matrix entry monomials.
  """

  max_degree: int
  ls: Integer[Array, 'num_rotation_matrix_monomials 9']
  cm: Float[Array, 'num_rotation_matrix_monomials num_wigner_d_entries']


def _generate_wigner_d_lookup_table(
    max_degree: int, num_processes: int = 1
) -> WignerDLookupTable:
  """Generates a table with Wigner-D matrix coefficients."""

  _check_degree_is_positive_or_zero(max_degree)

  def _init_empty_lookup_table(
      max_degree: int,
  ) -> WignerDLookupTable:
    """Initializes a lookup table of the correct size containing only zeros."""
    num_rot = _total_number_of_rotation_matrix_monomials(max_degree)
    num_wig = _total_number_of_wigner_d_entries(max_degree)
    return WignerDLookupTable(
        max_degree=max_degree,
        cm=np.zeros((num_rot, num_wig), dtype=np.float64),
        ls=np.zeros((num_rot, 9), dtype=np.int64),
    )

  def _load_from_cache(
      f: IO[bytes],
  ) -> Tuple[int, WignerDLookupTable]:
    """Loads a (compressed) lookup table from the cache and uncompresses it."""
    lookup_table = _init_empty_lookup_table(max_degree)
    with np.load(f) as cache:
      cached_max_degree = cache['max_degree']
      if cached_max_degree < 0:  # Lookup table contains nothing.
        return -1, lookup_table
      irot = 0
      iwig = 0
      for l in range(min(cached_max_degree, max_degree) + 1):
        nrot = _number_of_rotation_matrix_monomials_of_degree(l)
        nwig = _number_of_wigner_d_entries_of_degree(l)
        cm_for_l = np.zeros((nrot, nwig), dtype=np.float64)
        cm_for_l[cache[f'i0{l}'], cache[f'i1{l}']] = cache[f'cm{l}']
        lookup_table['cm'][irot : irot + nrot, iwig : iwig + nwig] = cm_for_l
        lookup_table['ls'][irot : irot + nrot] = cache[f'ls{l}']
        irot += nrot
        iwig += nwig
      return cached_max_degree, lookup_table

  def _compress(lookup_table: WignerDLookupTable) -> Dict[str, Any]:
    """Compress a lookup table to store only non-zero entries in lists."""
    cache = {'max_degree': lookup_table['max_degree']}
    irot = 0
    iwig = 0
    for l in range(cache['max_degree'] + 1):
      nrot = _number_of_rotation_matrix_monomials_of_degree(l)
      nwig = _number_of_wigner_d_entries_of_degree(l)
      cm = lookup_table['cm'][irot : irot + nrot, iwig : iwig + nwig]
      i0, i1 = np.nonzero(cm)
      cache[f'cm{l}'] = cm[i0, i1]
      cache[f'i0{l}'] = i0
      cache[f'i1{l}'] = i1
      cache[f'ls{l}'] = lookup_table['ls'][irot : irot + nrot]
      irot += nrot
      iwig += nwig
    return cache

  # Load cache stored on disk.
  cached_max_degree, lookup_table = _load_lookup_table_from_disk(
      max_degree=max_degree,
      lookup_table_name=_wigner_d_lut_name,
      config_cache_path=Config.wigner_d_cache,
      package_cache_path=_wigner_d_lut_path,
      load_from_cache=_load_from_cache,
      init_empty_lookup_table=_init_empty_lookup_table,
  )
  lookup_table = cast(WignerDLookupTable, lookup_table)

  # Return immediately if all values are contained.
  if max_degree <= cached_max_degree:
    return lookup_table

  lstart = cached_max_degree + 1  # Start generation from degree=lstart.

  # Inform user that it might be preferable to cache the results.
  _print_cache_usage_information(
      lstart=lstart,
      max_degree=max_degree,
      config_cache_path=Config.wigner_d_cache,
      set_cache_method_name='set_wigner_d_cache',
      lookup_table_name=_wigner_d_lut_name,
      pregeneration_name=__name__,
  )

  # Calculate all combinations of degrees and orders.
  degrees_and_orders = []
  for l in range(lstart, max_degree + 1):
    for m in range(-l, l + 1):
      degrees_and_orders.append((l, m))

  def _construct_polynomial_pairs(sph_polynomials, rot_polynomials):
    """Helper function to create pairs of (un)rotated polynomials."""
    poly_pairs = []
    for l in range(lstart, max_degree + 1):
      offset = l**2 + l - lstart**2
      for mrot in range(-l, l + 1):
        irot = offset + mrot
        for msph in range(-l, l + 1):
          isph = offset + msph
          poly_pairs.append((sph_polynomials[isph], rot_polynomials[irot]))
    return poly_pairs

  # Calculate Wigner-D entries.
  if num_processes > 1:  # Use multiple processes in parallel.
    with mp.Pool(num_processes) as pool:
      sph_polynomials = pool.starmap(_spherical_harmonics, degrees_and_orders)
      rot_polynomials = pool.map(_rotate_xyz_polynomial, sph_polynomials)
      poly_pairs = _construct_polynomial_pairs(sph_polynomials, rot_polynomials)
      wigner_d_entries = pool.starmap(_polynomial_dot_product, poly_pairs)
  else:  # Sequential computation.
    sph_polynomials = [
        _spherical_harmonics(*args) for args in degrees_and_orders
    ]
    rot_polynomials = [_rotate_xyz_polynomial(poly) for poly in sph_polynomials]
    poly_pairs = _construct_polynomial_pairs(sph_polynomials, rot_polynomials)
    wigner_d_entries = [_polynomial_dot_product(*args) for args in poly_pairs]

  # Create index mapping for the rotation matrix monomials and store
  # corresponding powers.
  if lstart > 0:
    idx = _total_number_of_rotation_matrix_monomials(lstart - 1)
  else:
    idx = 0
  monomial_map = {}
  for l in range(lstart, max_degree + 1):
    for powers in _rotation_matrix_powers_of_degree(l):
      monomial_map[powers] = idx
      lookup_table['ls'][idx] = np.asarray(powers, dtype=np.int64)
      idx += 1

  # Store results in lookup table.
  if lstart > 0:
    offset = _total_number_of_wigner_d_entries(lstart - 1)
  else:
    offset = 0
  for i, polynomial in enumerate(wigner_d_entries):
    iwig = i + offset
    for monomial, coefficient in polynomial.terms():
      irot = monomial_map[monomial]
      lookup_table['cm'][irot, iwig] = sp.simplify(coefficient)

  # Save lookup table to disk cache.
  _save_lookup_table_to_disk(
      lookup_table=_compress(lookup_table),
      lookup_table_name=_wigner_d_lut_name,
      config_cache_path=Config.wigner_d_cache,
  )

  return lookup_table


if __name__ == '__main__':
  mp.freeze_support()  # Might be necessary for Windows support.
  parser = argparse.ArgumentParser(
      description='Generates lookup tables for computing Wigner-D matrices.'
  )
  parser.add_argument(
      '--max_degree',
      required=True,
      type=int,
      help='Maximum degree of the Wigner-D matrices.',
  )
  parser.add_argument(
      '--path',
      required=False,
      type=str,
      default=epath.Path(__file__).parent / _wigner_d_lut_path,
      help='Path to .npz file for storing the lookup table.',
  )
  parser.add_argument(
      '--num_processes',
      required=False,
      type=int,
      default=mp.cpu_count(),
      help='Number of processes for parallel computation.',
  )
  args = parser.parse_args()
  logging.set_verbosity(logging.INFO)
  Config.set_wigner_d_cache(args.path)
  _generate_wigner_d_lookup_table(
      max_degree=args.max_degree, num_processes=args.num_processes
  )
