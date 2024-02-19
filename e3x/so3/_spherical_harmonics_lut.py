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

"""Code for generating spherical harmonics lookup tables."""

import argparse
import multiprocessing as mp
from typing import IO, Tuple, TypedDict, cast
from absl import logging
from etils import epath
import jaxtyping
import numpy as np
import sympy as sp

from ..config import Config
from ._common import _check_degree_is_positive_or_zero
from ._common import _monomial_powers_of_degree
from ._common import _total_number_of_cartesian_monomials
from ._common import _total_number_of_spherical_harmonics
from ._lookup_table_generation_utility import _load_lookup_table_from_disk
from ._lookup_table_generation_utility import _print_cache_usage_information
from ._lookup_table_generation_utility import _save_lookup_table_to_disk
from ._symbolic import _spherical_harmonics
# pylint: enable=g-importing-member

Array = jaxtyping.Array
Float = jaxtyping.Float
Integer = jaxtyping.Integer


_spherical_harmonics_lut_name = 'spherical harmonics'
_spherical_harmonics_lut_path = '_spherical_harmonics_lut.npz'


class SphericalHarmonicsLookupTable(TypedDict):
  """A lookup table with coefficients for computing spherical harmonics.

  Attributes:
    max_degree: Maximum degree of spherical harmonics for which coefficients are
      stored in the table.
    ls: Vector containing the powers for the (x, y, z)-components of the
      Cartesian monomials.
    cm: Coefficient matrix for computing the spherical harmonics by matrix
      multiplication with a vector containing Cartesian monomials.
  """

  max_degree: int
  ls: Integer[Array, '(max_degree+1)*(max_degree+2)*(max_degree+3)/6 3']
  cm: Float[
      Array, '(max_degree+1)*(max_degree+2)*(max_degree+3)/6 (max_degree+1)**2'
  ]


class _CompressedLookupTable(TypedDict):
  """Stores only non-zero entries of the lookup table.

  Attributes:
    max_degree: Maximum degree of spherical harmonics for which coefficients are
      stored in the table.
    ls: Vector containing the powers for the (x, y, z)-components of the
      Cartesian monomials.
    cm: Compressed coefficient matrix.
    i0: Index array used for uncompressing (first dimension).
    i1: Index array used for uncompressing (second dimension).
  """

  max_degree: int  # Abbreviated as L in array dimensions.
  ls: Integer[Array, '(max_degree+1)*(max_degree+2)*(max_degree+3)/6 3']
  cm: Float[Array, 'num_nonzero']
  i0: Integer[Array, 'num_nonzero']
  i1: Integer[Array, 'num_nonzero']


def _generate_spherical_harmonics_lookup_table(
    max_degree: int, num_processes: int = 1
) -> SphericalHarmonicsLookupTable:
  """Generates a table with coefficients for the spherical harmonics."""

  _check_degree_is_positive_or_zero(max_degree)

  def _init_empty_lookup_table(
      max_degree: int,
  ) -> SphericalHarmonicsLookupTable:
    """Initializes a lookup table of the correct size containing only zeros."""
    num_car = _total_number_of_cartesian_monomials(max_degree)
    num_sph = _total_number_of_spherical_harmonics(max_degree)
    return SphericalHarmonicsLookupTable(
        max_degree=max_degree,
        cm=np.zeros((num_car, num_sph), dtype=np.float64),
        ls=np.zeros((num_car, 3), dtype=np.int64),
    )

  def _load_from_cache(
      f: IO[bytes],
  ) -> Tuple[int, SphericalHarmonicsLookupTable]:
    """Loads a (compressed) lookup table from the cache and uncompresses it."""
    with np.load(f) as cache:
      cached_max_degree = cache['max_degree']
      if cached_max_degree < 0:  # Lookup table contains nothing.
        return -1, _init_empty_lookup_table(max_degree)
      cached_num_car = _total_number_of_cartesian_monomials(cached_max_degree)
      cached_num_sph = _total_number_of_spherical_harmonics(cached_max_degree)
      cm = np.zeros((cached_num_car, cached_num_sph), dtype=np.float64)
      cm[cache['i0'], cache['i1']] = cache['cm']
      if max_degree <= cached_max_degree:  # All necessary values exist.
        num_car = _total_number_of_cartesian_monomials(max_degree)
        num_sph = _total_number_of_spherical_harmonics(max_degree)
        return cached_max_degree, SphericalHarmonicsLookupTable(
            max_degree=max_degree,
            cm=cm[:num_car, :num_sph],
            ls=cache['ls'][:num_car],
        )
      else:  # Necessary values exist only partially.
        # Initialize partially filled lookup table.
        lookup_table = _init_empty_lookup_table(max_degree)
        lookup_table['cm'][:cached_num_car, :cached_num_sph] = cm
        lookup_table['ls'][:cached_num_car] = cache['ls']
        return cached_max_degree, lookup_table

  def _compress(
      lookup_table: SphericalHarmonicsLookupTable,
  ) -> _CompressedLookupTable:
    """Compress a lookup table to store only non-zero entries."""
    i0, i1 = np.nonzero(lookup_table['cm'])
    return _CompressedLookupTable(
        max_degree=lookup_table['max_degree'],
        ls=lookup_table['ls'],
        cm=lookup_table['cm'][i0, i1],
        i0=i0,
        i1=i1,
    )

  # Load cache stored on disk.
  cached_max_degree, lookup_table = _load_lookup_table_from_disk(
      max_degree=max_degree,
      lookup_table_name=_spherical_harmonics_lut_name,
      config_cache_path=Config.spherical_harmonics_cache,
      package_cache_path=_spherical_harmonics_lut_path,
      load_from_cache=_load_from_cache,
      init_empty_lookup_table=_init_empty_lookup_table,
  )
  lookup_table = cast(SphericalHarmonicsLookupTable, lookup_table)

  # Return immediately if all values are contained.
  if max_degree <= cached_max_degree:
    return lookup_table

  lstart = cached_max_degree + 1  # Start generation from degree=lstart.

  # Inform user that it might be preferable to cache the results.
  _print_cache_usage_information(
      lstart=lstart,
      max_degree=max_degree,
      config_cache_path=Config.spherical_harmonics_cache,
      set_cache_method_name='set_spherical_harmonics_cache',
      lookup_table_name=_spherical_harmonics_lut_name,
      pregeneration_name=__name__,
  )

  # Create index mapping for the monomials and store corresponding powers.
  idx = _total_number_of_cartesian_monomials(lstart - 1) if lstart > 0 else 0
  monomial_map = {}
  for l in range(lstart, max_degree + 1):
    for powers in _monomial_powers_of_degree(l):
      monomial_map[powers] = idx
      lookup_table['ls'][idx] = np.asarray(powers, dtype=int)
      idx += 1

  # Calculate all combinations of degrees and orders.
  degrees_and_orders = []
  for l in range(lstart, max_degree + 1):
    for m in range(-l, l + 1):
      degrees_and_orders.append((l, m))

  # Calculate spherical harmonics polynomials.
  if num_processes > 1:  # Use multiple processes in parallel.
    with mp.Pool(num_processes) as pool:
      sph_polynomials = pool.starmap(_spherical_harmonics, degrees_and_orders)
  else:  # Sequential computation.
    sph_polynomials = [
        _spherical_harmonics(*args) for args in degrees_and_orders
    ]

  # Store results in lookup table.
  for (l, m), polynomial in zip(degrees_and_orders, sph_polynomials):
    isph = l**2 + l + m
    for monomial, coefficient in polynomial.terms():
      icar = monomial_map[monomial]
      lookup_table['cm'][icar, isph] = sp.simplify(coefficient)

  # Save lookup table to disk cache.
  _save_lookup_table_to_disk(
      lookup_table=_compress(lookup_table),
      lookup_table_name=_spherical_harmonics_lut_name,
      config_cache_path=Config.spherical_harmonics_cache,
  )

  return lookup_table


if __name__ == '__main__':
  mp.freeze_support()  # Might be necessary for Windows support.
  parser = argparse.ArgumentParser(
      description='Generates lookup tables for computing spherical harmonics.'
  )
  parser.add_argument(
      '--max_degree',
      required=True,
      type=int,
      help='Maximum degree of the spherical harmonics.',
  )
  parser.add_argument(
      '--path',
      required=False,
      type=str,
      default=epath.Path(__file__).parent / _spherical_harmonics_lut_path,
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
  Config.set_spherical_harmonics_cache(args.path)
  _generate_spherical_harmonics_lookup_table(
      max_degree=args.max_degree, num_processes=args.num_processes
  )
