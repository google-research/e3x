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

"""Code for generating Clebsch-Gordan lookup tables."""

import argparse
import multiprocessing as mp
from typing import IO, Tuple, TypedDict, cast
from absl import logging
from etils import epath
import jaxtyping
import numpy as np

from ..config import Config
from ._common import _check_degree_is_positive_or_zero
from ._common import _total_number_of_spherical_harmonics
from ._lookup_table_generation_utility import _load_lookup_table_from_disk
from ._lookup_table_generation_utility import _print_cache_usage_information
from ._lookup_table_generation_utility import _save_lookup_table_to_disk
from ._symbolic import _clebsch_gordan
# pylint: enable=g-importing-member

Array = jaxtyping.Array
Float = jaxtyping.Float
Integer = jaxtyping.Integer


_clebsch_gordan_lut_name = 'Clebsch-Gordan'
_clebsch_gordan_lut_path = '_clebsch_gordan_lut.npz'


class ClebschGordanLookupTable(TypedDict):
  """A lookup table with Clebsch-Gordan coefficients.

  Attributes:
    max_degree: Maximum degree of spherical harmonics for which Clebsch-Gordan
      coefficients are stored in the table.
    cg: Array storing the Clebsch-Gordan coefficients.
  """

  max_degree: int
  cg: Float[Array, '(max_degree+1)**2 (max_degree+1)**2 (max_degree+1)**2']


class _CompressedLookupTable(TypedDict):
  """Stores only non-zero entries of the lookup table.

  Attributes:
    max_degree: Maximum degree of spherical harmonics for which Clebsch-Gordan
      coefficients are stored in the table.
    cg: Compressed array storing the non-zero Clebsch-Gordan coefficients.
    i0: Index array used for uncompressing (first dimension).
    i1: Index array used for uncompressing (second dimension).
    i2: Index array used for uncompressing (third dimension).
  """

  max_degree: int  # Abbreviated as L in array dimensions.
  cg: Float[Array, 'num_nonzero']
  i0: Integer[Array, 'num_nonzero']
  i1: Integer[Array, 'num_nonzero']
  i2: Integer[Array, 'num_nonzero']


def _generate_clebsch_gordan_lookup_table(
    max_degree: int, num_processes: int = 1
) -> ClebschGordanLookupTable:
  """Generates a table with Clebsch-Gordan coefficients."""

  _check_degree_is_positive_or_zero(max_degree)

  def _init_empty_lookup_table(max_degree: int) -> ClebschGordanLookupTable:
    """Initializes a lookup table of the correct size containing only zeros."""
    num_sph = _total_number_of_spherical_harmonics(max_degree)
    return ClebschGordanLookupTable(
        max_degree=max_degree,
        cg=np.zeros((num_sph, num_sph, num_sph), dtype=np.float64),
    )

  def _load_from_cache(f: IO[bytes]) -> Tuple[int, ClebschGordanLookupTable]:
    """Loads a (compressed) lookup table from the cache and uncompresses it."""
    with np.load(f) as cache:
      cached_max_degree = cache['max_degree']
      if cached_max_degree < 0:  # Lookup table contains nothing.
        return -1, _init_empty_lookup_table(max_degree)
      cached_num_sph = _total_number_of_spherical_harmonics(cached_max_degree)
      cg = np.zeros(
          (cached_num_sph, cached_num_sph, cached_num_sph), dtype=np.float64
      )
      cg[cache['i0'], cache['i1'], cache['i2']] = cache['cg']
      if max_degree <= cached_max_degree:  # All necessary values exist.
        num_sph = _total_number_of_spherical_harmonics(max_degree)
        return cached_max_degree, ClebschGordanLookupTable(
            max_degree=max_degree, cg=cg[:num_sph, :num_sph, :num_sph]
        )
      else:  # Necessary values exist only partially.
        # Return partially filled lookup table.
        lookup_table = _init_empty_lookup_table(max_degree)
        lookup_table['cg'][
            :cached_num_sph, :cached_num_sph, :cached_num_sph
        ] = cg
        return cached_max_degree, lookup_table

  def _compress(
      lookup_table: ClebschGordanLookupTable,
  ) -> _CompressedLookupTable:
    """Compress a lookup table to store only non-zero entries."""
    i0, i1, i2 = np.nonzero(lookup_table['cg'])
    return _CompressedLookupTable(
        max_degree=lookup_table['max_degree'],
        cg=lookup_table['cg'][i0, i1, i2],
        i0=i0,
        i1=i1,
        i2=i2,
    )

  # Load cache stored on disk.
  cached_max_degree, lookup_table = _load_lookup_table_from_disk(
      max_degree=max_degree,
      lookup_table_name=_clebsch_gordan_lut_name,
      config_cache_path=Config.clebsch_gordan_cache,
      package_cache_path=_clebsch_gordan_lut_path,
      load_from_cache=_load_from_cache,
      init_empty_lookup_table=_init_empty_lookup_table,
  )
  lookup_table = cast(ClebschGordanLookupTable, lookup_table)

  # Return immediately if all values are contained.
  if max_degree <= cached_max_degree:
    return lookup_table

  lstart = cached_max_degree + 1  # Start generation from degree=lstart.

  # Inform user that it might be preferable to cache the results.
  _print_cache_usage_information(
      lstart=lstart,
      max_degree=max_degree,
      config_cache_path=Config.clebsch_gordan_cache,
      set_cache_method_name='set_clebsch_gordan_cache',
      lookup_table_name=_clebsch_gordan_lut_name,
      pregeneration_name=__name__,
  )

  # Calculate all combinations of degrees and orders.
  degrees_and_orders = []
  for l1 in range(max_degree + 1):
    for l2 in range(max_degree + 1):
      # Valid l3 range from abs(l1-l2) to min(l1+l2, max_degree).
      for l3 in range(abs(l1 - l2), min(l1 + l2, max_degree) + 1):
        if l1 < lstart and l2 < lstart and l3 < lstart:
          continue
        for m1 in range(-l1, l1 + 1):
          for m2 in range(-l2, l2 + 1):
            # Only a few m3 can be nonzero.
            for m3 in (m1 + m2, m1 - m2, m2 - m1, -m1 - m2):
              if abs(m3) > l3:  # invalid m
                continue
              degrees_and_orders.append((l1, m1, l2, m2, l3, m3))

  # Calculate Clebsch-Gordan coefficients.
  if num_processes > 1:  # Use multiple processes in parallel.
    with mp.Pool(num_processes) as pool:
      cg_coefficients = pool.starmap(_clebsch_gordan, degrees_and_orders)
  else:  # Sequential computation.
    cg_coefficients = [_clebsch_gordan(*args) for args in degrees_and_orders]

  # Store results in lookup table.
  for (l1, m1, l2, m2, l3, m3), cg in zip(degrees_and_orders, cg_coefficients):
    lookup_table['cg'][
        l1 * l1 + l1 + m1, l2 * l2 + l2 + m2, l3 * l3 + l3 + m3
    ] = cg

  # Save lookup table to disk cache.
  _save_lookup_table_to_disk(
      lookup_table=_compress(lookup_table),
      lookup_table_name=_clebsch_gordan_lut_name,
      config_cache_path=Config.clebsch_gordan_cache,
  )

  return lookup_table


if __name__ == '__main__':
  mp.freeze_support()  # Might be necessary for Windows support.
  parser = argparse.ArgumentParser(
      description='Generates lookup tables with Clebsch-Gordan coefficients.'
  )
  parser.add_argument(
      '--max_degree',
      required=True,
      type=int,
      help='Maximum degree of the Clebsch-Gordan coefficients.',
  )
  parser.add_argument(
      '--path',
      required=False,
      type=str,
      default=epath.Path(__file__).parent / _clebsch_gordan_lut_path,
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
  Config.set_clebsch_gordan_cache(args.path)
  _generate_clebsch_gordan_lookup_table(
      max_degree=args.max_degree, num_processes=args.num_processes
  )
