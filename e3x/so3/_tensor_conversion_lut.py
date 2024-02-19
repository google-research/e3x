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

"""Code for generating tensor conversion lookup tables."""

import argparse
import functools
import itertools
import multiprocessing as mp
import operator
from typing import Any, Dict, IO, List, Tuple, TypedDict, cast
from absl import logging
from etils import epath
import jaxtyping
import numpy as np
import sympy as sp

from ..config import Config
from ._common import _check_degree_is_positive_or_zero
from ._common import _monomial_powers_of_degree
from ._common import _number_of_cartesian_monomials_of_degree
from ._common import _number_of_spherical_harmonics_of_degree
from ._lookup_table_generation_utility import _load_lookup_table_from_disk
from ._lookup_table_generation_utility import _print_cache_usage_information
from ._lookup_table_generation_utility import _save_lookup_table_to_disk
from ._symbolic import _spherical_harmonics
# pylint: enable=g-importing-member

Array = jaxtyping.Array
Float = jaxtyping.Float


_tensor_conversion_lut_name = 'tensor conversion'
_tensor_conversion_lut_path = '_tensor_conversion_lut.npz'


def _traceless_tensor(n: int) -> Float[Array, '...']:
  """Traceless tensor 𝒯ₙr⁽ⁿ⁾ with degree n.

  Computes the traceless tensor 𝒯ₙr⁽ⁿ⁾ using the detracing projection operator
  𝒯ₙ defined in Eq. (5.1) of:
  Applequist, J. (1989). "Traceless cartesian tensor forms for spherical
  harmonic functions: new theorems and applications to electrostatics of
  dielectric media." Journal of Physics A: Mathematical and General, 22(20),
  4303.

  Args:
    n: Degree of the traceless tensor.

  Returns:
    An array containing symbolic expressions for the components of the traceless
    tensor 𝒯ₙr⁽ⁿ⁾ of degree n. The tensor is returned in compressed format, i.e.
    a flattened view of the (n+1)*(n+2)/2 unique components is returned.
  """

  # Degree 0 is trivial, since a 0-th degree tensor is just a scalar.
  if n == 0:
    return np.asarray([1], dtype=object)

  # Degree 1 is also trivial (vectors have no trace).
  x, y, z = sp.symbols('x y z')
  r = np.asarray([x, y, z], dtype=object)
  if n == 1:
    return r

  # Construct traced tensor of degree n by repeated outer products.
  a = r.copy()
  for _ in range(2, n + 1):
    r = np.expand_dims(r, axis=-2)
    a = np.expand_dims(a, axis=-1) * r

  def multi_dirac_delta(alphas):
    """Returns the value of a sum of products of dirac delta combinations.

    Given an even number of indices [i1, i2, ..., iN], this function returns the
    sum over the product of dirac deltas with all unique pairs chosen from the
    indices. For example:

    [i, j] -> δ(i, j)

    [i, j, k, l] -> δ(i, j) * δ(k, l) +
                    δ(i, k) * δ(j, l) +
                    δ(i, l) * δ(j, k)

    Args:
      alphas: Input indices [i1, i2, ..., iN] (N must be even).
    """
    assert len(alphas) % 2 == 0  # Should never be called for odd lengths.
    _, counts = np.unique(alphas, return_counts=True)
    if np.any(np.mod(counts, 2) == 1):  # At least one δ in each product is 0.
      return 0
    return functools.reduce(
        operator.mul, [sp.factorial2(i) for i in counts - 1]
    )

  # Implementation of the detracing operator 𝒯ₙ.
  b = a.copy()  # Initialize traceless tensor (corresponds to m=0 iteration).
  trace = a.copy()  # Stores the current (m-fold) trace.
  for m in range(1, n // 2 + 1):  # Iteration for m=0 is already handled.
    # All traces of totally symmetric tensors are identical, so the last two
    # dimensions are always chosen here without loss of generality.
    trace = np.asarray(np.trace(trace, axis1=-2, axis2=-1))

    # Calculate the fixed part of the normalization prefactor. Note: The
    # constant used here differs from the one used in the Applequist paper by a
    # factor of 1/(2n-1)!!. This is done so that the detracing operator becomes
    # a true projection operator, see the note at the end of section 5.1.
    # "The detracer theorem".
    normalization_ratio = (-1) ** m * (
        sp.factorial2(2 * n - 1 - 2 * m) / sp.factorial2(2 * n - 1)
    )

    # This loop runs only over those combinations of indices alpha which
    # correspond to unique entries in the tensor (since the tensor is returned
    # in compressed format anyway, the other entries are not needed).
    for alphas_as_tuple in itertools.combinations_with_replacement(
        (0, 1, 2), r=n
    ):
      alphas = np.asarray(alphas_as_tuple)
      for index_combination_as_tuple in itertools.combinations(range(n), 2 * m):
        index_combination = np.asarray(index_combination_as_tuple)
        chosen_alphas = alphas[index_combination]
        leftover_alphas = np.delete(alphas, index_combination)
        prefactor = multi_dirac_delta(chosen_alphas) * normalization_ratio
        b[(..., *alphas)] += prefactor * trace[(..., *leftover_alphas)]

  # Return tensor in compressed format.
  indices = np.asarray(
      [i for i in itertools.combinations_with_replacement((0, 1, 2), r=n)]
  )
  indices = tuple(np.squeeze(i, axis=-1) for i in np.hsplit(indices, n))
  return b[indices]


def _tensor_conversion_matrices(l: int):
  """Coefficient matrices for converting tensors to spherical harmonics.

  The entries of symmetric traceless tensors 𝒯ₙr⁽ⁿ⁾ with degree n and the
  spherical harmonics of degree l=n are linear combinations of each other. This
  function calculates coefficient matrices for doing these transformations. The
  symmetric traceless tensors are assumed to be in compressed format, i.e. a
  flattened view of the (n+1)*(n+2)/2 unique components.

  Args:
    l: Degree of the traceless tensor and spherical harmonics.

  Returns:
    A tuple of two symbolic coefficient matrices for converting from spherical
    harmonics to traceless tensors and vice versa.
  """
  num_car = _number_of_cartesian_monomials_of_degree(l)
  num_sph = _number_of_spherical_harmonics_of_degree(l)

  # Generate monomial map and monomials.
  x, y, z = sp.symbols('x y z')
  monomials = sp.zeros(num_car, 1)
  idx = 0
  monomial_map = {}
  for a, b, c in _monomial_powers_of_degree(l):
    monomial_map[(a, b, c)] = idx
    monomials[idx] = x**a * y**b * z**c
    idx += 1

  # Generate coefficient matrix for converting monomials to spherical
  # harmonics.
  m2s = sp.zeros(num_car, num_sph)
  for isph in range(0, num_sph):
    ylm = _spherical_harmonics(l, isph - l)
    for monomial_index, coefficient in ylm.terms():
      icar = monomial_map[monomial_index]
      m2s[icar, isph] = sp.simplify(coefficient)

  # Generate coefficient matrix for converting monomials to traceless tensors.
  traceless_tensor = _traceless_tensor(l)
  m2t = sp.zeros(num_car, num_car)
  for i, component in enumerate(traceless_tensor):
    for monomial_index, coefficient in sp.Poly(component, x, y, z).terms():
      icar = monomial_map[monomial_index]
      m2t[icar, i] = sp.simplify(coefficient)

  # Conversion matrix from spherical harmonics to traceless tensors.
  s2t = sp.zeros(num_sph, num_car)
  for i in range(num_car):
    s2t[:, i] = sp.simplify(m2s.LUsolve(m2t[:, i]))

  # The back conversion matrix is given by the Moore-Penrose pseudoinverse.
  t2s = sp.simplify(s2t.T * (s2t * s2t.T) ** (-1))

  return s2t, t2s


class TensorConversionLookupTable(TypedDict):
  """A lookup table with coefficients for converting tensors.

  The entries of symmetric traceless tensors 𝒯ₙr⁽ⁿ⁾ with degree n and the
  spherical harmonics of degree l=n are linear combinations of each other. This
  table contains coefficient matrices for doing these transformations. The
  symmetric traceless tensors are assumed to be in compressed format, i.e. a
  flattened view of the (n+1)*(n+2)/2 unique components.

  Note: The lookup table contains lists of arrays (one entry for each degree).
  While it is possible to save lists as object arrays with numpy, loading them
  is only possible with allow_pickle=True. Since the pickle module is not
  secure, instead, the individual list entries are unpacked to separate
  dictionary entries (i.e. 't2s0', 't2s1', ... and 's2t0', 's2t1', ...) prior to
  saving to disk and packed into a list when they are loaded.

  Attributes:
    max_degree: Maximum degree of tensors/spherical harmonics for which
      conversion coefficients are stored in the table.
    s2t: List of arrays storing the conversion coefficients from spherical
      harmonics to tensors.
    t2s: List of arrays storing the conversion coefficients from tensors to
      spherical harmonics.
  """

  max_degree: int
  # The l in the array dimensions stands for the degree of the tensor/spherical
  # harmonics (of entry l in the list), which runs from 0 to max_degree.
  s2t: List[Float[Array, '2*l+1 (l+1)*(l+2)/2']]
  t2s: List[Float[Array, '(l+1)*(l+2)/2 2*l+1']]


def _generate_tensor_conversion_lookup_table(
    max_degree: int, num_processes: int = 1
) -> TensorConversionLookupTable:
  """Generates a table with tensor conversion coefficients."""

  _check_degree_is_positive_or_zero(max_degree)

  def _init_empty_lookup_table(max_degree: int) -> TensorConversionLookupTable:
    """Initializes an empty lookup table."""
    return TensorConversionLookupTable(max_degree=max_degree, t2s=[], s2t=[])

  def _load_from_cache(
      f: IO[bytes],
  ) -> Tuple[int, TensorConversionLookupTable]:
    """Loads a lookup table from the cache and packs entries into lists."""
    with np.load(f) as cache:
      cached_max_degree = cache['max_degree']
      lookup_table = _init_empty_lookup_table(max_degree)
      if cached_max_degree < 0:  # Lookup table contains nothing.
        return -1, lookup_table
      for l in range(min(max_degree, cached_max_degree) + 1):
        lookup_table['s2t'].append(cache[f's2t{l}'])
        lookup_table['t2s'].append(cache[f't2s{l}'])
      return cached_max_degree, lookup_table

  def _unpack_lists(
      lookup_table: TensorConversionLookupTable,
  ) -> Dict[str, Any]:
    """Unpacks lists and stores them as individual entries."""
    cache = {'max_degree': lookup_table['max_degree']}
    for l in range(lookup_table['max_degree'] + 1):
      cache[f's2t{l}'] = lookup_table['s2t'][l]
      cache[f't2s{l}'] = lookup_table['t2s'][l]
    return cache

  # Load cache stored on disk.
  cached_max_degree, lookup_table = _load_lookup_table_from_disk(
      max_degree=max_degree,
      lookup_table_name=_tensor_conversion_lut_name,
      config_cache_path=Config.tensor_conversion_cache,
      package_cache_path=_tensor_conversion_lut_path,
      load_from_cache=_load_from_cache,
      init_empty_lookup_table=_init_empty_lookup_table,
  )
  lookup_table = cast(TensorConversionLookupTable, lookup_table)

  # Return immediately if all values are contained.
  if max_degree <= cached_max_degree:
    return lookup_table

  lstart = cached_max_degree + 1  # Start generation from degree=lstart.

  # Inform user that it might be preferable to cache the results.
  _print_cache_usage_information(
      lstart=lstart,
      max_degree=max_degree,
      config_cache_path=Config.tensor_conversion_cache,
      set_cache_method_name='set_tensor_conversion_cache',
      lookup_table_name=_tensor_conversion_lut_name,
      pregeneration_name=__name__,
  )

  # Calculate conversion matrices.
  degrees = [(l,) for l in range(lstart, max_degree + 1)]
  if num_processes > 1:  # Use multiple processes in parallel.
    with mp.Pool(num_processes) as pool:
      s2t_and_t2s = pool.starmap(_tensor_conversion_matrices, degrees)
  else:  # Sequential computation.
    s2t_and_t2s = [_tensor_conversion_matrices(*args) for args in degrees]

  # Store results in lookup table.
  for s2t, t2s in s2t_and_t2s:
    lookup_table['s2t'].append(np.asarray(s2t, dtype=np.float64))
    lookup_table['t2s'].append(np.asarray(t2s, dtype=np.float64))

  # Save lookup table to disk cache.
  _save_lookup_table_to_disk(
      lookup_table=_unpack_lists(lookup_table),
      lookup_table_name=_tensor_conversion_lut_name,
      config_cache_path=Config.tensor_conversion_cache,
  )

  return lookup_table


if __name__ == '__main__':
  mp.freeze_support()  # Might be necessary for Windows support.
  parser = argparse.ArgumentParser(
      description=(
          'Generates lookup tables for converting symmetric traceless tensors'
          ' to spherical harmonics (and vice versa).'
      )
  )
  parser.add_argument(
      '--max_degree',
      required=True,
      type=int,
      help='Maximum degree of the spherical harmonics/tensors.',
  )
  parser.add_argument(
      '--path',
      required=False,
      type=str,
      default=epath.Path(__file__).parent / _tensor_conversion_lut_path,
      help='Path to .npz file for the lookup table.',
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
  Config.set_tensor_conversion_cache(args.path)
  _generate_tensor_conversion_lookup_table(
      max_degree=args.max_degree, num_processes=args.num_processes
  )
