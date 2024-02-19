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

"""Common utility code used to generate lookup tables."""

import io
import pkgutil
from typing import Any, Callable, Dict, IO, Optional, Tuple
import zipfile
from absl import logging
from etils import epath
import numpy as np


def _load_lookup_table_from_disk(
    max_degree: int,
    lookup_table_name: str,
    config_cache_path: epath.Path,
    package_cache_path: epath.PathLike,
    load_from_cache: Callable[[IO[bytes]], Tuple[int, Dict[str, Any]]],
    init_empty_lookup_table: Callable[[int], Dict[str, Any]],
) -> Tuple[int, Dict[str, Any]]:
  """Load a lookup table from disk.

  Args:
    max_degree: Maximum degree for which values should exist in the returned
      lookup table.
    lookup_table_name: Name of the lookup table (used only for printing error
      messages).
    config_cache_path: Path to a cached file on disk, which can be specified
      using the Config class.
    package_cache_path: Path to the cached file on disk included with the e3x
      package, which is used as a fallback if no config_cache_path is provided
      (or does not exist yet).
    load_from_cache: Function that reads the lookup table from a binary file
      handle and returns a tuple consisting of the stored max_degree and the
      loaded lookup table.
    init_empty_lookup_table: Function that returns an empty version of the
      lookup table (in case all attempts to load a cached version from disk
      fail).

  Returns:
    A tuple consisting of an int with the stored max_degree in the lookup table
    and the loaded lookup table itself.
  """
  # Load cached lookup table from path specified in Config.
  if config_cache_path is not None and config_cache_path.exists():
    with config_cache_path.open('rb') as f:
      return load_from_cache(f)
  else:  # Load pre-computed lookup table included with package.
    try:
      f = io.BytesIO(pkgutil.get_data(__name__, package_cache_path))
      return load_from_cache(f)
    except (zipfile.BadZipFile, OSError, IOError, KeyError, ValueError):
      logging.exception(
          (
              'Failed to load pre-computed %s lookup table included with the %s'
              ' package (data may be corrupted). Consider re-installing the'
              ' package to fix this problem.'
          ),
          lookup_table_name,
          __name__,
      )
      return -1, init_empty_lookup_table(max_degree)


def _print_cache_usage_information(
    lstart: int,
    max_degree: int,
    config_cache_path: Optional[epath.Path],
    set_cache_method_name: str,
    lookup_table_name: str,
    pregeneration_name: str,
) -> None:
  """Print information about the possibility to use disk caches.

  Calculating lookup tables for high degrees is very computationally intensive.
  The e3x library supports saving/loading lookup tables to disk, which is
  generally preferable compared to calculating them on the fly for each run.
  However, this feature must be explicitly enabled: This function prints an
  explanation of how to enable the disk caching feature.

  Args:
    lstart: Degree from which the generation of lookup table entries starts.
    max_degree: Maximum degree for which entries in the lookup table will be
      generated.
    config_cache_path: Path to a cached file on disk, which can be specified
      using the Config class.
    set_cache_method_name: Name of the method which can be used to enable disk
      caching for the given lookup table.
    lookup_table_name: Name of the lookup table.
    pregeneration_name: Name of the module that can be called for pregeneration.
  """
  if config_cache_path is None:
    logging.warning(
        (
            'Generating %s lookup table with values for degrees up to'
            ' max_degree=%d (starting from degree=%d). Calculating the %s'
            ' lookup table is computationally expensive and will be repeated on'
            ' re-runs. It might be preferable to cache the results on disk and'
            ' load (instead of re-generate) them on successive runs. To enable'
            " disk caching, call 'e3x.Config.%s(<path>)' at the start of your"
            ' program to specify a disk location for saving/loading the lookup'
            " table. It is recommended to run 'python -m %s --path <path>"
            " --num_processes <#cpu>' to pre-generate the lookup table with"
            ' multiple processes in parallel for faster results (preferably on'
            ' a machine with many CPUs).'
        ),
        lookup_table_name,
        max_degree,
        lstart,
        lookup_table_name,
        set_cache_method_name,
        pregeneration_name,
    )
  else:
    logging.info(
        (
            'Generating %s lookup table with values for degrees up to '
            'max_degree=%d (starting from degree=%d).'
        ),
        lookup_table_name,
        max_degree,
        lstart,
    )


def _save_lookup_table_to_disk(
    lookup_table: Dict[str, Any],
    lookup_table_name: str,
    config_cache_path: epath.Path,
) -> None:
  """Save a lookup table to disk.

  Args:
    lookup_table: A dictionary containing the values that should be saved to
      disk.
    lookup_table_name: Name of the lookup table (used only for printing).
    config_cache_path: Path to the cache file on disk, which can be specified
      using the Config class.
  """
  logging.info('Saving %s lookup table to disk.', lookup_table_name)
  if config_cache_path is not None:
    try:
      with config_cache_path.open('wb') as f:
        np.savez_compressed(f, **lookup_table)
    except (OSError, IOError):
      logging.exception(
          "Failed to save %s lookup table to '%s'. Continuing anyway...",
          lookup_table_name,
          config_cache_path,
      )
