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

"""Utilities for testing."""

from typing import Any, Dict

import pytest


def subtests(subtest_dictionary: Dict[str, Dict[str, Any]]):
  """Wrapped pytest.mark.parametrize decorator (to make it more readable).

  For example, instead of:

  @pytest.mark.parametrize(
    'test_input,expected',
    [('3+5', 8), ('6*9', 42)],
    ids=['addition', 'multiplication'])
  def test_eval(test_input, expected):
    assert eval(test_input) == expected

  this allows to write:

  @subtests({
    'addition': dict(
      test_input='3+5',
      expected=8
    ),
    'multiplication': dict(
      test_input='6*9',
      expected=42
    )
  })
  def test_eval(test_input, expected):
    assert eval(test_input) == expected

  which is more readable, especially when defining many subtests.

  Args:
    subtest_dictionary: Dictionary of dictionaries, where the keys of the outer
      dictionary are the subtest ids, and the inner dictionary contains subtest
      parameters stored as key: value pairs of the form argname: argvalue. Since
      pytest.mark.parametrize works with positional arguments only, all subtests
      must specify the same set of arguments, or the function will throw a
      KeyError.

  Returns:
    The equivalent pytest.mark.parametrize fixture (see example above).
  """
  argnames = set()
  for v in subtest_dictionary.values():
    for k in v.keys():
      argnames.add(k)
  argnames = sorted(argnames)
  return pytest.mark.parametrize(
      argnames=argnames,
      argvalues=[[v[k] for k in argnames] for v in subtest_dictionary.values()],
      ids=subtest_dictionary.keys(),
  )
