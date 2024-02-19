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

"""Utility functions."""

from typing import Any, Sequence, Union

Shape = Sequence[Union[int, Any]]


def is_broadcastable(shape1: Shape, shape2: Shape) -> bool:
  """Determine whether two shapes are broadcastable.

  Args:
    shape1: First shape.
    shape2: Second shape.

  Returns:
    True if the shapes are broadcastable, False otherwise.
  """
  for a, b in zip(shape1[::-1], shape2[::-1]):
    if not (a == b or a == 1 or b == 1):
      return False
  return True
