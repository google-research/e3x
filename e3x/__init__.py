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

"""e3x API.

All functions and classes from the :obj:`nn <e3x.nn>`, :obj:`ops <e3x.ops>`, and
:obj:`so3 <e3x.so3>` submodules can be used without specifying their full path
within each submodule (to prevent extremely long names). For example,
``e3x.nn.relu`` is equivalent to ``e3x.nn.activations.relu``.
"""


from . import nn
from . import ops
from . import so3
from .config import Config
from .version import __version__
