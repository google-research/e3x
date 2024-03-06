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

r"""Custom basic operations used in e3x.

Operations defined in this submodule either simplify working with index lists,
parameter initialization, or are numerically safe alternatives to standard
operations.
"""


from .helpers import evaluate_derivatives
from .helpers import inverse_softplus
from .indexed import dense_pairwise_indices
from .indexed import dense_to_sparse_indices
from .indexed import gather_dst
from .indexed import gather_src
from .indexed import indexed_max
from .indexed import indexed_min
from .indexed import indexed_softmax
from .indexed import indexed_sum
from .indexed import sparse_pairwise_indices
from .indexed import sparse_to_dense_indices
from .safe import norm
from .safe import normalize
from .safe import normalize_and_return_norm
