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

"""Components for building equivariant deep learning architectures."""


from . import initializers
from .activations import bent_identity
from .activations import celu
from .activations import elu
from .activations import gelu
from .activations import hard_silu
from .activations import hard_swish
from .activations import hard_tanh
from .activations import leaky_relu
from .activations import mish
from .activations import relu
from .activations import relu6
from .activations import selu
from .activations import serf
from .activations import shifted_softplus
from .activations import silu
from .activations import soft_sign
from .activations import swish
from .features import add
from .features import change_max_degree_or_type
from .features import even_degree_mask
from .features import odd_degree_mask
from .features import reflect
from .features import rotate
from .functions import basic_bernstein
from .functions import basic_chebyshev
from .functions import basic_fourier
from .functions import basic_gaussian
from .functions import cosine_cutoff
from .functions import exponential_bernstein
from .functions import exponential_chebyshev
from .functions import exponential_fourier
from .functions import exponential_gaussian
from .functions import exponential_mapping
from .functions import reciprocal_bernstein
from .functions import reciprocal_chebyshev
from .functions import reciprocal_fourier
from .functions import reciprocal_gaussian
from .functions import reciprocal_mapping
from .functions import rectangular_window
from .functions import sinc
from .functions import smooth_cutoff
from .functions import smooth_damping
from .functions import smooth_switch
from .functions import smooth_window
from .functions import triangular_window
from .modules import _Conv
from .modules import Dense
from .modules import Embed
from .modules import FusedTensor
from .modules import MessagePass
from .modules import MultiHeadAttention
from .modules import SelfAttention
from .modules import Tensor
from .modules import TensorDense
from .wrappers import basis
from .wrappers import ExponentialBasis
