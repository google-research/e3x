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

"""Equivariant neural network modules.

.. _Modules:
"""

import dataclasses
import functools
import itertools
import math
from typing import Any, List, Optional, Sequence, Tuple, Union

from e3x import ops
from e3x import so3
from e3x import util
from e3x.config import Config
from flax import linen as nn
from flax.linen.dtypes import promote_dtype
import jax
import jax.numpy as jnp
import jaxtyping

from . import initializers
from .features import _extract_max_degree_and_check_shape
from .features import change_max_degree_or_type

FusedTensorInitializerFn = initializers.FusedTensorInitializerFn
InitializerFn = initializers.InitializerFn
Array = jaxtyping.Array
Bool = jaxtyping.Bool
Float = jaxtyping.Float
Integer = jaxtyping.Integer
UInt32 = jaxtyping.UInt32
Shape = Sequence[Union[int, Any]]
Dtype = Any  # This could be a real type if support for that is added.
PRNGKey = UInt32[Array, '2']
PrecisionLike = jax.lax.PrecisionLike


default_embed_init = jax.nn.initializers.variance_scaling(
    1.0, 'fan_in', 'normal', out_axis=0
)


class Embed(nn.Module):
  """Embedding module.

  A parameterized function from integers :math:`[0, n)` to :math:`d`-dimensional
  scalar features.

  Attributes:
    num_embeddings: Number of embeddings :math:`n`.
    features: Dimension :math:`d` of the feature space.
    dtype: The :class:`dtype <jax.numpy.dtype>` of the embedding vectors.
    param_dtype: The dtype passed to parameter initializers.
    embedding_init: Embedding initializer.
  """

  num_embeddings: int
  features: int
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32
  embedding_init: InitializerFn = default_embed_init
  embedding: Float[Array, 'num_embeddings 1 1 features'] = dataclasses.field(
      init=False
  )

  def setup(self):
    self.embedding = self.param(
        'embedding',
        self.embedding_init,
        (self.num_embeddings, 1, 1, self.features),
        self.param_dtype,
    )

  def __call__(
      self, inputs: Integer[Array, '...']
  ) -> Float[Array, '... 1 1 F']:
    """Embeds the inputs along the last dimension.

    Scalar features are returned with a shape consistent with the conventions
    used in other equivariant operations.

    Args:
      inputs: Input data, all dimensions are considered batch dimensions.

    Returns:
      Output which is embedded input data. The output shape follows the input,
      with additional ``1,1,features`` dimensions appended.
    """
    if not jnp.issubdtype(inputs.dtype, jnp.integer):
      raise ValueError('input type must be an integer or unsigned integer')
    (embedding,) = promote_dtype(
        self.embedding, dtype=self.dtype, inexact=False
    )
    return jnp.take(embedding, inputs, axis=0)


default_kernel_init = jax.nn.initializers.lecun_normal()


class Dense(nn.Module):
  r"""A linear transformation applied over the last dimension of the input.

  The transformation can be written as

  .. math::

    \mathbf{y}^{(\ell_p)} = \begin{cases}
    \mathbf{x}^{(\ell_p)}\mathbf{W}_{(\ell_p)} + \mathbf{b} & \ell_p = 0_+ \\
    \mathbf{x}^{(\ell_p)}\mathbf{W}_{(\ell_p)} & \ell_p \neq 0_+
    \end{cases}

  where
  :math:`\mathbf{x} \in \mathbb{R}^{P\times (L+1)^2 \times F_{\mathrm{in}}}` and
  :math:`\mathbf{y} \in \mathbb{R}^{P\times (L+1)^2 \times F_{\mathrm{out}}}`
  are the inputs and outputs, respectively. Here, :math:`P` is either :math:`1`
  or :math:`2` (depending on whether the inputs contain pseudotensors or not),
  :math:`L` is the maximum degree of the input features, and
  :math:`F_{\mathrm{in}}` and :math:`F_{\mathrm{out}}` = ``features`` are the
  number of input and output features. Every combination of degree :math:`\ell`
  and parity :math:`p` has separate weight matrices
  :math:`\mathbf{W}_{(\ell_p)}`. Note that a bias term
  :math:`\mathbf{b} \in \mathbb{R}^{1\times 1 \times F_{\mathrm{out}}}` is only
  applied to the scalar channel (:math:`\ell_p= 0_+`) when ``use_bias=True``.

  Attributes:
    features: The number of output features :math:`F_{\mathrm{out}}`.
    use_bias: Whether to add a bias to the scalar channel of the output.
    dtype: The dtype of the computation.
    param_dtype: The dtype passed to parameter initializers.
    precision: Numerical precision of the computation, see `jax.lax.Precision`
      for details.
    kernel_init: Initializer function for the weight matrix.
    bias_init: Initializer function for the bias.
  """

  features: int
  use_bias: bool = True
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32
  precision: PrecisionLike = None
  kernel_init: InitializerFn = default_kernel_init
  bias_init: InitializerFn = jax.nn.initializers.zeros

  @nn.nowrap
  def _make_dense_for_each_degree(
      self, max_degree: int, use_bias: bool, name_suffix: Optional[str] = None
  ) -> List[nn.Dense]:
    """Helper function for generating Modules."""
    if name_suffix is None:
      parity = ['+', '-']
      name = [f'{l}{parity[l%2]}' for l in range(max_degree + 1)]
    else:
      name = [f'{l}{name_suffix}' for l in range(max_degree + 1)]
    dense = []
    for l in range(max_degree + 1):
      dense.append(
          nn.Dense(
              features=self.features,
              use_bias=use_bias and l == 0,  # Apply bias only for scalars!
              dtype=self.dtype,
              param_dtype=self.param_dtype,
              precision=self.precision,
              kernel_init=self.kernel_init,
              bias_init=self.bias_init,
              name=name[l],
          )
      )
    return dense

  @nn.compact
  def __call__(
      self,
      inputs: Union[
          Float[Array, '... 1 (max_degree+1)**2 in_features'],
          Float[Array, '... 2 (max_degree+1)**2 in_features'],
      ],
  ) -> Union[
      Float[Array, '... 1 (max_degree+1)**2 out_features'],
      Float[Array, '... 2 (max_degree+1)**2 out_features'],
  ]:
    """Applies a linear transformation to the inputs along the last dimension.

    Args:
      inputs: The nd-array to be transformed.

    Returns:
      The transformed input.
    """
    max_degree = _extract_max_degree_and_check_shape(inputs.shape)

    if inputs.shape[-3] == 2:  # Has pseudotensors.
      dense_e = self._make_dense_for_each_degree(max_degree, self.use_bias, '+')
      dense_o = self._make_dense_for_each_degree(max_degree, False, '-')
      return jnp.stack(
          [
              # Even parity (tensors).
              jnp.concatenate(
                  [
                      dense_e[l](inputs[..., 0, l**2 : (l + 1) ** 2, :])
                      for l in range(max_degree + 1)
                  ],
                  axis=-2,
              ),
              # Odd parity (pseudotensors).
              jnp.concatenate(
                  [
                      dense_o[l](inputs[..., 1, l**2 : (l + 1) ** 2, :])
                      for l in range(max_degree + 1)
                  ],
                  axis=-2,
              ),
          ],
          axis=-3,
      )
    elif inputs.shape[-3] == 1:  # Has no pseudotensors.
      dense = self._make_dense_for_each_degree(max_degree, self.use_bias)
      return jnp.concatenate(
          [
              dense[l](inputs[..., l**2 : (l + 1) ** 2, :])
              for l in range(max_degree + 1)
          ],
          axis=-2,
      )
    else:
      assert False, 'Shape has passed checks even though it is invalid!'


def _duplication_indices_for_max_degree(
    max_degree: int,
) -> Integer[Array, '(max_degree+1)**2']:
  """Returns indices for use in jnp.take to expand degree-wise arrays.

  This functionality is often needed to duplicate the values of an array that
  stores degree-wise values for each order of the degree. The value for degree
  l needs to be repeated 2*l+1 times. For example, for max_degree=2, the
  duplication indices are [0, 1, 1, 1, 2, 2, 2, 2, 2].

  Args:
    max_degree: The maximum degree for which to construct indices.

  Returns:
    The corresponding indices for use in jnp.take.
  """
  l = jnp.arange(max_degree + 1)  # [0, 1, 2, ..., max_degree]
  return jnp.repeat(l, 2 * l + 1, total_repeat_length=(max_degree + 1) ** 2)


def _make_tensor_product_mask(
    shape: Tuple[int, int, int, int, int, int],
    dtype: Dtype = jnp.float32,
) -> Float[Array, 'S1 L1 S2 L2 S3 L3 1']:
  """Helper function for generating the tensor product mask.

  Can be multiplied with a parameter matrix to zero out all forbidden (parity
  violating) coupling paths. A coupling path is forbidden whenever the sum of
  the degrees of the coupled irreps and the degree of the output irrep are not
  both even or both odd.

  The input shape tuple must have the form (S1, L1, S2, L2, S3, L3), where each
  S? is either 1 or 2 and each L? is short for (max_degree?+1)**2.

  Args:
    shape: The input shape tuple (see above).
    dtype: The dtype of the returned mask.

  Returns:
    A mask array containing only 0s and 1s with the same shape as specified by
    the input shape tuple with an appended size 1 dimension at position -1.
  """

  def _make_index_combinations(
      parity: int, max_degree: int
  ) -> List[Tuple[int, int, int]]:
    """Helper function for generating index combinations (useful for loops).

    The parity input is either 1 (a single dimension that stores irreps with
    mixed parity, where even degrees have even parity and odd degrees have odd
    parity) or 2 (two dimensions, with all irreps with even parity in position
    0 and all irreps with odd parity in position 1).

    Args:
      parity: Determines the desired parity convention (either 1 or 2, see
        above).
      max_degree: The maximum degree to consider.

    Returns:
      A list of tuples (p, l, d), where p is the parity index, l is the degree
      index, and d is 0 or 1, indicating even or odd parity, respectively.
    """
    assert parity in (1, 2)
    if parity == 2:  # All entries for p=0 are even and for p=1 odd.
      idx = [(0, l, 0) for l in range(max_degree + 1)]
      idx += [(1, l, 1) for l in range(max_degree + 1)]
    else:  # Entries are even if l is even and odd if l is odd.
      idx = [(0, l, l % 2) for l in range(max_degree + 1)]
    return idx

  # Initialize mask to ones (forbidden paths are set to zero below).
  mask = jnp.ones((*shape, 1), dtype=dtype)
  # Generate lists of index combinations for the input shape.
  idx1 = _make_index_combinations(shape[0], shape[1])
  idx2 = _make_index_combinations(shape[2], shape[3])
  idx3 = _make_index_combinations(shape[4], shape[5])
  # Loop over all possible index combinations.
  for pld1, pld2, pld3 in itertools.product(idx1, idx2, idx3):
    p1, l1, d1 = pld1
    p2, l2, d2 = pld2
    p3, l3, d3 = pld3
    if (d1 + d2) % 2 != d3:  # Parity violation!
      mask = mask.at[p1, l1, p2, l2, p3, l3, :].set(0)
  return mask


default_tensor_kernel_init = initializers.tensor_lecun_normal()


class Tensor(nn.Module):
  r"""Tensor product of two equivariant feature representations.

  Computes linear combinations (with learnable coefficients) of the direct sum
  representation of all possible tensor products of irreps in the input
  features. If the inputs are
  :math:`\mathbf{x} \in \mathbb{R}^{P_1\times (L_1+1)^2 \times F}` and
  :math:`\mathbf{y} \in \mathbb{R}^{P_2\times (L_2+1)^2 \times F}`, the output
  is :math:`\mathbf{z} \in \mathbb{R}^{P_3\times (L_3+1)^2 \times F}`. Here,
  :math:`P_1`, :math:`P_2`, and :math:`P_3` are either :math:`1` or
  :math:`2` (depending on whether the inputs/output contain pseudotensors or
  not) and  :math:`L_1`, :math:`L_2`, and :math:`L_3` nonnegative integers
  (:math:`L_3` = ``max_degree``). The entries of :math:`\mathbf{z}` are
  given by

  .. math::

    \mathbf{z}^{(c_\gamma)} = \sum_{(a_\alpha,b_\beta)\in V}
      \mathbf{w}_{(a_\alpha,b_\beta,c_\gamma)} \circ \left(
      \mathbf{x}^{(a_\alpha)} \otimes^{(c_\gamma)}\mathbf{y}^{(b_\beta)}
      \right)\,,

  where the sum runs over all :math:`(a_\alpha,b_\beta)` in the set of valid
  combinations :math:`V` and :math:`\mathbf{w}_{(a_\alpha,b_\beta,c_\gamma)} \in
  \mathbb{R}^{1\times 1\times F}` are learnable (feature-wise) weight
  parameters. Each combination :math:`(a_\alpha,b_\beta,c_\gamma)` has separate
  parameters and the element-wise product ':math:`\circ`' implies broadcasting
  over dimensions. The set :math:`V` contains all :math:`(a_\alpha,b_\beta)` for
  which the condition

  .. math::

    \lvert a - b \rvert \leq c \leq a + b \enspace \land \enspace
    \left(
      \left( \gamma = +1 \enspace \land \enspace \alpha = \beta \right)
    \enspace \lor \enspace
      \left( \gamma = -1 \enspace \land \enspace \alpha \neq \beta \right)
    \right)

  is true. If ``include_pseudotensors = False``, coupling paths that lead to
  pseudotensors are not computed. This means that all entries that do not
  satisfy

  .. math::

    \left(c \in  \{2n+1 : n\in \mathbb{N}_0\} \enspace \land \enspace
     \gamma = -1 \right) \enspace \lor \enspace
    \left(c \in  \{2n : n\in \mathbb{N}_0\} \enspace  \land \enspace
    \gamma = +1 \right)

  (either :math:`c` is odd *and* the parity is odd, *or* :math:`c` is
  even *and* the parity is even) are omitted. See also
  :ref:`here <CouplingIrreps>` for more details on the notation used here and
  the coupling of irreps in general. The following diagram shows a visualization
  of the computation for the example :math:`P_1=P_2=P_3=2`,
  :math:`L_1=L_2=L_3=1`. For better clarity, weights are only labelled for 2 out
  of the 20 possible coupling paths.

  .. image:: ../_static/tensor_product_visualization.svg
   :scale: 100 %
   :align: center
   :alt: visualization

  |

  Attributes:
    max_degree: Maximum degree of the output. If not given, ``max_degree`` is
      chosen as the maximum of the maximum degrees of inputs1 and inputs2.
    include_pseudotensors: If ``False``, all coupling paths that produce
      pseudotensors are omitted.
    cartesian_order: If ``True``, Cartesian order is assumed.
    dtype: The dtype of the computation.
    param_dtype: The dtype passed to parameter initializers.
    precision: Numerical precision of the computation, see `jax.lax.Precision`
      for details.
    kernel_init: Initializer function for the weight matrix.
  """

  max_degree: Optional[int] = None
  include_pseudotensors: bool = True
  cartesian_order: bool = Config.cartesian_order
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32
  precision: PrecisionLike = None
  kernel_init: InitializerFn = default_tensor_kernel_init

  @nn.compact
  def __call__(
      self,
      inputs1: Union[
          Float[Array, '... 1 (max_degree1+1)**2 num_features'],
          Float[Array, '... 2 (max_degree1+1)**2 num_features'],
      ],
      inputs2: Union[
          Float[Array, '... 1 (max_degree2+1)**2 num_features'],
          Float[Array, '... 2 (max_degree2+1)**2 num_features'],
      ],
  ) -> Union[
      Float[Array, '... 1 (max_degree3+1)**2 num_features'],
      Float[Array, '... 2 (max_degree3+1)**2 num_features'],
  ]:
    """Computes the tensor product of inputs1 and inputs2.

    Args:
      inputs1: The first factor of the tensor product.
      inputs2: The second factor of the tensor product.

    Returns:
      The tensor product of inputs1 and inputs2, where each output irrep is a
      weighted linear combination with learnable weights of all valid coupling
      paths.
    """
    # Determine max_degree of inputs and output.
    max_degree1 = _extract_max_degree_and_check_shape(inputs1.shape)
    max_degree2 = _extract_max_degree_and_check_shape(inputs2.shape)
    max_degree3 = (
        max(max_degree1, max_degree2)
        if self.max_degree is None
        else self.max_degree
    )

    # Check that max_degree3 is not larger than is sensible.
    if max_degree3 > max_degree1 + max_degree2:
      raise ValueError(
          'max_degree for the tensor product of inputs with max_degree'
          f' {max_degree1} and {max_degree2} can be at most'
          f' {max_degree1 + max_degree2}, received max_degree={max_degree3}'
      )

    # Check that axis -1 (number of features) of both inputs matches in size.
    if inputs1.shape[-1] != inputs2.shape[-1]:
      raise ValueError(
          'axis -1 of inputs1 and input2 must have the same size, '
          f'received shapes {inputs1.shape} and {inputs2.shape}'
      )

    # Extract number of features from size of axis -1.
    features = inputs1.shape[-1]

    # If both inputs contain no pseudotensors and at least one input or the
    # output has max_degree == 0, the tensor product will not produce
    # pseudotensors, in this case, the output will be returned with no
    # pseudotensor channel, regardless of whether self.include_pseudotensors is
    # True or False.
    if (inputs1.shape[-3] == inputs2.shape[-3] == 1) and (
        max_degree1 == 0 or max_degree2 == 0 or max_degree3 == 0
    ):
      include_pseudotensors = False
    else:
      include_pseudotensors = self.include_pseudotensors

    # Determine number of parity channels.
    num_parity1 = inputs1.shape[-3]
    num_parity2 = inputs2.shape[-3]
    num_parity3 = 2 if include_pseudotensors else 1

    # Initialize parameters.
    kernel_shape = (
        num_parity1,
        max_degree1 + 1,
        num_parity2,
        max_degree2 + 1,
        num_parity3,
        max_degree3 + 1,
        features,
    )
    kernel = self.param(
        'kernel', self.kernel_init, kernel_shape, self.param_dtype
    )
    (kernel,) = promote_dtype(kernel, dtype=self.dtype)

    # If any of the two inputs or the output do not contain pseudotensors, the
    # forbidded coupling paths correspond to "mixed entries within array
    # slices". However, if all inputs and the output contain pseudotensors, the
    # forbidden coupling paths all correspond to "whole slices" of the arrays.
    # Instead of masking specific entries, it is then more efficient to slice
    # the arrays and compute the allowed paths separately, effectively cutting
    # the number of necessary computations in half.
    mixed_coupling_paths = not num_parity1 == num_parity2 == num_parity3 == 2

    # Initialize constants.
    with jax.ensure_compile_time_eval():
      # Clebsch-Gordan tensor.
      cg = so3.clebsch_gordan(
          max_degree1,
          max_degree2,
          max_degree3,
          cartesian_order=self.cartesian_order,
      )

      # Mask for zeroing out forbidden (parity violating) coupling paths.
      if mixed_coupling_paths:
        mask = _make_tensor_product_mask(kernel_shape[:-1])
      else:
        mask = 1

      # Indices for expanding shape of kernel.
      idx1 = _duplication_indices_for_max_degree(max_degree1)
      idx2 = _duplication_indices_for_max_degree(max_degree2)
      idx3 = _duplication_indices_for_max_degree(max_degree3)

    # Mask kernel (only necessary for mixed coupling paths)
    if mixed_coupling_paths:
      kernel *= mask

    # Expand shape (necessary for correct broadcasting).
    kernel = jnp.take(kernel, idx1, axis=1, indices_are_sorted=True)
    kernel = jnp.take(kernel, idx2, axis=3, indices_are_sorted=True)
    kernel = jnp.take(kernel, idx3, axis=5, indices_are_sorted=True)

    if mixed_coupling_paths:
      return jnp.einsum(
          '...plf,...qmf,plqmrnf,lmn->...rnf',
          inputs1,
          inputs2,
          kernel,
          cg,
          precision=self.precision,
          optimize='optimal',
      )
    else:
      # Compute all allowed even/odd + even/odd -> even/odd coupling paths.
      def _couple_slices(
          i: int, j: int, k: int
      ) -> Float[Array, '... (max_degree3+1)**2 num_features']:
        """Helper function for coupling slice (i, j, k)."""
        return jnp.einsum(
            '...lf,...mf,lmnf,lmn->...nf',
            inputs1[..., i, :, :],
            inputs2[..., j, :, :],
            kernel[i, :, j, :, k, :, :],
            cg,
            precision=self.precision,
            optimize='optimal',
        )

      eee = _couple_slices(0, 0, 0)  # even + even -> even
      ooe = _couple_slices(1, 1, 0)  # odd + odd -> even
      eoo = _couple_slices(0, 1, 1)  # even + odd -> odd
      oeo = _couple_slices(1, 0, 1)  # odd + even -> odd

      # Combine same parities and return stacked features.
      return jnp.stack((eee + ooe, eoo + oeo), axis=-3)


default_fused_tensor_kernel_init = initializers.fused_tensor_normal


class FusedTensor(nn.Module):
  r"""Fused tensor product of two equivariant feature representations.

  This module performs a similar function as
  :class:`Tensor <e3x.nn.modules.Tensor>`, but has a lower computational
  complexity and fewer learnable parameters. Given two inputs
  :math:`\mathbf{x} \in \mathbb{R}^{P_1\times (L_1+1)^2 \times F}` and
  :math:`\mathbf{y} \in \mathbb{R}^{P_2\times (L_2+1)^2 \times F}` the output
  is :math:`\mathbf{z} \in \mathbb{R}^{P_3\times (L_3+1)^2 \times F}`. Here,
  :math:`P_1`, :math:`P_2`, and :math:`P_3` are either :math:`1` or
  :math:`2` (depending on whether the inputs/output contain pseudotensors or
  not) and  :math:`L_1`, :math:`L_2`, and :math:`L_3` are positive integers or
  zero (:math:`L_3` = ``max_degree``). The computation consists of the following
  steps:

  1a. The constituent irreps
  :math:`\mathbf{x}^{(a_\alpha)} \in \mathbb{R}^{1\times (2a+1) \times F}` and
  :math:`\mathbf{y}^{(b_\beta)} \in \mathbb{R}^{1\times (2b+1) \times F}` of the
  features :math:`\mathbf{x}` and :math:`\mathbf{y}` are transformed via a
  change of basis ("vectors" to "matrices") to
  :math:`\mathbf{\tilde{x}}^{(a_\alpha)}, \mathbf{\tilde{y}}^{(b_\beta)} \in
  \mathbb{R}^{(2\tilde{l}+1) \times (2\tilde{l}+1) \times F}` with
  :math:`\tilde{l} =
  \left\lceil\frac{\mathrm{max}(L_1, L_2, L_3)}{2}\right\rceil`.

  1b. The individual "matrix irreps" with equal parities are multiplied with
  (separate) learnable weights :math:`\mathbf{w} \in
  \mathbb{R}^{1 \times 1 \times F}` and added to form the matrices
  :math:`\mathbf{X}^{(+)},\mathbf{X}^{(-)},\mathbf{Y}^{(+)},\mathbf{Y}^{(-)}
  \in \mathbb{R}^{(2\tilde{l}+1) \times (2\tilde{l}+1) \times F}` (the
  element-wise product ':math:`\circ`' implies broadcasting over
  dimensions):

  .. math::
    \mathbf{X}^{(+)} &= \sum_{a=0}^{L_1} \mathbf{w}_{\mathbf{x}^{(a_+)}}
               \circ \mathbf{\tilde{x}}^{(a_+)}

    \mathbf{X}^{(-)} &= \sum_{a=0}^{L_1} \mathbf{w}_{\mathbf{x}^{(a_-)}}
               \circ \mathbf{\tilde{x}}^{(a_-)}

    \mathbf{Y}^{(+)} &= \sum_{b=0}^{L_2} \mathbf{w}_{\mathbf{y}^{(b_+)}}
               \circ \mathbf{\tilde{y}}^{(b_+)}

    \mathbf{Y}^{(-)} &= \sum_{b=0}^{L_2} \mathbf{w}_{\mathbf{y}^{(b_-)}}
               \circ \mathbf{\tilde{y}}^{(b_-)}

  Any potentially "missing" irreps, e.g. :math:`\mathbf{\tilde{x}}^{(1_+)}` if
  :math:`\mathbf{x}` does not contain pseudotensors, are assumed to be zero.

  2. The so-formed matrices are coupled by matrix multiplication to produce new
  matrices as follows ("batch matrix multiplication" over the last dimension
  with size :math:`F` is implied):

  .. math::
    \mathbf{Z}^{(+,+)} &= \mathbf{X}^{(+)}\mathbf{Y}^{(+)}

    \mathbf{Z}^{(-,-)} &= \mathbf{X}^{(-)}\mathbf{Y}^{(-)}

    \mathbf{Z}^{(+,-)} &= \mathbf{X}^{(+)}\mathbf{Y}^{(-)}

    \mathbf{Z}^{(-,+)} &= \mathbf{X}^{(-)}\mathbf{Y}^{(+)}

  Note: :math:`\mathbf{Z}^{(+,+)}` and :math:`\mathbf{Z}^{(-,-)}` have even
  parity and :math:`\mathbf{Z}^{(+,-)}` and :math:`\mathbf{Z}^{(-,+)}` have odd
  parity.

  3a. The matrices :math:`\mathbf{Z}^{(+,+)}, \mathbf{Z}^{(-,-)},
  \mathbf{Z}^{(+,-)}, \mathbf{Z}^{(-,+)}` are "decomposed" into their
  constituent "matrix irreps" :math:`\mathbf{\tilde{z}}^{(+,+,c)},
  \mathbf{\tilde{z}}^{(-,-,c)},\mathbf{\tilde{z}}^{(+,-,c)},
  \mathbf{\tilde{z}}^{(-,+,c)}
  \in \mathbb{R}^{(2\tilde{l}+1) \times (2\tilde{l}+1)\times F}` with
  :math:`c = 0,\dots,L_3`. During this decomposition, the individual matrix
  irreps are multiplied with (separate) learnable weights
  :math:`\mathbf{w} \in \mathbb{R}^{\times 1 \times 1\times F}`. This step can
  be thought of as performing the inverse of the operation in step 1b.

  3b. The so-obtained "matrix irreps" are transformed via a change of basis
  ("matrices" to "vectors") to obtain the irreps
  :math:`\mathbf{z}^{(+,+,c)},\mathbf{z}^{(-,-,c)},\mathbf{z}^{(+,-,c)},
  \mathbf{z}^{(-,+,c)} \in \mathbb{R}^{1 \times (2c+1) \times F}` for each value
  of :math:`c`. This step can be thought of as performing the inverse of the
  operation in step 1a.

  4. Finally, the output irreps of degree :math:`c` are obtained by summing
  matching parities:

  .. math::
    \mathbf{z}^{(c_+)} &= \mathbf{z}^{(+,+,c)} + \mathbf{z}^{(-,-,c)}

    \mathbf{z}^{(c_-)} &= \mathbf{z}^{(+,-,c)} + \mathbf{z}^{(-,+,c)}

  If ``include_pseudotensors = False``, all irreps that correspond to
  pseudotensors are discarded (set to zero).

  The following diagram shows a visualization of the computation for the example
  :math:`P_1=P_2=P_3=2`, :math:`L_1=L_2=L_3=2`.

  .. image:: ../_static/fused_tensor_product_visualization.svg
   :scale: 100 %
   :align: center
   :alt: visualization

  |

  While it is helpful to think of the change of basis and weighting of
  individual components performed in steps 1a,b and 3a,b separately, these steps
  are really performed concurrently using Clebsch-Gordan coefficients
  :math:`C^{l_3,m_3}_{l_1,m_1,l_2,m_2}`. The actual computation performed for
  the "vectors to matrices" change of basis plus multiplication by weights is:

  .. math::
    X_{\tilde{m},\tilde{m}'} = \sum_{l=0}^{L} w_{l}\sum_{m=-l}^{l}
    C^{l,m}_{\tilde{l},\tilde{m},\tilde{l},\tilde{m}'} x_{l}^{m}

  whereas the computation performed for the "matrices to vectors" change of
  basis plus multiplication by weights is:

  .. math::
    x_{l}^{m} = \tilde{w}_{l}\sum_{\tilde{m}=-\tilde{l}}^{\tilde{l}}
    \sum_{\tilde{m}'=-\tilde{l}}^{\tilde{l}}
    C^{l,m}_{\tilde{l},\tilde{m},\tilde{l},\tilde{m}'} X_{\tilde{m},\tilde{m}'}

  Here, :math:`X_{\tilde{m},\tilde{m}'}` are the elements of matrix
  :math:`\mathbf{X} \in \mathbb{R}^{(2\tilde{l}+1)\times(2\tilde{l}+1)}`

  .. math::
    \mathbf{X} = \begin{bmatrix}
    X_{-\tilde{l},-\tilde{l}} & X_{-\tilde{l},-\tilde{l}+1} &
    \cdots & X_{-\tilde{l},\tilde{l}} \\
    X_{-\tilde{l}+1,-\tilde{l}} & X_{-\tilde{l}+1,-\tilde{l}+1} &
    \cdots & X_{-\tilde{l}+1,\tilde{l}} \\
    \vdots & \vdots & \ddots & \vdots \\
    X_{\tilde{l},-\tilde{l}} & X_{\tilde{l},-\tilde{l}+1} &
    \cdots & X_{\tilde{l},\tilde{l}} \\ \end{bmatrix}

  and :math:`x_{l}^{m}` are the individual entries of irreps :math:`\mathbf{x}
  \in \mathbb{R}^{1\times(L+1)^2}`

  .. math::
    \mathbf{x} = [\underbrace{x_{0}^{0}}_{\mathbf{x}^{(0)}} \quad
    \underbrace{x_{1}^{-1} \;\; x_{1}^{0} \;\;  x_{1}^{1}}_{\mathbf{x}^{(1)}}
    \quad \cdots \quad \underbrace{\quad x_{L}^{-L}  \;\; \cdots \;\;
    x_{L}^{L}}_{\mathbf{x}^{(L)}} ]

  (the feature dimension with size :math:`F` and parity indicators :math:`+/-`
  are omitted for clarity).

  Attributes:
    max_degree: Maximum degree of the output. If not given, ``max_degree`` is
      chosen as the maximum of the maximum degrees of inputs1 and inputs2.
    include_pseudotensors: If ``False``, pseudotensors are omitted in the
      output.
    cartesian_order: If ``True``, Cartesian order is assumed.
    dtype: The dtype of the computation.
    param_dtype: The dtype passed to parameter initializers.
    precision: Numerical precision of the computation, see `jax.lax.Precision`
      for details.
    kernel_init: Initializer function for the weight kernel.
  """

  max_degree: Optional[int] = None
  include_pseudotensors: bool = True
  cartesian_order: bool = Config.cartesian_order
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32
  precision: PrecisionLike = None
  kernel_init: FusedTensorInitializerFn = default_fused_tensor_kernel_init

  @nn.compact
  def __call__(
      self,
      inputs1: Union[
          Float[Array, '... 1 (max_degree1+1)**2 num_features'],
          Float[Array, '... 2 (max_degree1+1)**2 num_features'],
      ],
      inputs2: Union[
          Float[Array, '... 1 (max_degree2+1)**2 num_features'],
          Float[Array, '... 2 (max_degree2+1)**2 num_features'],
      ],
  ) -> Union[
      Float[Array, '... 1 (max_degree3+1)**2 num_features'],
      Float[Array, '... 2 (max_degree3+1)**2 num_features'],
  ]:
    """Computes the fused tensor product of inputs1 and inputs2.

    Args:
      inputs1: The first factor of the tensor product.
      inputs2: The second factor of the tensor product.

    Returns:
      The tensor product of inputs1 and inputs2, where each output irrep is a
      weighted linear combination with (partially shared) learnable weights of
      all valid coupling paths.
    """
    # Determine max_degree of inputs and output.
    max_degree1 = _extract_max_degree_and_check_shape(inputs1.shape)
    max_degree2 = _extract_max_degree_and_check_shape(inputs2.shape)
    max_degree3 = (
        max(max_degree1, max_degree2)
        if self.max_degree is None
        else self.max_degree
    )

    # Check that max_degree3 is not larger than is sensible.
    if max_degree3 > max_degree1 + max_degree2:
      raise ValueError(
          'max_degree for the tensor product of inputs with max_degree'
          f' {max_degree1} and {max_degree2} can be at most'
          f' {max_degree1 + max_degree2}, received max_degree={max_degree3}'
      )

    # Check that axis -1 (number of features) of both inputs matches in size.
    if inputs1.shape[-1] != inputs2.shape[-1]:
      raise ValueError(
          'axis -1 of inputs1 and input2 must have the same size, '
          f'received shapes {inputs1.shape} and {inputs2.shape}'
      )

    # Extract number of features from size of axis -1.
    features = inputs1.shape[-1]

    # Determine number of parity channels for inputs.
    num_parity1 = inputs1.shape[-3]
    num_parity2 = inputs2.shape[-3]

    # If both inputs contain no pseudotensors and at least one input or the
    # output has max_degree == 0, the tensor product will not produce
    # pseudotensors, in this case, the output will be returned with no
    # pseudotensor channel, regardless of whether self.include_pseudotensors is
    # True or False.
    if (num_parity1 == num_parity2 == 1) and (
        max_degree1 == 0 or max_degree2 == 0 or max_degree3 == 0
    ):
      include_pseudotensors = False
    else:
      include_pseudotensors = self.include_pseudotensors

    # Initialize constants.
    with jax.ensure_compile_time_eval():
      max_max_degree = max(max_degree1, max_degree2, max_degree3)

      # Create Clebsch-Gordan tensor and extract relevant slice.
      matrix_degree = math.ceil(max_max_degree / 2)
      cg = so3.clebsch_gordan(
          matrix_degree,
          matrix_degree,
          max_max_degree,
          cartesian_order=self.cartesian_order,
      )
      i = matrix_degree**2
      j = (matrix_degree + 1) ** 2
      cg = cg[i:j, i:j, :]

      # Create masks for even/odd degrees.
      degrees = jnp.arange(max_max_degree + 1)
      repeats = 2 * degrees + 1
      even = (degrees + 1) % 2
      odd = degrees % 2
      max_length = (max_max_degree + 1) ** 2
      even_mask = jnp.repeat(even, repeats, total_repeat_length=max_length)
      even_mask = jnp.expand_dims(even_mask, axis=-1)
      odd_mask = jnp.repeat(odd, repeats, total_repeat_length=max_length)
      odd_mask = jnp.expand_dims(odd_mask, axis=-1)

      # Masks for initialization of parameters (so unused parameters are zeros).
      mask_e1 = True if num_parity1 == 2 else even[: max_degree1 + 1, None]
      mask_o1 = True if num_parity1 == 2 else odd[: max_degree1 + 1, None]
      mask_e2 = True if num_parity2 == 2 else even[: max_degree2 + 1, None]
      mask_o2 = True if num_parity2 == 2 else odd[: max_degree2 + 1, None]
      if num_parity1 == num_parity2 == 1:  # Output has no pseudoscalars.
        mask_o3 = jnp.ones((max_degree3 + 1, 1)).at[0].set(0)
      else:
        mask_o3 = True

      # Variance scaling factor for inputs.
      num_mat = 2 * matrix_degree + 1
      var_in = 1.0 / math.sqrt(num_mat)  # Normalization from matrix mult.
      var_in *= num_mat / min(max_degree1 + 1, max_degree2 + 1)

      # Variance scaling factor for outputs.
      if num_parity1 == num_parity2 == 2:
        var_out = 1.0 / 2.0
      elif num_parity1 == num_parity2 == 1:
        if max_degree1 == 0 or max_degree2 == 0:
          var_out = 1.0
        else:
          var_out = (
              jnp.full((max_degree3 + 1, 1), fill_value=2.0).at[0].set(1.0)
          )
      else:
        var_out = 1.0

    # Initialize parameters.
    shape1 = (max_degree1 + 1, features)
    kernel_e1 = self.param(
        'kernel_e1',
        self.kernel_init(var_in, mask_e1),
        shape1,
        self.param_dtype,
    )
    kernel_o1 = self.param(
        'kernel_o1',
        self.kernel_init(var_in, mask_o1),
        shape1,
        self.param_dtype,
    )
    shape2 = (max_degree2 + 1, features)
    kernel_e2 = self.param(
        'kernel_e2',
        self.kernel_init(var_in, mask_e2),
        shape2,
        self.param_dtype,
    )
    kernel_o2 = self.param(
        'kernel_o2',
        self.kernel_init(var_in, mask_o2),
        shape2,
        self.param_dtype,
    )
    shape3 = (max_degree3 + 1, features)
    kernel_eee = self.param(
        'kernel_eee', self.kernel_init(var_out), shape3, self.param_dtype
    )
    kernel_ooe = self.param(
        'kernel_ooe', self.kernel_init(var_out), shape3, self.param_dtype
    )
    kernel_eoo = self.param(
        'kernel_eoo',
        self.kernel_init(var_out, mask_o3),
        shape3,
        self.param_dtype,
    )
    kernel_oeo = self.param(
        'kernel_oeo',
        self.kernel_init(var_out, mask_o3),
        shape3,
        self.param_dtype,
    )

    # Promote parameters to desired dtype.
    (
        kernel_e1,
        kernel_o1,
        kernel_e2,
        kernel_o2,
        kernel_eee,
        kernel_ooe,
        kernel_eoo,
        kernel_oeo,
    ) = promote_dtype(
        kernel_e1,
        kernel_o1,
        kernel_e2,
        kernel_o2,
        kernel_eee,
        kernel_ooe,
        kernel_eoo,
        kernel_oeo,
        dtype=self.dtype,
    )

    # Compute "stop indices" for slicing CG tensor and other arrays.
    l1 = (max_degree1 + 1) ** 2
    l2 = (max_degree2 + 1) ** 2
    l3 = (max_degree3 + 1) ** 2

    # Expand shape of parameters (repeat degree channels).
    repeats1 = repeats[: max_degree1 + 1]
    kernel_e1 = jnp.repeat(kernel_e1, repeats1, axis=0, total_repeat_length=l1)
    kernel_o1 = jnp.repeat(kernel_o1, repeats1, axis=0, total_repeat_length=l1)
    repeats2 = repeats[: max_degree2 + 1]
    kernel_e2 = jnp.repeat(kernel_e2, repeats2, axis=0, total_repeat_length=l2)
    kernel_o2 = jnp.repeat(kernel_o2, repeats2, axis=0, total_repeat_length=l2)
    repeats3 = repeats[: max_degree3 + 1]
    kernel_eee = jnp.repeat(
        kernel_eee, repeats3, axis=0, total_repeat_length=l3
    )
    kernel_ooe = jnp.repeat(
        kernel_ooe, repeats3, axis=0, total_repeat_length=l3
    )
    kernel_eoo = jnp.repeat(
        kernel_eoo, repeats3, axis=0, total_repeat_length=l3
    )
    kernel_oeo = jnp.repeat(
        kernel_oeo, repeats3, axis=0, total_repeat_length=l3
    )

    def _split_into_even_and_odd_components(
        x: Union[
            Float[Array, '... 1 (max_degree+1)**2 num_features'],
            Float[Array, '... 2 (max_degree+1)**2 num_features'],
        ],
        l: int,  # l = (desired_max_degree+1)**2
    ) -> Tuple[
        Float[Array, '... l num_features'],
        Float[Array, '... l num_features'],
    ]:
      if x.shape[-3] == 2:  # Different parities are already nicely separated.
        return x[..., 0, :, :], x[..., 1, :, :]
      else:  # Extract even and odd components with masking.
        x = jnp.squeeze(x, axis=-3)  # Squeeze parity channel.
        return x * even_mask[:l, :], x * odd_mask[:l, :]

    # Split inputs into even and odd components.
    e1, o1 = _split_into_even_and_odd_components(inputs1, l1)
    e2, o2 = _split_into_even_and_odd_components(inputs2, l2)

    # Convert inputs into "matrix basis".
    def _to_matrix(
        x: Float[Array, '... (max_degree+1)**2 num_features'],
        kernel: Float[Array, '(max_degree+1)**2 num_features'],
        ls: int,
    ) -> Float[Array, '... 2*matrix_degree+1 2*matrix_degree+1 num_features']:
      """Helper function for converting to matrix basis."""
      return jnp.einsum(
          '...nf,nf,lmn->...lmf',
          x,
          kernel,
          cg[..., :ls],
          precision=self.precision,
          optimize='optimal',
      )

    e1 = _to_matrix(e1, kernel_e1, l1)
    o1 = _to_matrix(o1, kernel_o1, l1)
    e2 = _to_matrix(e2, kernel_e2, l2)
    o2 = _to_matrix(o2, kernel_o2, l2)

    # Compute the different coupling paths (matrix multiplication).
    def _couple(
        x1: Float[
            Array, '... 2*matrix_degree+1 2*matrix_degree+1 num_features'
        ],
        x2: Float[
            Array, '... 2*matrix_degree+1 2*matrix_degree+1 num_features'
        ],
    ) -> Float[Array, '... 2*matrix_degree+1 2*matrix_degree+1 num_features']:
      """Helper function for computing coupling paths."""
      return jnp.einsum(
          '...lmf,...mnf->...lnf',
          x1,
          x2,
          precision=self.precision,
          optimize='optimal',
      )

    eee = _couple(e1, e2)
    ooe = _couple(o1, o2)
    eoo = _couple(e1, o2)
    oeo = _couple(o1, e2)

    # Convert results back into "vector basis".
    def _to_vector(
        x: Float[Array, '... 2*matrix_degree+1 2*matrix_degree+1 num_features'],
        kernel: Float[Array, '(max_degree+1)**2 num_features'],
    ) -> Float[Array, '... (max_degree+1)**2 num_features']:
      """Helper function for converting to vector basis."""
      return jnp.einsum(
          '...lmf,nf,lmn->...nf',
          x,
          kernel,
          cg[..., :l3],
          precision=self.precision,
          optimize='optimal',
      )

    eee = _to_vector(eee, kernel_eee)
    ooe = _to_vector(ooe, kernel_ooe)
    eoo = _to_vector(eoo, kernel_eoo)
    oeo = _to_vector(oeo, kernel_oeo)

    # Combine same parities (even/odd).
    e3 = eee + ooe
    o3 = eoo + oeo

    # Combine even and odd output features (usual feature shape conventions).
    if include_pseudotensors:
      return jnp.stack((e3, o3), axis=-3)
    else:
      return jnp.expand_dims(
          e3 * even_mask[:l3, :] + o3 * odd_mask[:l3, :], axis=-3
      )


def _create_tensor(
    use_fused_tensor: bool,
    tensor_kernel_init: Optional[
        Union[InitializerFn, FusedTensorInitializerFn]
    ] = None,
) -> Any:
  """Helper function for creating either FusedTensor or Tensor modules."""
  if use_fused_tensor:
    return functools.partial(
        FusedTensor,
        name='fused_tensor',
        kernel_init=(
            default_fused_tensor_kernel_init
            if tensor_kernel_init is None
            else tensor_kernel_init
        ),
    )
  else:
    return functools.partial(
        Tensor,
        name='tensor',
        kernel_init=(
            default_tensor_kernel_init
            if tensor_kernel_init is None
            else tensor_kernel_init
        ),
    )


class TensorDense(nn.Module):
  r"""Linear projection followed by a tensor product.

  This module first applies a :class:`Dense` layer to linearly combine the input
  features to two different projections, which are then coupled across the
  degree dimension with a tensor product. The transformation can be written as

  .. math::

    \mathbf{a} &= \mathrm{dense}_1(\mathbf{x}) \\
    \mathbf{b} &= \mathrm{dense}_2(\mathbf{x}) \\
    \mathbf{y} &= \mathrm{tensor}(\mathbf{a}, \mathbf{b}) \\

  where
  :math:`\mathbf{x} \in \mathbb{R}^{P_{\mathrm{in}}\times (L_{\mathrm{in}}+1)^2 \times F_{\mathrm{in}}}`
  is the input and
  :math:`\mathbf{y} \in \mathbb{R}^{P_{\mathrm{out}}\times (L_{\mathrm{out}}+1)^2 \times F_{\mathrm{out}}}`
  is the output. The :math:`\mathrm{tensor}` transformation corresponds to
  either a :class:`Tensor` (``use_fused_tensor=False``) or a
  :class:`FusedTensor` (``use_fused_tensor=True``) layer.
  :math:`P_{\mathrm{out}}` is either :math:`1`
  (``include_pseudotensors=False``) or :math:`2`
  (``include_pseudotensors=True``).

  Attributes:
    features: The number of output features :math:`F_{\mathrm{out}}`. If not
      given, keeps the same number features as the input
      :math:`F_{\mathrm{in}}`.
    max_degree: Maximum degree :math:`L_{\mathrm{out}}` of the output. If not
      given, keeps the same max_degree :math:`L_{\mathrm{in}}` as the input.
    use_bias: Whether to use a bias for the :class:`Dense` layer.
    include_pseudotensors: If ``False``, all coupling paths that produce
      pseudotensors are omitted.
    cartesian_order: If ``True``, Cartesian order is assumed.
    use_fused_tensor: If ``True``, :class:`FusedTensor` is used instead of
      :class:`Tensor` for computing the tensor product.
    dtype: The dtype of the computation.
    param_dtype: The dtype passed to parameter initializers.
    precision: Numerical precision of the computation, see `jax.lax.Precision`
      for details.
    dense_kernel_init: Initializer function for the weight matrix of the Dense
      layer.
    dense_bias_init: Initializer function for the bias of the Dense layer.
    tensor_kernel_init: Initializer function for the weight matrix of the Tensor
      layer.
  """

  features: Optional[int] = None
  max_degree: Optional[int] = None
  use_bias: bool = True
  include_pseudotensors: bool = True
  cartesian_order: bool = Config.cartesian_order
  use_fused_tensor: bool = Config.use_fused_tensor
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32
  precision: PrecisionLike = None
  dense_kernel_init: InitializerFn = default_kernel_init
  dense_bias_init: InitializerFn = jax.nn.initializers.zeros
  tensor_kernel_init: Optional[
      Union[InitializerFn, FusedTensorInitializerFn]
  ] = None

  @nn.compact
  def __call__(
      self,
      inputs: Union[
          Float[Array, '... 1 (in_max_degree+1)**2 in_features'],
          Float[Array, '... 2 (in_max_degree+1)**2 in_features'],
      ],
  ) -> Union[
      Float[Array, '... 1 (out_max_degree+1)**2 out_features'],
      Float[Array, '... 2 (out_max_degree+1)**2 out_features'],
  ]:
    """Computes the tensor product of two linear projections of inputs.

    Args:
      inputs: The input features to be transformed.

    Returns:
      The tensor product of two different linear projections of the input
      features.
    """

    # Extract features from size of axis -1 if it is not given.
    features = inputs.shape[-1] if self.features is None else self.features

    # Compute two separate linear projections.
    x1, x2 = jnp.split(
        Dense(
            features=2 * features,
            use_bias=self.use_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            kernel_init=self.dense_kernel_init,
            bias_init=self.dense_bias_init,
            name='dense',
        )(inputs),
        indices_or_sections=2,
        axis=-1,
    )

    # Return tensor product of both projections.
    return _create_tensor(
        use_fused_tensor=self.use_fused_tensor,
        tensor_kernel_init=self.tensor_kernel_init,
    )(
        max_degree=self.max_degree,
        include_pseudotensors=self.include_pseudotensors,
        cartesian_order=self.cartesian_order,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        precision=self.precision,
    )(
        x1, x2
    )


class _Conv(nn.Module):
  r"""Basic "continuous convolution" layer.

  This layer is not meant to be used directly, but provides common functionality
  to other higher-level modules.

  Attributes:
    max_degree: Maximum degree of the output. If not given, the max_degree is
      chosen as the maximum of the max_degree of inputs and basis.
    use_basis_bias: Whether to add a bias to the linear combination of basis
      functions.
    include_pseudotensors: If False, all coupling paths that produce
      pseudotensors are omitted.
    cartesian_order: If True, Cartesian order is assumed.
    use_fused_tensor: If True, :class:`FusedTensor` is used instead of
      :class:`Tensor` for computing the tensor product.
    dtype: The dtype of the computation.
    param_dtype: The dtype passed to parameter initializers.
    precision: Numerical precision of the computation, see `jax.lax.Precision`
      for details.
    dense_kernel_init: Initializer function for the weight matrix of the Dense
      layer.
    dense_bias_init: Initializer function for the bias of the Dense layer.
    tensor_kernel_init: Initializer function for the weight matrix of the Tensor
      layer.
  """

  max_degree: Optional[int] = None
  use_basis_bias: bool = False
  include_pseudotensors: bool = True
  cartesian_order: bool = Config.cartesian_order
  use_fused_tensor: bool = Config.use_fused_tensor
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32
  precision: PrecisionLike = None
  dense_kernel_init: InitializerFn = default_kernel_init
  dense_bias_init: InitializerFn = jax.nn.initializers.zeros
  tensor_kernel_init: Optional[
      Union[InitializerFn, FusedTensorInitializerFn]
  ] = None

  @nn.compact
  def __call__(
      self,
      inputs: Union[
          Union[
              Float[Array, '... N M 1 (in_max_degree+1)**2 num_features'],
              Float[Array, '... N M 2 (in_max_degree+1)**2 num_features'],
          ],
          Union[
              Float[Array, '... P 1 (in_max_degree+1)**2 num_features'],
              Float[Array, '... P 2 (in_max_degree+1)**2 num_features'],
          ],
      ],
      basis: Optional[
          Union[
              Float[Array, '... N M 1 (basis_max_degree+1)**2 num_basis'],
              Float[Array, '... P 1 (basis_max_degree+1)**2 num_basis'],
          ]
      ] = None,
      *,
      adj_idx: Optional[Integer[Array, '... N M']] = None,
      where: Optional[Bool[Array, '... N M']] = None,
      dst_idx: Optional[Integer[Array, '... P']] = None,
      num_segments: Optional[int] = None,
      indices_are_sorted: bool = False,
  ) -> Union[
      Float[Array, '... N 1 (out_max_degree+1)**2 num_features'],
      Float[Array, '... N 2 (out_max_degree+1)**2 num_features'],
  ]:
    """Applies a basic "continuous convolution".

    This computation requires either a dense or sparse index list to determine
    which entries of the inputs will be summed to which output entry.

    Args:
      inputs: "Expanded" inputs to be summed according to provided index list.
      basis: Optional basis functions, if provided, they are linearly combined
        to a learned "convolutional filter" that is tensor-multiplied with the
        inputs before summation.
      adj_idx: Adjacency indices (dense index list), or None.
      where: Mask to specify which values to sum over, required for dense index
        lists.
      dst_idx: Destination indices (sparse index list), or None.
      num_segments: Number of segments after summation, required for sparse
        index lists.
      indices_are_sorted: If True, dst_idx is assumed to be sorted, which may
        increase performance (only used for sparse index lists).

    Returns:
      The indexed summation over the inputs (if basis is None), or the
      tensorproduct of the inputs with learned filters (if basis is not None).

    Raises:
      RuntimeError: If neither dense nor sparse index lists are provided, or if
        both are provided.
    """

    if basis is not None:
      # Check that shapes of inputs and basis are consistent.
      if inputs.shape[:-3] != basis.shape[:-3]:
        raise ValueError('inputs and basis have incompatible shapes')

      # Generate convolution filters as linear combination of basis.
      filters = Dense(
          features=inputs.shape[-1],
          use_bias=self.use_basis_bias,
          dtype=self.dtype,
          param_dtype=self.param_dtype,
          precision=self.precision,
          kernel_init=self.dense_kernel_init,
          bias_init=self.dense_bias_init,
          name='filter',
      )(basis)

      # Calculate tensor product of convolution filters and inputs.
      products = _create_tensor(
          use_fused_tensor=self.use_fused_tensor,
          tensor_kernel_init=self.tensor_kernel_init,
      )(
          max_degree=self.max_degree,
          include_pseudotensors=self.include_pseudotensors,
          cartesian_order=self.cartesian_order,
          dtype=self.dtype,
          param_dtype=self.param_dtype,
          precision=self.precision,
      )(
          filters, inputs
      )
    else:
      products = inputs

    return ops.indexed_sum(
        inputs=products,
        adj_idx=adj_idx,
        where=where,
        dst_idx=dst_idx,
        num_segments=num_segments,
        indices_are_sorted=indices_are_sorted,
    )


class MessagePass(_Conv):
  r"""Equivariant message-passing step.

  Attributes:
    max_degree: Maximum degree of the output. If not given, the max_degree is
      chosen as the maximum of the max_degree of inputs and basis.
    use_basis_bias: Whether to add a bias to the linear combination of basis
      functions.
    include_pseudotensors: If False, all coupling paths that produce
      pseudotensors are omitted.
    cartesian_order: If True, Cartesian order is assumed.
    use_fused_tensor: If True, :class:`FusedTensor` is used instead of
      :class:`Tensor` for computing the tensor product.
    dtype: The dtype of the computation.
    param_dtype: The dtype passed to parameter initializers.
    precision: Numerical precision of the computation, see `jax.lax.Precision`
      for details.
    dense_kernel_init: Initializer function for the weight matrix of the Dense
      layer.
    dense_bias_init: Initializer function for the bias of the Dense layer.
    tensor_kernel_init: Initializer function for the weight matrix of the Tensor
      layer.
  """

  @nn.compact
  def __call__(  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
      self,
      inputs: Union[
          Float[Array, '... N 1 (in_max_degree+1)**2 num_features'],
          Float[Array, '... N 2 (in_max_degree+1)**2 num_features'],
      ],
      basis: Union[
          Float[Array, '... N M 1 (basis_max_degree+1)**2 num_basis'],
          Float[Array, '... P 1 (basis_max_degree+1)**2 num_basis'],
      ],
      weights: Optional[
          Float[Array, '_*broadcastable_to_gathered_inputs']
      ] = None,
      *,
      adj_idx: Optional[Integer[Array, '... N M']] = None,
      where: Optional[Bool[Array, '... N M']] = None,
      dst_idx: Optional[Integer[Array, '... P']] = None,
      src_idx: Optional[Integer[Array, '... P']] = None,
      num_segments: Optional[int] = None,
      indices_are_sorted: bool = False,
  ) -> Union[
      Float[Array, '... N 1 (out_max_degree+1)**2 num_features'],
      Float[Array, '... N 2 (out_max_degree+1)**2 num_features'],
  ]:
    r"""Applies a single message-passing step.

    This layer computes "messages" :math:`\mathbf{m}` as

    .. math::

      \mathbf{f} &= \mathrm{dense}(\mathbf{b})\\
      \mathbf{m}[i] &= \sum_{j \in \mathcal{N}[i]}
      \mathrm{tensor}(\mathbf{x}[j],\mathbf{f}[ij])

    where :math:`\mathbf{x}[1],\dots,\mathbf{x}[N]` are the :math:`N` input
    features, and :math:`\mathbf{b}` are
    `basis functions <../basis_functions.html>`_ for all relevant interactions
    between pairs :math:`i` and :math:`j` from the :math:`N` inputs. The
    relevant interactions for index :math:`i` are given by the set of
    "neighborhood indices" :math:`\mathcal{N}[i]` specified by either a dense
    (``adj_idx``) or sparse (``dst_idx`` and ``src_idx``)
    `index list <../neighbor_lists.html>`_. The :math:`\mathrm{tensor}`
    transformation corresponds to either a :class:`Tensor`
    (``use_fused_tensor=False``) or a :class:`FusedTensor`
    (``use_fused_tensor=True``) layer. If the ``weights`` argument is not
    ``None``, :math:`\mathbf{m}` is computed as

    .. math::

      \mathbf{m}[i] = \sum_{j \in \mathcal{N}_i} w[ij] \cdot
      \mathrm{tensor}(\mathbf{x}[j],\mathbf{f}[ij])

    instead, where :math:`w[ij]` is the entry of ``weights`` corresponding to
    the interaction between pairs :math:`i` and :math:`j.

    Args:
      inputs: A set of :math:`N` feature representations.
      basis: Basis functions for all relevant interactions between pairs
        :math:`i` and :math:`j` from the :math:`N` inputs (either in dense or
        sparse indexed format).
      weights: Optional weights for interactions between pairs :math:`i` and
        :math:`j`.
      adj_idx: Adjacency indices (dense index list), or `None`.
      where: Mask to specify which values to sum over (only for dense index
        lists). If this is `None`, the `where` mask is auto-determined from
        `inputs`.
      dst_idx: Destination indices (sparse index list), or `None`.
      src_idx: Source indices (sparse index list), or `None`.
      num_segments: Number of segments after summation (only for sparse index
        lists). If this is `None`, `num_segments` is auto-determined from
        `inputs`.
      indices_are_sorted: If `True`, `dst_idx` is assumed to be sorted, which
        may increase performance (only used for sparse index lists).

    Returns:
      The result of the message passing step.

    Raises:
      ValueError: If weights are not `None` and cannot be broadcasted to the
        gathered inputs.
    """

    gathered_inputs = ops.gather_src(
        inputs=inputs, adj_idx=adj_idx, src_idx=src_idx
    )

    # Optionally multiply gathered_inputs with weights (if given).
    if weights is not None:
      # Shape check.
      if not util.is_broadcastable(gathered_inputs.shape, weights.shape):
        raise ValueError(
            f'weights with shape {weights.shape} cannot be broadcasted to '
            f'gathered_inputs with shape {gathered_inputs.shape}'
        )
      gathered_inputs *= weights

    # Auto-determine num segments and where mask for indexed ops (if not given).
    if num_segments is None:
      num_segments = inputs.shape[-4]
    if where is None and adj_idx is not None:
      where = adj_idx < inputs.shape[-4]

    return super().__call__(
        inputs=gathered_inputs,
        basis=basis,
        adj_idx=adj_idx,
        where=where,
        dst_idx=dst_idx,
        num_segments=num_segments,
        indices_are_sorted=indices_are_sorted,
    )


class MultiHeadAttention(_Conv):
  r"""Equivariant multi-head attention.

  Attributes:
    max_degree: Maximum degree of the output. If not given, the max_degree is
      chosen as the maximum of the max_degree of inputs and basis.
    use_basis_bias: Whether to add a bias to the linear combination of basis
      functions.
    include_pseudotensors: If False, all coupling paths that produce
      pseudotensors are omitted.
    cartesian_order: If True, Cartesian order is assumed.
    use_fused_tensor: If True, :class:`FusedTensor` is used instead of
      :class:`Tensor` for computing the tensor product.
    dtype: The dtype of the computation.
    param_dtype: The dtype passed to parameter initializers.
    precision: Numerical precision of the computation, see `jax.lax.Precision`
      for details.
    dense_kernel_init: Initializer function for the weight matrix of the Dense
      layer.
    dense_bias_init: Initializer function for the bias of the Dense layer.
    tensor_kernel_init: Initializer function for the weight matrix of the Tensor
      layer.
    num_heads: Number of attention heads.
    qkv_features: Number of features used for queries, keys and values. If this
      is `None`, the same number of features as in `inputs_q` is used.
    out_features: Number of features for the output. If this is `None`, the same
      number of features as in `inputs_q` is used.
    use_relative_positional_encoding_qk: If this is `True`, relative positional
      encodings are used for computing the dot product between queries and keys.
    use_relative_positional_encoding_v: If this is `True`, a relative positional
      encoding (with respect to the queries) is used for computing the values.
    query_kernel_init: Initializer function for the weight matrix of the
      :class:`Dense` layer for computing queries.
    query_bias_init: Initializer function for the bias terms of the
      :class:`Dense` layer for computing queries.
    query_use_bias: Whether to use bias terms in the :class:`Dense` layer for
      computing queries.
    key_kernel_init: Initializer function for the weight matrix of the
      :class:`Dense` layer for computing keys.
    key_bias_init: Initializer function for the bias terms of the :class:`Dense`
      layer for computing keys.
    key_use_bias: Whether to use bias terms in the :class:`Dense` layer for
      computing keys.
    value_kernel_init: Initializer function for the weight matrix of the
      :class:`Dense` layer for computing values.
    value_bias_init: Initializer function for the bias terms of the
      :class:`Dense` layer for computing values.
    value_use_bias: Whether to use bias terms in the :class:`Dense` layer for
      computing values.
    output_kernel_init: Initializer function for the weight matrix of the
      :class:`Dense` layer for computing outputs.
    output_bias_init: Initializer function for the bias terms of the
      :class:`Dense` layer for computing outputs.
    output_use_bias: Whether to use bias terms in the :class:`Dense` layer for
      computing outputs.
  """

  num_heads: Optional[int] = 1
  qkv_features: Optional[int] = None
  out_features: Optional[int] = None
  use_relative_positional_encoding_qk: bool = True
  use_relative_positional_encoding_v: bool = True
  query_kernel_init: InitializerFn = default_kernel_init
  query_bias_init: InitializerFn = jax.nn.initializers.zeros
  query_use_bias: bool = False
  key_kernel_init: InitializerFn = default_kernel_init
  key_bias_init: InitializerFn = jax.nn.initializers.zeros
  key_use_bias: bool = False
  value_kernel_init: InitializerFn = default_kernel_init
  value_bias_init: InitializerFn = jax.nn.initializers.zeros
  value_use_bias: bool = False
  output_kernel_init: InitializerFn = default_kernel_init
  output_bias_init: InitializerFn = jax.nn.initializers.zeros
  output_use_bias: bool = True

  @nn.compact
  def __call__(  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
      self,
      inputs_q: Union[
          Float[Array, '... N 1 (max_degree+1)**2 q_features'],
          Float[Array, '... N 2 (max_degree+1)**2 q_features'],
      ],
      inputs_kv: Union[
          Float[Array, '... M 1 (max_degree+1)**2 kv_features'],
          Float[Array, '... M 2 (max_degree+1)**2 kv_features'],
      ],
      basis: Optional[
          Union[
              Float[Array, '... N M 1 (basis_max_degree+1)**2 num_basis'],
              Float[Array, '... P 1 (basis_max_degree+1)**2 num_basis'],
          ]
      ] = None,
      cutoff_value: Optional[
          Union[
              Float[Array, '... N M 1 (basis_max_degree+1)**2 num_basis'],
              Float[Array, '... P 1 (basis_max_degree+1)**2 num_basis'],
          ]
      ] = None,
      *,
      adj_idx: Optional[Integer[Array, '... N M']] = None,
      where: Optional[Bool[Array, '... N M']] = None,
      dst_idx: Optional[Integer[Array, '... P']] = None,
      src_idx: Optional[Integer[Array, '... P']] = None,
      num_segments: Optional[int] = None,
      indices_are_sorted: bool = False,
  ) -> Union[
      Float[Array, '... N 1 (max_degree+1)**2 out_features'],
      Float[Array, '... N 2 (max_degree+1)**2 out_features'],
  ]:
    """Applies multi-head attention.

    Args:
      inputs_q: Input features that are used to compute queries.
      inputs_kv: Input features that are used to compute keys and values.
      basis: Basis functions for all relevant interactions between queries and
        keys (either in dense or sparse indexed format).
      cutoff_value: Multiplicative cutoff values that are applied to the "raw"
        softmax values (before normalization), can be used for smooth cutoffs.
      adj_idx: Adjacency indices (dense index list), or `None`.
      where: Mask to specify which values to sum over (only for dense index
        lists). If this is `None`, the `where` mask is auto-determined from
        `inputs_kv`.
      dst_idx: Destination indices (sparse index list), or `None`.
      src_idx: Source indices (sparse index list), or `None`.
      num_segments: Number of segments after summation (only for sparse index
        lists). If this is `None`, `num_segments` is auto-determined from
        `inputs_q`.
      indices_are_sorted: If `True`, `dst_idx` is assumed to be sorted, which
        may increase performance (only used for sparse index lists).

    Returns:
      The result of the multi-head attention computation.

    Raises:
      ValueError: If `inputs_q` and `inputs_kv` have incompatible shapes, or if
        `qkv_features` is not divisible by `num_heads`.
      TypeError: When relative positional encodings are requested, but no input
        for `basis` is provided.
    """

    # Shape check.
    if inputs_q.shape[:-4] != inputs_kv.shape[:-4]:
      raise ValueError('inputs_q and inputs_kv have incompatible shapes')

    # Check that positional encodings are possible.
    if (
        self.use_relative_positional_encoding_qk
        or self.use_relative_positional_encoding_v
    ) and basis is None:
      raise TypeError(
          "when using relative positional encodings, 'basis' is "
          'a required argument, received basis=None'
      )

    # Determine features and check for compatibility with num_heads.
    out_features = (
        inputs_q.shape[-1] if self.out_features is None else self.out_features
    )
    qkv_features = (
        inputs_q.shape[-1] if self.qkv_features is None else self.qkv_features
    )

    if qkv_features % self.num_heads != 0:
      raise ValueError(
          f'qkv_features ({qkv_features}) must be divisible by '
          f'num_heads ({self.num_heads})'
      )

    # For query and key projections (used to calculate the dot product), we have
    # to make sure that the final query and key have the same number of
    # parity/degree channels, or the dot product would be ill-defined.
    max_degree_q = _extract_max_degree_and_check_shape(inputs_q.shape)
    max_degree_k = _extract_max_degree_and_check_shape(inputs_kv.shape)
    max_degree_qk = min(max_degree_q, max_degree_k)
    has_pseudotensors_q = inputs_q.shape[-3] == 2
    has_pseudotensors_k = inputs_kv.shape[-3] == 2
    has_pseudotensors_qk = has_pseudotensors_q and has_pseudotensors_k
    query_inputs = change_max_degree_or_type(
        inputs_q,
        max_degree=max_degree_qk,
        include_pseudotensors=has_pseudotensors_qk,
    )
    key_inputs = change_max_degree_or_type(
        inputs_kv,
        max_degree=max_degree_qk,
        include_pseudotensors=has_pseudotensors_qk,
    )

    # Query, key and value projections.
    query = Dense(
        features=qkv_features,
        use_bias=self.query_use_bias,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        precision=self.precision,
        kernel_init=self.query_kernel_init,
        bias_init=self.query_bias_init,
        name='query',
    )(query_inputs)
    key = Dense(
        features=qkv_features,
        use_bias=self.key_use_bias,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        precision=self.precision,
        kernel_init=self.key_kernel_init,
        bias_init=self.key_bias_init,
        name='key',
    )(key_inputs)
    value = Dense(
        features=qkv_features,
        use_bias=self.value_use_bias,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        precision=self.precision,
        kernel_init=self.value_kernel_init,
        bias_init=self.value_bias_init,
        name='value',
    )(inputs_kv)

    # Split heads -> shape=(..., parity, degrees, features, heads).
    query = jnp.reshape(query, (*query.shape[:-1], -1, self.num_heads))
    key = jnp.reshape(key, (*key.shape[:-1], -1, self.num_heads))

    # Scale query by 1/sqrt(depth) to normalize the dot product.
    depth = math.prod(query.shape[-4:-1])  # parity * degrees * features
    query /= jnp.sqrt(depth).astype(query.dtype)

    # Gather queries and keys according to index lists.
    query = ops.gather_dst(query, adj_idx=adj_idx, dst_idx=dst_idx)
    key = ops.gather_src(key, adj_idx=adj_idx, src_idx=src_idx)

    # Dot product.
    if self.use_relative_positional_encoding_qk:
      # Compute the relative positional encoding for queries and keys from the
      # p=0, l=0 component of the basis.
      num_parity_channels = 2 if has_pseudotensors_qk else 1
      rel_pos_encoding = nn.Dense(
          features=num_parity_channels * (max_degree_qk + 1) * qkv_features,
          use_bias=self.use_basis_bias,
          dtype=self.dtype,
          param_dtype=self.param_dtype,
          precision=self.precision,
          kernel_init=self.dense_kernel_init,
          bias_init=self.dense_bias_init,
          name='relative_positional_encoding',
      )(basis[..., 0, 0, :])

      # Reshape to (..., num_parity_channels, max_degree+1, qkv_features).
      rel_pos_encoding = jnp.reshape(
          rel_pos_encoding,
          (
              *rel_pos_encoding.shape[:-1],
              num_parity_channels,
              max_degree_qk + 1,
              qkv_features,
          ),
      )

      # Duplicate entries for individual degrees to get the shape:
      # (..., num_parity_channels, (max_degree+1)**2, qkv_features).
      with jax.ensure_compile_time_eval():
        idx = _duplication_indices_for_max_degree(max_degree_qk)
      rel_pos_encoding = jnp.take(
          rel_pos_encoding, idx, axis=-2, indices_are_sorted=True
      )

      # Split heads -> shape=(..., parity, degrees, features, heads).
      rel_pos_encoding = jnp.reshape(
          rel_pos_encoding, (*rel_pos_encoding.shape[:-1], -1, self.num_heads)
      )

      # Position encoding weighted dot product.
      dot = jnp.einsum(
          '...plfh,...plfh,...plfh->...h',
          query,
          key,
          rel_pos_encoding,
          precision=self.precision,
          optimize='optimal',
      )
    else:
      # Normal dot product.
      dot = jnp.einsum(
          '...plfh,...plfh->...h',
          query,
          key,
          precision=self.precision,
          optimize='optimal',
      )

    # Auto-determine num segments and where mask for indexed ops (if not given).
    if num_segments is None:
      num_segments = inputs_q.shape[-4]
    if where is None and adj_idx is not None:
      where = adj_idx < inputs_kv.shape[-4]

    # Attention weights.
    weight = jax.vmap(
        functools.partial(
            ops.indexed_softmax,
            multiplicative_mask=cutoff_value,
            adj_idx=adj_idx,
            where=where,
            dst_idx=dst_idx,
            num_segments=num_segments,
            indices_are_sorted=indices_are_sorted,
        ),
        in_axes=-1,
        out_axes=-1,
    )(dot)

    # Duplicate weights for each feature in a head.
    weight = jnp.repeat(weight, qkv_features // self.num_heads, axis=-1)

    # Expand shape of weight for broadcasting (add parity and degree channel).
    weight = jnp.expand_dims(weight, (-2, -3))

    # Expand shape of value by gathering, so that it matches shape of weight.
    value = ops.gather_src(inputs=value, adj_idx=adj_idx, src_idx=src_idx)

    # Attention weighted values (with optional relative positional encoding).
    attention = super().__call__(
        inputs=weight * value,
        basis=basis if self.use_relative_positional_encoding_v else None,
        adj_idx=adj_idx,
        where=where,
        dst_idx=dst_idx,
        num_segments=num_segments,
        indices_are_sorted=indices_are_sorted,
    )

    # Linear combination of individual attention heads.
    outputs = Dense(
        features=out_features,
        use_bias=self.output_use_bias,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        precision=self.precision,
        kernel_init=self.output_kernel_init,
        bias_init=self.output_bias_init,
        name='out',
    )(attention)

    return outputs


class SelfAttention(MultiHeadAttention):
  r"""Equivariant self-attention.

  Attributes:
    max_degree: Maximum degree of the output. If not given, the max_degree is
      chosen as the maximum of the max_degree of inputs and basis.
    use_basis_bias: Whether to add a bias to the linear combination of basis
      functions.
    include_pseudotensors: If False, all coupling paths that produce
      pseudotensors are omitted.
    cartesian_order: If True, Cartesian order is assumed.
    use_fused_tensor: If True, :class:`FusedTensor` is used instead of
      :class:`Tensor` for computing the tensor product.
    dtype: The dtype of the computation.
    param_dtype: The dtype passed to parameter initializers.
    precision: Numerical precision of the computation, see `jax.lax.Precision`
      for details.
    dense_kernel_init: Initializer function for the weight matrix of the Dense
      layer.
    dense_bias_init: Initializer function for the bias of the Dense layer.
    tensor_kernel_init: Initializer function for the weight matrix of the Tensor
      layer.
    num_heads: Number of attention heads.
    qkv_features: Number of features used for queries, keys and values. If this
      is `None`, the same number of features as in `inputs_q` is used.
    out_features: Number of features for the output. If this is `None`, the same
      number of features as in `inputs_q` is used.
    use_relative_positional_encoding_qk: If this is `True`, relative positional
      encodings are used for computing the dot product between queries and keys.
    use_relative_positional_encoding_v: If this is `True`, a relative positional
      encoding (with respect to the queries) is used for computing the values.
    query_kernel_init: Initializer function for the weight matrix of the
      :class:`Dense` layer for computing queries.
    query_bias_init: Initializer function for the bias terms of the
      :class:`Dense` layer for computing queries.
    query_use_bias: Whether to use bias terms in the :class:`Dense` layer for
      computing queries.
    key_kernel_init: Initializer function for the weight matrix of the
      :class:`Dense` layer for computing keys.
    key_bias_init: Initializer function for the bias terms of the :class:`Dense`
      layer for computing keys.
    key_use_bias: Whether to use bias terms in the :class:`Dense` layer for
      computing keys.
    value_kernel_init: Initializer function for the weight matrix of the
      :class:`Dense` layer for computing values.
    value_bias_init: Initializer function for the bias terms of the
      :class:`Dense` layer for computing values.
    value_use_bias: Whether to use bias terms in the :class:`Dense` layer for
      computing values.
    output_kernel_init: Initializer function for the weight matrix of the
      :class:`Dense` layer for computing outputs.
    output_bias_init: Initializer function for the bias terms of the
      :class:`Dense` layer for computing outputs.
    output_use_bias: Whether to use bias terms in the :class:`Dense` layer for
      computing outputs.
  """

  @nn.compact
  def __call__(
      self,
      inputs: Union[
          Float[Array, '... N 1 (max_degree+1)**2 num_features'],
          Float[Array, '... N 2 (max_degree+1)**2 num_features'],
      ],
      basis: Optional[
          Union[
              Float[Array, '... N M 1 (basis_max_degree+1)**2 num_basis'],
              Float[Array, '... P 1 (basis_max_degree+1)**2 num_basis'],
          ]
      ] = None,
      cutoff_value: Optional[
          Union[
              Float[Array, '... N M 1 #(basis_max_degree+1)**2 #num_basis'],
              Float[Array, '... P 1 #(basis_max_degree+1)**2 #num_basis'],
          ]
      ] = None,
      *,
      adj_idx: Optional[Integer[Array, '... N M']] = None,
      where: Optional[Bool[Array, '... N M']] = None,
      dst_idx: Optional[Integer[Array, '... P']] = None,
      src_idx: Optional[Integer[Array, '... P']] = None,
      num_segments: Optional[int] = None,
      indices_are_sorted: bool = False,
  ) -> Union[
      Float[Array, '... N 1 (max_degree+1)**2 num_features'],
      Float[Array, '... N 2 (max_degree+1)**2 num_features'],
  ]:
    """Applies self-attention.

    In principle, self-attention is very similar to message-passing, but
    with an
    additional weight factor for each summand, with the weights summing up
    to 1.
    In contrast, in ordinary message-passing, all summands have an implicit
    weight of 1.

    Args:
      inputs: A set of :math:`N` input features.
      basis: Basis functions for all relevant interactions between pairs
        :math:`i` and :math:`j` from the :math:`N` inputs (either in dense or
        sparse indexed format).
      cutoff_value: Multiplicative cutoff values that are applied to the "raw"
        softmax values (before normalization), can be used for smooth cutoffs.
      adj_idx: Adjacency indices (dense index list), or `None`.
      where:  Mask to specify which values to sum over (only for dense index
        lists). If this is `None`, the `where` mask is auto-determined from
        `inputs`.
      dst_idx:  Destination indices (sparse index list), or `None`.
      src_idx: Source indices (sparse index list), or `None`.
      num_segments: Number of segments after summation (only for sparse index
        lists). If this is `None`, `num_segments` is auto-determined from
        `inputs`.
      indices_are_sorted: If `True`, `dst_idx` is assumed to be sorted, which
        may increase performance (only used for sparse index lists).

    Returns:
      The output of self-attention.
    """
    return super().__call__(
        inputs_q=inputs,
        inputs_kv=inputs,
        basis=basis,
        cutoff_value=cutoff_value,
        adj_idx=adj_idx,
        where=where,
        dst_idx=dst_idx,
        src_idx=src_idx,
        num_segments=num_segments,
        indices_are_sorted=indices_are_sorted,
    )
