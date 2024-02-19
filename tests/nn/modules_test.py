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

import functools
import re
from typing import Any, Optional, Sequence, Union
import e3x
from ..testing import subtests
import jax
import jax.numpy as jnp
import jaxtyping
import pytest

InitializerFn = e3x.nn.initializers.InitializerFn
Array = jaxtyping.Array
Bool = jaxtyping.Bool
Float = jaxtyping.Float
Integer = jaxtyping.Integer
UInt32 = jaxtyping.UInt32
Dtype = Any
Shape = Sequence[Union[int, Any]]
PRNGKey = UInt32[Array, '2']


def test_embed(num_embeddings: int = 4, features: int = 8) -> None:
  embed = e3x.nn.Embed(num_embeddings=num_embeddings, features=features)
  x = jnp.asarray([[0, 1], [2, 3]])
  y, params = embed.init_with_output(jax.random.PRNGKey(0), x)
  assert y.shape == (*x.shape, 1, 1, features)
  assert not jnp.allclose(y[0], y[1], atol=1e-5)
  assert jnp.array_equal(params['params']['embedding'][x], y)


@pytest.mark.parametrize(
    'create_module',
    [
        functools.partial(
            e3x.nn.Dense, bias_init=jax.nn.initializers.normal(stddev=1.0)
        ),
        functools.partial(
            e3x.nn.TensorDense,
            dense_bias_init=jax.nn.initializers.normal(stddev=1.0),
            use_fused_tensor=False,
        ),
        functools.partial(
            e3x.nn.TensorDense,
            dense_bias_init=jax.nn.initializers.normal(stddev=1.0),
            use_fused_tensor=True,
        ),
    ],
)
@pytest.mark.parametrize('num_parity', [1, 2])
def test_dense_and_tensor_dense(
    create_module: Any,
    num_parity: int,
    max_degree: int = 2,
    in_features: int = 8,
    out_features: int = 16,
) -> None:
  x_key, rot_key, init_key = jax.random.split(jax.random.PRNGKey(0), num=3)
  # Random input features.
  x = jax.random.normal(x_key, (num_parity, (max_degree + 1) ** 2, in_features))
  # Random rotation matrix.
  rot = e3x.so3.random_rotation(rot_key)
  wigner_d = e3x.so3.wigner_d(rot, max_degree=max_degree)
  # Rotated and reflected features (for checking equivariance).
  x_rot = e3x.nn.rotate(x, wigner_d)
  x_ref = e3x.nn.reflect(x)
  # Initialize Dense layer.
  module = create_module(features=out_features)
  # Apply module.
  y, params = module.init_with_output(init_key, x)
  y_rot = module.apply(params, x_rot)
  y_ref = module.apply(params, x_ref)
  # Check for equivariance.
  assert jnp.allclose(e3x.nn.rotate(y, wigner_d), y_rot, atol=1e-5)
  assert jnp.allclose(e3x.nn.reflect(y), y_ref, atol=1e-5)


@pytest.mark.parametrize(
    'create_module',
    [
        e3x.nn.Dense,
        functools.partial(
            e3x.nn.TensorDense,
            use_fused_tensor=False,
        ),
    ],
)
@pytest.mark.parametrize('num_parity', [1, 2])
@pytest.mark.parametrize('max_degree', [1, 2])
@pytest.mark.parametrize('in_features', [32, 64, 128])
def test_dense_and_tensor_dense_default_init_preserves_zero_mean_and_unit_variance(
    create_module: Any,
    num_parity: int,
    max_degree: int,
    in_features: int,
    out_features: int = 64,
    num_batch: int = 512,
) -> None:
  x_key, init_key = jax.random.split(jax.random.PRNGKey(0), num=2)
  x = jax.random.normal(
      x_key, (num_batch, num_parity, (max_degree + 1) ** 2, in_features)
  )
  module = create_module(features=out_features)
  y, _ = module.init_with_output(init_key, x)
  for p in range(num_parity):
    for l in range(max_degree + 1):
      mean = jnp.mean(y[..., p, l**2 : (l + 1) ** 2, :])
      std = jnp.std(y[..., p, l**2 : (l + 1) ** 2, :])
      assert jnp.isclose(mean, 0.0, atol=0.25)
      assert jnp.isclose(std, 1.0, atol=0.25)


@pytest.mark.parametrize(
    'max_degree, expected',
    [
        (0, jnp.asarray([0])),
        (1, jnp.asarray([0, 1, 1, 1])),
        (2, jnp.asarray([0, 1, 1, 1, 2, 2, 2, 2, 2])),
        (3, jnp.asarray([0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3])),
    ],
)
def test__duplication_indices_for_max_degree(
    max_degree: int, expected: Integer[Array, '(max_degree+1)**2']
) -> None:
  assert jnp.array_equal(
      e3x.nn.modules._duplication_indices_for_max_degree(max_degree), expected
  )


@pytest.mark.parametrize('create_module', [e3x.nn.FusedTensor, e3x.nn.Tensor])
@pytest.mark.parametrize('num_parity_in1', [1, 2])
@pytest.mark.parametrize('num_parity_in2', [1, 2])
@pytest.mark.parametrize('max_degree_in1', [0, 2])
@pytest.mark.parametrize('max_degree_in2', [1, 2])
@pytest.mark.parametrize('include_pseudotensors', [False, True])
@pytest.mark.parametrize('max_degree', [None, 1])
def test_tensor_and_fused_tensor(
    create_module: Any,
    num_parity_in1: int,
    num_parity_in2: int,
    max_degree_in1: int,
    max_degree_in2: int,
    include_pseudotensors: bool,
    max_degree: Optional[int],
    features: int = 8,
) -> None:
  x1_key, x2_key, rot_key, init_key = jax.random.split(
      jax.random.PRNGKey(0), num=4
  )
  # Random input features.
  x1 = jax.random.normal(
      x1_key, (num_parity_in1, (max_degree_in1 + 1) ** 2, features)
  )
  x2 = jax.random.normal(
      x2_key, (num_parity_in2, (max_degree_in2 + 1) ** 2, features)
  )
  # Random rotation matrix.
  rot = e3x.so3.random_rotation(rot_key)
  wigner_d1 = e3x.so3.wigner_d(rot, max_degree=max_degree_in1)
  wigner_d2 = e3x.so3.wigner_d(rot, max_degree=max_degree_in2)
  # Rotated and reflected features (for checking equivariance).
  x1_rot = e3x.nn.rotate(x1, wigner_d1)
  x2_rot = e3x.nn.rotate(x2, wigner_d2)
  x1_ref = e3x.nn.reflect(x1)
  x2_ref = e3x.nn.reflect(x2)
  # Initialize module.
  module = create_module(
      max_degree=max_degree, include_pseudotensors=include_pseudotensors
  )
  # Apply module.
  y, params = module.init_with_output(init_key, x1, x2)
  y_rot = module.apply(params, x1_rot, x2_rot)
  y_ref = module.apply(params, x1_ref, x2_ref)
  # Check for equivariance.
  wigner_d = e3x.so3.wigner_d(
      rot,
      max_degree=(
          e3x.nn.features._extract_max_degree_and_check_shape(y.shape)
          if max_degree is None
          else max_degree
      ),
  )
  assert jnp.allclose(e3x.nn.rotate(y, wigner_d), y_rot, atol=1e-5)
  assert jnp.allclose(e3x.nn.reflect(y), y_ref, atol=1e-5)


@pytest.mark.parametrize('create_module', [e3x.nn.FusedTensor, e3x.nn.Tensor])
def test_tensor_and_fused_tensor_raises_with_invalid_max_degree(
    create_module: Any,
) -> None:
  x1_key, x2_key, init_key = jax.random.split(jax.random.PRNGKey(0), num=3)
  max_degree1 = 0
  max_degree2 = 1
  max_degree3 = 2
  x1 = jax.random.normal(x1_key, (1, (max_degree1 + 1) ** 2, 1))
  x2 = jax.random.normal(x2_key, (1, (max_degree2 + 1) ** 2, 1))
  with pytest.raises(
      ValueError, match='can be at most 1, received max_degree=2'
  ):
    create_module(max_degree=max_degree3).init(init_key, x1, x2)


@pytest.mark.parametrize('num_parity_in1', [1, 2])
@pytest.mark.parametrize('num_parity_in2', [1, 2])
@pytest.mark.parametrize('max_degree_in1', [0, 2])
@pytest.mark.parametrize('max_degree_in2', [1, 2])
@pytest.mark.parametrize('include_pseudotensors', [False, True])
@pytest.mark.parametrize('max_degree', [0, None, 3])
def test_tensor_default_init_preserves_zero_mean_and_unit_variance(
    num_parity_in1: int,
    num_parity_in2: int,
    max_degree_in1: int,
    max_degree_in2: int,
    include_pseudotensors: bool,
    max_degree: Optional[int],
    features: int = 64,
    num_batch: int = 512,
) -> None:
  # Skip invalid combinations.
  if max_degree is not None and max_degree > max_degree_in1 + max_degree_in2:
    pytest.skip('invalid max_degree for max_degree_in1 and max_degree_in2')

  x1_key, x2_key, init_key = jax.random.split(jax.random.PRNGKey(0), num=3)
  # Random input features.
  x1 = jax.random.normal(
      x1_key, (num_batch, num_parity_in1, (max_degree_in1 + 1) ** 2, features)
  )
  x2 = jax.random.normal(
      x2_key, (num_batch, num_parity_in2, (max_degree_in2 + 1) ** 2, features)
  )
  tensor = e3x.nn.Tensor(
      max_degree=max_degree, include_pseudotensors=include_pseudotensors
  )
  y, _ = tensor.init_with_output(init_key, x1, x2)
  max_degree_out = e3x.nn.features._extract_max_degree_and_check_shape(y.shape)
  for p in range(y.shape[-3]):
    for l in range(max_degree_out + 1):
      mean = jnp.mean(y[..., p, l**2 : (l + 1) ** 2, :])
      std = jnp.std(y[..., p, l**2 : (l + 1) ** 2, :])
      assert jnp.isclose(mean, 0.0, atol=0.25)
      if not (mean == 0.0 and std == 0.0):  # Skip zero-outputs.
        assert jnp.isclose(std, 1.0, atol=0.25)


# Note about this test: The normalization for fused tensor only works perfectly
# when the output and inputs have the same, even max_degree. In other cases, the
# normalization will only be approximate, and only the l=0 channel(s) will have
# unit variance.
@pytest.mark.parametrize('include_pseudotensors', [False, True])
@pytest.mark.parametrize('num_parity_in2', [1, 2])
@pytest.mark.parametrize('num_parity_in1', [1, 2])
@pytest.mark.parametrize('max_degree_in1', [0, 2])
@pytest.mark.parametrize('max_degree_in2', [2, 4])
@pytest.mark.parametrize('max_degree', [0, 2, 4])
def test_fused_tensor_default_init_preserves_zero_mean_and_unit_variance(
    num_parity_in1: int,
    num_parity_in2: int,
    max_degree_in1: int,
    max_degree_in2: int,
    include_pseudotensors: bool,
    max_degree: Optional[int],
    features: int = 8192,
    num_batch: int = 16,
) -> None:
  # Skip invalid combinations.
  if max_degree is not None and max_degree > max_degree_in1 + max_degree_in2:
    pytest.skip('invalid max_degree for max_degree_in1 and max_degree_in2')

  x1_key, x2_key, init_key = jax.random.split(jax.random.PRNGKey(0), num=3)
  # Random input features.
  x1 = jax.random.normal(
      x1_key, (num_batch, num_parity_in1, (max_degree_in1 + 1) ** 2, features)
  )
  x2 = jax.random.normal(
      x2_key, (num_batch, num_parity_in2, (max_degree_in2 + 1) ** 2, features)
  )
  fused_tensor = e3x.nn.FusedTensor(
      max_degree=max_degree, include_pseudotensors=include_pseudotensors
  )
  y, _ = fused_tensor.init_with_output(init_key, x1, x2)

  # Choose max_degree_out (for the loop) to prevent failures for input
  # combinations that are known to not work perfectly.
  if num_parity_in1 == 1 and num_parity_in2 == 1:
    max_degree_out = 0  # Only check scalar channels.
  else:
    max_degree_out = max_degree
  if max_degree % 2 != 0 or not (
      max_degree_in1 == max_degree or max_degree_in2 == max_degree
  ):
    max_degree_out = 0

  for p in range(y.shape[-3]):
    for l in range(max_degree_out + 1):
      mean = jnp.mean(y[..., p, l**2 : (l + 1) ** 2, :])
      std = jnp.std(y[..., p, l**2 : (l + 1) ** 2, :])
      assert jnp.isclose(mean, 0.0, atol=0.25)
      if not (mean == 0.0 and std == 0.0):  # Skip zero-outputs.
        assert jnp.isclose(std, 1.0, atol=0.25)


@subtests({
    'distinguishable by 3-body invariants': dict(
        ra=jnp.asarray([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
        rb=jnp.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        max_degree=1,
        num_iterations=1,
    ),
    'distinguishable by 3-body invariants (l=1 features cancel)': dict(
        ra=jnp.asarray([
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ]),
        rb=jnp.asarray([
            [0.57735027, 0.57735027, 0.57735027],
            [0.57735027, -0.57735027, -0.57735027],
            [-0.57735027, 0.57735027, -0.57735027],
            [-0.57735027, -0.57735027, 0.57735027],
        ]),
        max_degree=2,
        num_iterations=1,
    ),
    'distinguishable by 3-body invariants (l=1-3 features cancel)': dict(
        ra=jnp.asarray([
            [0.57735027, 0.57735027, 0.57735027],
            [0.57735027, -0.57735027, -0.57735027],
            [-0.57735027, 0.57735027, -0.57735027],
            [-0.57735027, -0.57735027, 0.57735027],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ]),
        rb=jnp.asarray([
            [0.57735027, 0.57735027, 0.57735027],
            [0.57735027, -0.57735027, -0.57735027],
            [-0.57735027, 0.57735027, -0.57735027],
            [-0.57735027, -0.57735027, 0.57735027],
            [0.70710678, 0.70710678, 0.0],
            [-0.70710678, -0.70710678, 0.0],
        ]),
        max_degree=4,
        num_iterations=1,
    ),
    'distinguishable by 4-body invariants': dict(
        ra=jnp.asarray([
            [-0.70710678, 0.0, -0.70710678],
            [0.57735027, 0.57735027, 0.57735027],
            [-0.57735027, -0.57735027, 0.57735027],
            [0.0, 0.70710678, -0.70710678],
        ]),
        rb=jnp.asarray([
            [-0.70710678, 0.0, -0.70710678],
            [0.57735027, 0.57735027, 0.57735027],
            [-0.57735027, -0.57735027, 0.57735027],
            [0.0, -0.70710678, -0.70710678],
        ]),
        max_degree=2,  # max_degree=1 is not sufficient.
        num_iterations=2,
    ),
    'distinguishable by 5-body invariants': dict(
        ra=jnp.asarray([
            [-8.41248952e-01, 4.83194237e-01, 2.42535625e-01],
            [-1.79843259e-01, -9.53327264e-01, 2.42535625e-01],
            [9.70142250e-01, 6.97181373e-04, 2.42535625e-01],
            [-9.61505556e-01, 1.29164764e-01, -2.42535625e-01],
            [1.94042744e-01, -9.50538734e-01, -2.42535625e-01],
            [8.97828101e-01, 3.67533907e-01, -2.42535625e-01],
            [0.00000000e00, 0.00000000e00, 1.00000000e00],
        ]),
        rb=jnp.asarray([
            [-8.41248952e-01, 4.83194237e-01, 2.42535625e-01],
            [-1.79843259e-01, -9.53327264e-01, 2.42535625e-01],
            [9.70142250e-01, 6.97181373e-04, 2.42535625e-01],
            [-9.61505556e-01, 1.29164764e-01, -2.42535625e-01],
            [1.94042744e-01, -9.50538734e-01, -2.42535625e-01],
            [8.97828101e-01, 3.67533907e-01, -2.42535625e-01],
            [0.00000000e00, 0.00000000e00, -1.00000000e00],
        ]),
        max_degree=2,
        num_iterations=3,
    ),
})
@pytest.mark.parametrize('create_module', [e3x.nn.FusedTensor, e3x.nn.Tensor])
def test_iterated_tensor_and_fused_tensor_invariants_distinguish_structures(
    create_module: Any,
    ra: Float[Array, '... 3'],
    rb: Float[Array, '... 3'],
    max_degree: int,
    num_iterations: int,
) -> None:
  def _construct_features(
      r: Float[Array, '... 3']
  ) -> Float[Array, '... 1 (max_degree+1)**2 1']:
    """Helper function for constructing features from input vectors."""
    return e3x.nn.change_max_degree_or_type(
        jnp.expand_dims(
            jnp.sum(
                e3x.so3.spherical_harmonics(r, max_degree=max_degree), axis=0
            ),
            axis=(0, -1),
        ),
        include_pseudotensors=True,
    )

  # Construct features a and b from the set of vectors ra and rb.
  a = _construct_features(ra)
  b = _construct_features(rb)
  # Initialize Tensor layer.
  module = create_module()
  params = module.init(jax.random.PRNGKey(0), a, b)
  # Apply tensor product.
  for _ in range(num_iterations):  # (num_iterations+2)-body invariants.
    a = module.apply(params, a, a)
    b = module.apply(params, b, b)
  # Check that invariants (scalar channel) are different.
  assert not jnp.isclose(a[0, 0, 0], b[0, 0, 0], atol=1e-5)


def identity_initializer(key: Any, shape: Any, dtype: Any) -> Any:  # pylint: disable=unused-argument
  """Helper function to initialize weight matrices to identity."""
  assert len(shape) == 2
  assert shape[0] == shape[1]
  return jnp.eye(shape[0], dtype=dtype)


def fused_tensor_ones(
    scale: Union[float, Float[Array, '*Shape']] = 1.0,  # pylint: disable=unused-argument
    mask: Union[bool, Array] = True,
    dtype: Any = jnp.float_,
) -> InitializerFn:
  """Helper function to initialize fused tensor kernels to ones (for testing)."""

  def init(
      key: PRNGKey, shape: Shape, dtype: Dtype = dtype  # pylint: disable=unused-argument
  ) -> Float[Array, '*shape']:
    dtype = jax.dtypes.canonicalize_dtype(dtype)
    params = jnp.ones(shape=shape, dtype=dtype)
    return jnp.where(mask, params, 0)

  return init


@pytest.mark.parametrize('use_fused_tensor', [True, False])
@subtests({
    'dense neighborlist': dict(
        inputs=jnp.asarray([
            [
                [[[0.0, 1.0, -1.0]]],
                [[[1.0, 0.1, 0.5]]],
            ],
            [
                [[[0.0, 0.0, 0.0]]],
                [[[9.9, 9.9, 9.9]]],  # Padding.
            ],
        ]),
        basis=None,
        adj_idx=jnp.asarray([[9, 9], [9, 9]]),  # Only the shape matters here.
        where=jnp.asarray([[True, True], [True, False]]),
        dst_idx=None,
        num_segments=None,
        expected_outputs=jnp.asarray(
            [[[[1.0, 1.1, -0.5]]], [[[0.0, 0.0, 0.0]]]]
        ),
    ),
    'sparse neighborlist': dict(
        inputs=jnp.asarray(
            [[[[0.0, 1.0, -1.0]]], [[[1.0, 0.1, 0.5]]], [[[0.0, 0.0, 0.0]]]]
        ),
        basis=None,
        adj_idx=None,
        where=None,
        dst_idx=jnp.asarray([0, 0, 1]),
        num_segments=2,
        expected_outputs=jnp.asarray(
            [[[[1.0, 1.1, -0.5]]], [[[0.0, 0.0, 0.0]]]]
        ),
    ),
    'with basis': dict(
        inputs=jnp.asarray(
            [[[[0.0, 1.0, -1.0]]], [[[1.0, 0.1, 0.5]]], [[[0.0, 0.0, 0.0]]]]
        ),
        basis=jnp.asarray(
            [[[[1.0, 1.0, 2.0]]], [[[1.5, 0.0, 1.0]]], [[[0.3, 1.0, -0.1]]]]
        ),
        adj_idx=None,
        where=None,
        dst_idx=jnp.asarray([0, 0, 1]),
        num_segments=2,
        expected_outputs=jnp.asarray(
            [[[[1.5, 1.0, -1.5]]], [[[0.0, 0.0, 0.0]]]]
        ),
    ),
})
def test__conv(
    use_fused_tensor: bool,
    inputs: Float[Array, '...'],
    basis: Optional[Float[Array, '...']],
    adj_idx: Optional[Integer[Array, '...']],
    where: Optional[Bool[Array, '...']],
    dst_idx: Optional[Integer[Array, '...']],
    num_segments: Optional[int],
    expected_outputs: Float[Array, '...'],
) -> None:
  conv = e3x.nn._Conv(
      dense_kernel_init=identity_initializer,
      tensor_kernel_init=(
          fused_tensor_ones if use_fused_tensor else jax.nn.initializers.ones
      ),
      use_fused_tensor=use_fused_tensor,
  )
  kwargs = dict(
      inputs=inputs,
      basis=basis,
      adj_idx=adj_idx,
      where=where,
      dst_idx=dst_idx,
      num_segments=num_segments,
  )
  outputs, _ = conv.init_with_output(jax.random.PRNGKey(0), **kwargs)
  assert jnp.allclose(outputs, expected_outputs, atol=1e-5)


@pytest.mark.parametrize('use_fused_tensor', [True, False])
def test__conv_is_equivariant(
    use_fused_tensor: bool,
    num_inputs: int = 3,
    num_parity: int = 2,
    max_degree: int = 2,
    num_features: int = 8,
    num_basis: int = 4,
    dst_idx: Integer[Array, '...'] = jnp.asarray([0, 0, 1]),
    num_segments: int = 2,
) -> None:
  inputs_key, basis_key, rot_key, init_key = jax.random.split(
      jax.random.PRNGKey(0), num=4
  )
  # Random inputs and basis.
  inputs = jax.random.normal(
      inputs_key, (num_inputs, num_parity, (max_degree + 1) ** 2, num_features)
  )
  basis = jax.random.normal(
      basis_key, (num_inputs, 1, (max_degree + 1) ** 2, num_basis)
  )
  # Random rotation matrix.
  rot = e3x.so3.random_rotation(rot_key)
  wigner_d = e3x.so3.wigner_d(rot, max_degree=max_degree)
  # Rotated and reflected inputs/basis (for checking equivariance).
  inputs_rot = e3x.nn.rotate(inputs, wigner_d)
  inputs_ref = e3x.nn.reflect(inputs)
  basis_rot = e3x.nn.rotate(basis, wigner_d)
  basis_ref = e3x.nn.reflect(basis)
  # Initialize _Conv layer.
  conv = e3x.nn._Conv(use_fused_tensor=use_fused_tensor)
  # Apply module.
  outputs, params = conv.init_with_output(
      init_key, inputs, basis, dst_idx=dst_idx, num_segments=num_segments
  )
  outputs_rot = conv.apply(
      params, inputs_rot, basis_rot, dst_idx=dst_idx, num_segments=num_segments
  )
  outputs_ref = conv.apply(
      params, inputs_ref, basis_ref, dst_idx=dst_idx, num_segments=num_segments
  )
  # Check for equivariance.
  assert jnp.allclose(e3x.nn.rotate(outputs, wigner_d), outputs_rot, atol=1e-5)
  assert jnp.allclose(e3x.nn.reflect(outputs), outputs_ref, atol=1e-5)


@pytest.mark.parametrize('use_fused_tensor', [True, False])
@subtests({
    'dense neighborlist': dict(
        inputs=jnp.asarray([
            [[[-0.5, -0.5]]],
            [[[1.0, 0.0]]],
            [[[0.0, 1.0]]],
        ]),
        basis=jnp.asarray([
            [[[[1.0]]], [[[1.0]]]],
            [[[[1.0]]], [[[1.0]]]],
            [[[[1.0]]], [[[1.0]]]],
        ]),
        weights=None,
        adj_idx=jnp.asarray([[1, 2], [0, 3], [0, 2]]),
        dst_idx=None,
        src_idx=None,
        expected_outputs=jnp.asarray([
            [[[1.0, 1.0]]],
            [[[-0.5, -0.5]]],
            [[[-0.5, 0.5]]],
        ]),
    ),
    'dense neighborlist with weights': dict(
        inputs=jnp.asarray([
            [[[-0.5, -0.5]]],
            [[[1.0, 0.0]]],
            [[[0.0, 1.0]]],
        ]),
        basis=jnp.asarray([
            [[[[1.0]]], [[[1.0]]]],
            [[[[1.0]]], [[[1.0]]]],
            [[[[1.0]]], [[[1.0]]]],
        ]),
        weights=jnp.asarray([
            [[[[0.3]]], [[[1.2]]]],
            [[[[0.2]]], [[[9.9]]]],
            [[[[0.2]]], [[[1.0]]]],
        ]),
        adj_idx=jnp.asarray([[1, 2], [0, 3], [0, 2]]),
        dst_idx=None,
        src_idx=None,
        expected_outputs=jnp.asarray([
            [[[0.3, 1.2]]],
            [[[-0.1, -0.1]]],
            [[[-0.1, 0.9]]],
        ]),
    ),
    'sparse neighborlist': dict(
        inputs=jnp.asarray([
            [[[-0.5, -0.5]]],
            [[[1.0, 0.0]]],
            [[[0.0, 1.0]]],
        ]),
        basis=jnp.asarray([
            [[[1.0]]],
            [[[1.0]]],
            [[[1.0]]],
            [[[1.0]]],
            [[[1.0]]],
        ]),
        weights=None,
        adj_idx=None,
        dst_idx=jnp.asarray([0, 0, 1, 2, 2]),
        src_idx=jnp.asarray([1, 2, 0, 0, 2]),
        expected_outputs=jnp.asarray([
            [[[1.0, 1.0]]],
            [[[-0.5, -0.5]]],
            [[[-0.5, 0.5]]],
        ]),
    ),
    'sparse neighborlist with weights': dict(
        inputs=jnp.asarray([
            [[[-0.5, -0.5]]],
            [[[1.0, 0.0]]],
            [[[0.0, 1.0]]],
        ]),
        basis=jnp.asarray([
            [[[1.0]]],
            [[[1.0]]],
            [[[1.0]]],
            [[[1.0]]],
            [[[1.0]]],
        ]),
        weights=jnp.asarray([
            [[[0.3]]],
            [[[1.2]]],
            [[[0.2]]],
            [[[0.2]]],
            [[[1.0]]],
        ]),
        adj_idx=None,
        dst_idx=jnp.asarray([0, 0, 1, 2, 2]),
        src_idx=jnp.asarray([1, 2, 0, 0, 2]),
        expected_outputs=jnp.asarray([
            [[[0.3, 1.2]]],
            [[[-0.1, -0.1]]],
            [[[-0.1, 0.9]]],
        ]),
    ),
})
def test_message_pass(
    use_fused_tensor: bool,
    inputs: Float[Array, '...'],
    basis: Float[Array, '...'],
    weights: Optional[Float[Array, '...']],
    adj_idx: Optional[Integer[Array, '...']],
    dst_idx: Optional[Integer[Array, '...']],
    src_idx: Optional[Integer[Array, '...']],
    expected_outputs: Float[Array, '...'],
) -> None:
  message_pass = e3x.nn.MessagePass(
      dense_kernel_init=jax.nn.initializers.ones,
      tensor_kernel_init=(
          fused_tensor_ones if use_fused_tensor else jax.nn.initializers.ones
      ),
      use_fused_tensor=use_fused_tensor,
  )
  kwargs = dict(
      inputs=inputs,
      weights=weights,
      basis=basis,
      adj_idx=adj_idx,
      dst_idx=dst_idx,
      src_idx=src_idx,
  )
  outputs, _ = message_pass.init_with_output(jax.random.PRNGKey(0), **kwargs)
  assert jnp.allclose(outputs, expected_outputs, atol=1e-5)


@pytest.mark.parametrize('use_fused_tensor', [True, False])
def test_message_pass_is_equivariant(
    use_fused_tensor: bool,
    num_inputs: int = 3,
    num_parity: int = 2,
    max_degree: int = 2,
    num_features: int = 8,
    num_basis: int = 4,
    dst_idx: Integer[Array, '...'] = jnp.asarray([0, 0, 1]),
    src_idx: Integer[Array, '...'] = jnp.asarray([0, 1, 0]),
) -> None:
  inputs_key, basis_key, rot_key, init_key = jax.random.split(
      jax.random.PRNGKey(0), num=4
  )
  # Random inputs and basis.
  inputs = jax.random.normal(
      inputs_key, (num_inputs, num_parity, (max_degree + 1) ** 2, num_features)
  )
  basis = jax.random.normal(
      basis_key, (len(dst_idx), 1, (max_degree + 1) ** 2, num_basis)
  )
  # Random rotation matrix.
  rot = e3x.so3.random_rotation(rot_key)
  wigner_d = e3x.so3.wigner_d(rot, max_degree=max_degree)
  # Rotated and reflected inputs/basis (for checking equivariance).
  inputs_rot = e3x.nn.rotate(inputs, wigner_d)
  inputs_ref = e3x.nn.reflect(inputs)
  basis_rot = e3x.nn.rotate(basis, wigner_d)
  basis_ref = e3x.nn.reflect(basis)
  # Initialize MessagePass layer.
  message_pass = e3x.nn.MessagePass(use_fused_tensor=use_fused_tensor)
  # Apply module.
  outputs, params = message_pass.init_with_output(
      init_key, inputs, basis, dst_idx=dst_idx, src_idx=src_idx
  )
  outputs_rot = message_pass.apply(
      params, inputs_rot, basis_rot, dst_idx=dst_idx, src_idx=src_idx
  )
  outputs_ref = message_pass.apply(
      params, inputs_ref, basis_ref, dst_idx=dst_idx, src_idx=src_idx
  )
  # Check for equivariance.
  assert jnp.allclose(e3x.nn.rotate(outputs, wigner_d), outputs_rot, atol=1e-5)
  assert jnp.allclose(e3x.nn.reflect(outputs), outputs_ref, atol=1e-5)


@pytest.mark.parametrize('use_fused_tensor', [True, False])
@subtests({
    'dense': dict(
        use_relative_positional_encoding_qk=False,
        use_relative_positional_encoding_v=False,
        inputs_q=jnp.asarray([[[[1.0, 0.0]]], [[[0.0, 1.0]]]]),
        inputs_kv=jnp.asarray([[[[0.5, 0.5]]], [[[3.0, 0.0]]]]),
        basis=None,
        cutoff_value=None,
        adj_idx=jnp.asarray([[0, 1, 2], [0, 1, 2]]),
        dst_idx=None,
        src_idx=None,
        expected_outputs=jnp.asarray(
            [[[[2.6354492, 0.07291014]]], [[[1.5313025, 0.2937395]]]]
        ),
    ),
    'cross-dense': dict(
        use_relative_positional_encoding_qk=False,
        use_relative_positional_encoding_v=False,
        inputs_q=jnp.asarray([[[[1.0, 0.0]]], [[[0.0, 1.0]]]]),
        inputs_kv=jnp.asarray([
            [
                [[0.5, 0.5], [0.5, 0.5], [1.0, 0.0], [0.0, -1.0]],
                [[3.0, 0.0], [3.0, 0.0], [0.0, -1.0], [1.0, 0.0]],
            ],
            [
                [[3.0, 0.0], [3.0, 0.0], [0.0, -1.0], [1.0, 0.0]],
                [[0.5, 0.5], [0.5, 0.5], [1.0, 0.0], [0.0, -1.0]],
            ],
            [
                [[9.9, 9.9], [9.9, 9.9], [9.9, 9.9], [9.9, 9.9]],
                [[9.9, 9.9], [9.9, 9.9], [9.9, 9.9], [9.9, 9.9]],
            ],
        ]),
        basis=None,
        cutoff_value=None,
        adj_idx=jnp.asarray([[0, 1], [0, 1]]),
        dst_idx=None,
        src_idx=None,
        expected_outputs=jnp.asarray([
            [
                [
                    [2.6354492, 0.07291014],
                    [2.6354492, 0.07291014],
                    [0.14582027, -0.85417974],
                    [0.85417974, -0.14582027],
                ],
                [
                    [0.8645507, 0.42708987],
                    [0.8645507, 0.42708987],
                    [0.85417974, -0.14582027],
                    [0.14582027, -0.85417974],
                ],
            ],
            [
                [
                    [1.5313025, 0.2937395],
                    [1.5313025, 0.2937395],
                    [0.587479, -0.412521],
                    [0.412521, -0.587479],
                ],
                [
                    [1.9686975, 0.2062605],
                    [1.9686975, 0.2062605],
                    [0.412521, -0.587479],
                    [0.587479, -0.412521],
                ],
            ],
        ]),
    ),
    'sparse': dict(
        use_relative_positional_encoding_qk=False,
        use_relative_positional_encoding_v=False,
        inputs_q=jnp.asarray([[[[1.0, 0.0]]], [[[0.0, 1.0]]]]),
        inputs_kv=jnp.asarray([[[[0.5, 0.5]]], [[[3.0, 0.0]]]]),
        basis=None,
        cutoff_value=None,
        adj_idx=None,
        dst_idx=jnp.asarray([0, 0, 1, 1]),
        src_idx=jnp.asarray([0, 1, 0, 1]),
        expected_outputs=jnp.asarray(
            [[[[2.6354492, 0.07291014]]], [[[1.5313025, 0.2937395]]]]
        ),
    ),
    'cross-sparse': dict(
        use_relative_positional_encoding_qk=False,
        use_relative_positional_encoding_v=False,
        inputs_q=jnp.asarray([[[[1.0, 0.0]]], [[[0.0, 1.0]]]]),
        inputs_kv=jnp.asarray([
            [
                [[0.5, 0.5], [0.5, 0.5], [1.0, 0.0], [0.0, -1.0]],
                [[3.0, 0.0], [3.0, 0.0], [0.0, -1.0], [1.0, 0.0]],
            ],
            [
                [[3.0, 0.0], [3.0, 0.0], [0.0, -1.0], [1.0, 0.0]],
                [[0.5, 0.5], [0.5, 0.5], [1.0, 0.0], [0.0, -1.0]],
            ],
            [
                [[9.9, 9.9], [9.9, 9.9], [9.9, 9.9], [9.9, 9.9]],
                [[9.9, 9.9], [9.9, 9.9], [9.9, 9.9], [9.9, 9.9]],
            ],
        ]),
        basis=None,
        cutoff_value=None,
        adj_idx=None,
        dst_idx=jnp.asarray([0, 0, 1, 1]),
        src_idx=jnp.asarray([0, 1, 0, 1]),
        expected_outputs=jnp.asarray([
            [
                [
                    [2.6354492, 0.07291014],
                    [2.6354492, 0.07291014],
                    [0.14582027, -0.85417974],
                    [0.85417974, -0.14582027],
                ],
                [
                    [0.8645507, 0.42708987],
                    [0.8645507, 0.42708987],
                    [0.85417974, -0.14582027],
                    [0.14582027, -0.85417974],
                ],
            ],
            [
                [
                    [1.5313025, 0.2937395],
                    [1.5313025, 0.2937395],
                    [0.587479, -0.412521],
                    [0.412521, -0.587479],
                ],
                [
                    [1.9686975, 0.2062605],
                    [1.9686975, 0.2062605],
                    [0.412521, -0.587479],
                    [0.587479, -0.412521],
                ],
            ],
        ]),
    ),
    'with positional encoding qk': dict(
        use_relative_positional_encoding_qk=True,
        use_relative_positional_encoding_v=False,
        inputs_q=jnp.asarray([[[[1.0, 0.0]]], [[[0.0, 1.0]]]]),
        inputs_kv=jnp.asarray([[[[0.5, 0.5]]], [[[3.0, 0.0]]]]),
        basis=jnp.asarray([
            [[[-999, -999]]],
            [[[1.0, 1.0]]],
            [[[1.0, 1.0]]],
            [[[1.0, 1.0]]],
        ]),
        cutoff_value=None,
        adj_idx=None,
        dst_idx=jnp.asarray([0, 0, 1, 1]),
        src_idx=jnp.asarray([0, 1, 0, 1]),
        expected_outputs=jnp.asarray(
            [[[[3.0, 0.0]]], [[[1.5313025, 0.2937395]]]]
        ),
    ),
    'with positional encoding v': dict(
        use_relative_positional_encoding_qk=False,
        use_relative_positional_encoding_v=True,
        inputs_q=jnp.asarray([[[[1.0, 0.0]]], [[[0.0, 1.0]]]]),
        inputs_kv=jnp.asarray([[[[0.5, 0.5]]], [[[3.0, 0.0]]]]),
        basis=jnp.asarray([
            [[[1.0, 1.0]]],
            [[[1.0, 1.0]]],
            [[[0.5, 0.5]]],
            [[[0.5, 0.5]]],
        ]),
        cutoff_value=None,
        adj_idx=None,
        dst_idx=jnp.asarray([0, 0, 1, 1]),
        src_idx=jnp.asarray([0, 1, 0, 1]),
        expected_outputs=jnp.asarray([
            [[[2.6354492, 0.07291014]]],
            [[[0.5 * 1.5313025, 0.5 * 0.2937395]]],
        ]),
    ),
    'with cutoff': dict(
        use_relative_positional_encoding_qk=False,
        use_relative_positional_encoding_v=False,
        inputs_q=jnp.asarray([[[[1.0, 0.0]]], [[[0.0, 1.0]]]]),
        inputs_kv=jnp.asarray([[[[0.5, 0.5]]], [[[3.0, 0.0]]]]),
        basis=None,
        cutoff_value=jnp.asarray([0.0, 1.0, 1.0, 0.0]),
        adj_idx=None,
        dst_idx=jnp.asarray([0, 0, 1, 1]),
        src_idx=jnp.asarray([0, 1, 0, 1]),
        expected_outputs=jnp.asarray([
            [[[3.0, 0.0]]],
            [[[0.5, 0.5]]],
        ]),
    ),
})
def test_multi_head_attention(
    use_fused_tensor: bool,
    use_relative_positional_encoding_qk: bool,
    use_relative_positional_encoding_v: bool,
    inputs_q: Float[Array, '...'],
    inputs_kv: Float[Array, '...'],
    basis: Optional[Float[Array, '...']],
    cutoff_value: Optional[Float[Array, '...']],
    adj_idx: Optional[Integer[Array, '...']],
    dst_idx: Optional[Integer[Array, '...']],
    src_idx: Optional[Integer[Array, '...']],
    expected_outputs: Float[Array, '...'],
    num_heads: int = 1,
) -> None:
  attention = e3x.nn.MultiHeadAttention(
      num_heads=num_heads,
      use_relative_positional_encoding_qk=use_relative_positional_encoding_qk,
      use_relative_positional_encoding_v=use_relative_positional_encoding_v,
      query_kernel_init=identity_initializer,
      key_kernel_init=identity_initializer,
      value_kernel_init=identity_initializer,
      output_kernel_init=identity_initializer,
      dense_kernel_init=identity_initializer,
      tensor_kernel_init=(
          fused_tensor_ones if use_fused_tensor else jax.nn.initializers.ones
      ),
      use_fused_tensor=use_fused_tensor,
  )
  outputs, _ = attention.init_with_output(
      jax.random.PRNGKey(0),
      inputs_q=inputs_q,
      inputs_kv=inputs_kv,
      basis=basis,
      cutoff_value=cutoff_value,
      adj_idx=adj_idx,
      dst_idx=dst_idx,
      src_idx=src_idx,
  )
  assert jnp.allclose(outputs, expected_outputs, atol=1e-5)


@pytest.mark.parametrize('use_fused_tensor', [True, False])
def test_multi_head_attention_is_equivariant(
    use_fused_tensor: bool,
    num_inputs: int = 3,
    num_parity: int = 2,
    max_degree: int = 1,
    num_heads: int = 2,
    num_features: int = 8,
    num_basis: int = 4,
    dst_idx: Integer[Array, '...'] = jnp.asarray([0, 0, 1]),
    src_idx: Integer[Array, '...'] = jnp.asarray([0, 1, 0]),
) -> None:
  inputs_q_key, inputs_kv_key, basis_key, rot_key, init_key = jax.random.split(
      jax.random.PRNGKey(0), num=5
  )
  # Random inputs and basis.
  inputs_q = jax.random.normal(
      inputs_q_key,
      (num_inputs, num_parity, (max_degree + 1) ** 2, num_features),
  )
  inputs_kv = jax.random.normal(
      inputs_kv_key,
      (num_inputs, num_parity, (max_degree + 1) ** 2, num_features),
  )
  basis = jax.random.normal(
      basis_key, (len(dst_idx), 1, (max_degree + 1) ** 2, num_basis)
  )
  # Random rotation matrix.
  rot = e3x.so3.random_rotation(rot_key)
  wigner_d = e3x.so3.wigner_d(rot, max_degree=max_degree)
  # Rotated and reflected inputs/basis (for checking equivariance).
  inputs_q_rot = e3x.nn.rotate(inputs_q, wigner_d)
  inputs_q_ref = e3x.nn.reflect(inputs_q)
  inputs_kv_rot = e3x.nn.rotate(inputs_kv, wigner_d)
  inputs_kv_ref = e3x.nn.reflect(inputs_kv)
  basis_rot = e3x.nn.rotate(basis, wigner_d)
  basis_ref = e3x.nn.reflect(basis)
  # Initialize MultiHeadAttention layer.
  attention = e3x.nn.MultiHeadAttention(
      num_heads=num_heads, use_fused_tensor=use_fused_tensor
  )
  # Apply module.
  outputs, params = attention.init_with_output(
      init_key,
      inputs_q=inputs_q,
      inputs_kv=inputs_kv,
      basis=basis,
      dst_idx=dst_idx,
      src_idx=src_idx,
  )
  outputs_rot = attention.apply(
      params,
      inputs_q=inputs_q_rot,
      inputs_kv=inputs_kv_rot,
      basis=basis_rot,
      dst_idx=dst_idx,
      src_idx=src_idx,
  )
  outputs_ref = attention.apply(
      params,
      inputs_q=inputs_q_ref,
      inputs_kv=inputs_kv_ref,
      basis=basis_ref,
      dst_idx=dst_idx,
      src_idx=src_idx,
  )
  # Check for equivariance.
  assert jnp.allclose(e3x.nn.rotate(outputs, wigner_d), outputs_rot, atol=1e-5)
  assert jnp.allclose(e3x.nn.reflect(outputs), outputs_ref, atol=1e-5)


def test_multi_head_attention_raises_if_q_kv_shapes_are_incompatible() -> None:
  with pytest.raises(
      ValueError, match='inputs_q and inputs_kv have incompatible shapes'
  ):
    e3x.nn.MultiHeadAttention().init(
        jax.random.PRNGKey(0),
        inputs_q=jnp.zeros((10, 2, 1, 1, 1)),
        inputs_kv=jnp.zeros((11, 1, 1, 2, 1)),
        basis=jnp.zeros((10, 2, 1, 1, 1)),
    )


def test_multi_head_attention_raises_if_basis_is_required_but_none() -> None:
  with pytest.raises(TypeError, match='received basis=None'):
    e3x.nn.MultiHeadAttention(use_relative_positional_encoding_qk=True).init(
        jax.random.PRNGKey(0),
        inputs_q=jnp.zeros((1, 1, 1, 1)),
        inputs_kv=jnp.zeros((1, 1, 1, 1)),
        basis=None,
    )


def test_multi_head_attention_raises_if_qkv_features_not_divisible_by_num_heads() -> (
    None
):
  with pytest.raises(
      ValueError,
      match=re.escape('qkv_features (4) must be divisible by num_heads (3)'),
  ):
    e3x.nn.MultiHeadAttention(qkv_features=4, num_heads=3).init(
        jax.random.PRNGKey(0),
        inputs_q=jnp.zeros((1, 1, 1, 1)),
        inputs_kv=jnp.zeros((1, 1, 1, 1)),
        basis=jnp.zeros((1, 1, 1, 1)),
    )


@pytest.mark.parametrize('use_fused_tensor', [True, False])
@subtests({
    'sparse': dict(
        use_relative_positional_encoding_qk=False,
        use_relative_positional_encoding_v=False,
        inputs=jnp.asarray([[[[1.0, 0.0]]], [[[0.0, 1.0]]]]),
        basis=None,
        cutoff_value=None,
        adj_idx=None,
        dst_idx=jnp.asarray([0, 1]),
        src_idx=jnp.asarray([1, 0]),
        expected_outputs=jnp.asarray([[[[0.0, 1.0]]], [[[1.0, 0.0]]]]),
    ),
    'dense': dict(
        use_relative_positional_encoding_qk=False,
        use_relative_positional_encoding_v=False,
        inputs=jnp.asarray([[[[1.0, 0.0]]], [[[0.0, 1.0]]]]),
        basis=None,
        cutoff_value=None,
        adj_idx=jnp.asarray([[1], [0]]),
        dst_idx=None,
        src_idx=None,
        expected_outputs=jnp.asarray([[[[0.0, 1.0]]], [[[1.0, 0.0]]]]),
    ),
    'with cutoff': dict(
        use_relative_positional_encoding_qk=False,
        use_relative_positional_encoding_v=False,
        inputs=jnp.asarray([[[[1.0, 0.0]]], [[[0.0, 1.0]]]]),
        basis=None,
        cutoff_value=jnp.asarray([1.0, 0.0, 1.0]),
        adj_idx=None,
        dst_idx=jnp.asarray([0, 0, 1]),
        src_idx=jnp.asarray([0, 1, 0]),
        expected_outputs=jnp.asarray([[[[1.0, 0.0]]], [[[1.0, 0.0]]]]),
    ),
    'with positional encoding qk': dict(
        use_relative_positional_encoding_qk=True,
        use_relative_positional_encoding_v=False,
        inputs=jnp.asarray([[[[1.0, 0.1]]], [[[0.1, 1.0]]]]),
        basis=jnp.asarray([
            [[[1.0, 1.0]]],
            [[[-999, -999]]],
            [[[1.0, 1.0]]],
        ]),
        cutoff_value=None,
        adj_idx=None,
        dst_idx=jnp.asarray([0, 0, 1]),
        src_idx=jnp.asarray([0, 1, 0]),
        expected_outputs=jnp.asarray([[[[1.0, 0.1]]], [[[1.0, 0.1]]]]),
    ),
    'with positional encoding v': dict(
        use_relative_positional_encoding_qk=False,
        use_relative_positional_encoding_v=True,
        inputs=jnp.asarray([[[[1.0, 0.0]]], [[[0.0, 1.0]]]]),
        basis=jnp.asarray([
            [[[1.0, 1.0]]],
            [[[0.5, 0.5]]],
        ]),
        cutoff_value=None,
        adj_idx=None,
        dst_idx=jnp.asarray([0, 1]),
        src_idx=jnp.asarray([1, 0]),
        expected_outputs=jnp.asarray([[[[0.0, 1.0]]], [[[0.5, 0.0]]]]),
    ),
})
def test_self_attention(
    use_fused_tensor: bool,
    use_relative_positional_encoding_qk: bool,
    use_relative_positional_encoding_v: bool,
    inputs: Float[Array, '...'],
    basis: Optional[Float[Array, '...']],
    cutoff_value: Optional[Float[Array, '...']],
    adj_idx: Optional[Integer[Array, '...']],
    dst_idx: Optional[Integer[Array, '...']],
    src_idx: Optional[Integer[Array, '...']],
    expected_outputs: Float[Array, '...'],
    num_heads: int = 1,
) -> None:
  attention = e3x.nn.SelfAttention(
      num_heads=num_heads,
      use_relative_positional_encoding_qk=use_relative_positional_encoding_qk,
      use_relative_positional_encoding_v=use_relative_positional_encoding_v,
      query_kernel_init=identity_initializer,
      key_kernel_init=identity_initializer,
      value_kernel_init=identity_initializer,
      output_kernel_init=identity_initializer,
      dense_kernel_init=identity_initializer,
      tensor_kernel_init=(
          fused_tensor_ones if use_fused_tensor else jax.nn.initializers.ones
      ),
      use_fused_tensor=use_fused_tensor,
  )
  outputs, _ = attention.init_with_output(
      jax.random.PRNGKey(0),
      inputs=inputs,
      basis=basis,
      cutoff_value=cutoff_value,
      adj_idx=adj_idx,
      dst_idx=dst_idx,
      src_idx=src_idx,
  )
  assert jnp.allclose(outputs, expected_outputs, atol=1e-5)


@pytest.mark.parametrize('use_fused_tensor', [True, False])
def test_self_attention_is_equivariant(
    use_fused_tensor: bool,
    num_inputs: int = 3,
    num_parity: int = 2,
    max_degree: int = 1,
    num_heads: int = 2,
    num_features: int = 8,
    num_basis: int = 4,
    dst_idx: Integer[Array, '...'] = jnp.asarray([0, 0, 1]),
    src_idx: Integer[Array, '...'] = jnp.asarray([0, 1, 0]),
) -> None:
  inputs_key, basis_key, rot_key, init_key = jax.random.split(
      jax.random.PRNGKey(0), num=4
  )
  # Random inputs and basis.
  inputs = jax.random.normal(
      inputs_key,
      (num_inputs, num_parity, (max_degree + 1) ** 2, num_features),
  )
  basis = jax.random.normal(
      basis_key, (len(dst_idx), 1, (max_degree + 1) ** 2, num_basis)
  )
  # Random rotation matrix.
  rot = e3x.so3.random_rotation(rot_key)
  wigner_d = e3x.so3.wigner_d(rot, max_degree=max_degree)
  # Rotated and reflected inputs/basis (for checking equivariance).
  inputs_rot = e3x.nn.rotate(inputs, wigner_d)
  inputs_ref = e3x.nn.reflect(inputs)
  basis_rot = e3x.nn.rotate(basis, wigner_d)
  basis_ref = e3x.nn.reflect(basis)
  # Initialize MultiHeadAttention layer.
  attention = e3x.nn.SelfAttention(
      num_heads=num_heads, use_fused_tensor=use_fused_tensor
  )
  # Apply module.
  outputs, params = attention.init_with_output(
      init_key,
      inputs=inputs,
      basis=basis,
      dst_idx=dst_idx,
      src_idx=src_idx,
  )
  outputs_rot = attention.apply(
      params,
      inputs=inputs_rot,
      basis=basis_rot,
      dst_idx=dst_idx,
      src_idx=src_idx,
  )
  outputs_ref = attention.apply(
      params,
      inputs=inputs_ref,
      basis=basis_ref,
      dst_idx=dst_idx,
      src_idx=src_idx,
  )
  # Check for equivariance.
  assert jnp.allclose(e3x.nn.rotate(outputs, wigner_d), outputs_rot, atol=1e-5)
  assert jnp.allclose(e3x.nn.reflect(outputs), outputs_ref, atol=1e-5)
