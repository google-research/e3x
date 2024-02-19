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

from typing import List, Tuple
import e3x
from ..testing import subtests
import jax
import jax.numpy as jnp
import jaxtyping
import pytest

Array = jaxtyping.Array
Float = jaxtyping.Float


@subtests({
    'rotation about x-axis by pi/2 (90 degrees)': dict(
        axis=jnp.array([1.0, 0.0, 0.0]),
        angle=jnp.pi / 2,
        expected=jnp.array(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]
        ),
    ),
    'rotation about x-axis by pi (180 degrees)': dict(
        axis=jnp.array([1.0, 0.0, 0.0]),
        angle=jnp.pi,
        expected=jnp.array(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
        ),
    ),
    'rotation about x-axis by 3pi/2 (270 degrees)': dict(
        axis=jnp.array([1.0, 0.0, 0.0]),
        angle=3 * jnp.pi / 2,
        expected=jnp.array(
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
        ),
    ),
    'rotation about y-axis by pi/2 (90 degrees)': dict(
        axis=jnp.array([0.0, 1.0, 0.0]),
        angle=jnp.pi / 2,
        expected=jnp.array(
            [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
        ),
    ),
    'rotation about y-axis by pi (180 degrees)': dict(
        axis=jnp.array([0.0, 1.0, 0.0]),
        angle=jnp.pi,
        expected=jnp.array(
            [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]
        ),
    ),
    'rotation about y-axis by 3pi/2 (270 degrees)': dict(
        axis=jnp.array([0.0, 1.0, 0.0]),
        angle=3 * jnp.pi / 2,
        expected=jnp.array(
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
        ),
    ),
    'rotation about z-axis by pi/2 (90 degrees)': dict(
        axis=jnp.array([0.0, 0.0, 1.0]),
        angle=jnp.pi / 2,
        expected=jnp.array(
            [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        ),
    ),
    'rotation about z-axis by pi (180 degrees)': dict(
        axis=jnp.array([0.0, 0.0, 1.0]),
        angle=jnp.pi,
        expected=jnp.array(
            [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]
        ),
    ),
    'rotation about z-axis by 3pi/2 (270 degrees)': dict(
        axis=jnp.array([0.0, 0.0, 1.0]),
        angle=3 * jnp.pi / 2,
        expected=jnp.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        ),
    ),
})
def test_rotation(
    axis: Float[Array, '... 3'], angle: float, expected: Float[Array, '... 3']
) -> None:
  xyz = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
  result = xyz @ e3x.so3.rotation(axis=axis, angle=angle)
  assert jnp.allclose(result, expected, atol=1e-5)


@pytest.fixture(name='random_axis_and_angle')
def fixture_random_axis_and_angle(
    num: int = 10,
) -> Tuple[Float[Array, 'num 3'], Float[Array, 'num']]:
  axis_key, angle_key = jax.random.split(jax.random.PRNGKey(0))
  axis = jax.random.normal(axis_key, shape=(num, 3))
  angle = jax.random.uniform(
      angle_key, shape=(num,), minval=-jnp.pi, maxval=jnp.pi
  )
  return axis, angle


def test_rotation_has_determinant_1(
    random_axis_and_angle: Tuple[Float[Array, '... 3'], Float[Array, '...']],
) -> None:
  axis, angle = random_axis_and_angle
  rot = e3x.so3.rotation(axis=axis, angle=angle)
  assert jnp.allclose(jnp.linalg.det(rot), 1.0, atol=1e-5)


def test_rotation_is_orthogonal(
    random_axis_and_angle: Tuple[Float[Array, '... 3'], Float[Array, '...']],
) -> None:
  axis, angle = random_axis_and_angle
  rot = e3x.so3.rotation(axis=axis, angle=angle)
  assert jnp.allclose(rot @ jnp.swapaxes(rot, -1, -2), jnp.eye(3), atol=1e-5)


def test_rotation_is_periodic_with_2pi(
    random_axis_and_angle: Tuple[Float[Array, '... 3'], Float[Array, '...']],
) -> None:
  axis, angle = random_axis_and_angle
  rot1 = e3x.so3.rotation(axis=axis, angle=angle)
  rot2 = e3x.so3.rotation(axis=axis, angle=angle + 2 * jnp.pi)
  assert jnp.allclose(rot1, rot2, atol=1e-5)


def test_reversing_axis_is_equivalent_to_reversing_angle(
    random_axis_and_angle: Tuple[Float[Array, '... 3'], Float[Array, '...']],
) -> None:
  axis, angle = random_axis_and_angle
  rot1 = e3x.so3.rotation(axis=-axis, angle=angle)
  rot2 = e3x.so3.rotation(axis=axis, angle=-angle)
  assert jnp.allclose(rot1, rot2, atol=1e-5)


@pytest.mark.parametrize(
    'u, v',
    [
        (jnp.asarray([1.0, 0.0, 0.0]), jnp.asarray([1.0, 0.0, 0.0])),
        (jnp.asarray([1.0, 0.0, 0.0]), jnp.asarray([0.0, 1.0, 0.0])),
        (jnp.asarray([2.0, 1.0, 0.0]), jnp.asarray([0.0, 0.0, 3.0])),
        (jnp.asarray([1.0, 0.0, 0.0]), jnp.asarray([-1.0, 0.0, 0.0])),
        (
            jnp.asarray([
                [1.0, 0.0, -1.0],
                [2.0, 1.0, -4.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, -1.0],
            ]),
            jnp.asarray([
                [0.0, 1.0, 2.0],
                [2.0, 3.0, 3.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]),
        ),
    ],
)
def test_alignment_rotation(
    u: Float[Array, '... 3'], v: Float[Array, '... 3']
) -> None:
  rot = e3x.so3.alignment_rotation(u, v)
  us = jnp.einsum('...a,...ab->...b', u, rot)  # Batch matrix multiplication.
  cos = jnp.sum(us * v, axis=-1) / (
      e3x.ops.norm(us, axis=-1) * e3x.ops.norm(v, axis=-1)
  )
  assert jnp.allclose(cos, 1.0, atol=1e-5)


@pytest.fixture(name='random_vectors')
def fixture_random_vectors(
    num: int = 10,
) -> List[Float[Array, 'num 3']]:
  vectors = jax.random.normal(jax.random.PRNGKey(0), shape=(2 * num, 3))
  return jnp.split(vectors, 2)


def test_alignment_rotation_has_determinant_1(
    random_vectors: Tuple[Float[Array, '... 3'], Float[Array, '... 3']],
) -> None:
  u, v = random_vectors
  rot = e3x.so3.alignment_rotation(u, v)
  assert jnp.allclose(jnp.linalg.det(rot), 1.0, atol=1e-5)


def test_alignment_rotation_is_orthogonal(
    random_vectors: Tuple[Float[Array, '... 3'], Float[Array, '... 3']],
) -> None:
  u, v = random_vectors
  rot = e3x.so3.alignment_rotation(u, v)
  assert jnp.allclose(rot @ jnp.swapaxes(rot, -1, -2), jnp.eye(3), atol=1e-5)


@subtests({
    'rotation about x-axis by pi/2 (90 degrees)': dict(
        a=jnp.pi / 2,
        b=0.0,
        c=0.0,
        expected=jnp.array(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]
        ),
    ),
    'rotation about x-axis by pi (180 degrees)': dict(
        a=jnp.pi,
        b=0.0,
        c=0.0,
        expected=jnp.array(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
        ),
    ),
    'rotation about x-axis by 3pi/2 (270 degrees)': dict(
        a=3 * jnp.pi / 2,
        b=0.0,
        c=0.0,
        expected=jnp.array(
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
        ),
    ),
    'rotation about y-axis by pi/2 (90 degrees)': dict(
        a=0.0,
        b=jnp.pi / 2,
        c=0.0,
        expected=jnp.array(
            [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
        ),
    ),
    'rotation about y-axis by pi (180 degrees)': dict(
        a=0.0,
        b=jnp.pi,
        c=0.0,
        expected=jnp.array(
            [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]
        ),
    ),
    'rotation about y-axis by 3pi/2 (270 degrees)': dict(
        a=0.0,
        b=3 * jnp.pi / 2,
        c=0.0,
        expected=jnp.array(
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
        ),
    ),
    'rotation about z-axis by pi/2 (90 degrees)': dict(
        a=0.0,
        b=0.0,
        c=jnp.pi / 2,
        expected=jnp.array(
            [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        ),
    ),
    'rotation about z-axis by pi (180 degrees)': dict(
        a=0.0,
        b=0.0,
        c=jnp.pi,
        expected=jnp.array(
            [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]
        ),
    ),
    'rotation about z-axis by 3pi/2 (270 degrees)': dict(
        a=0.0,
        b=0.0,
        c=3 * jnp.pi / 2,
        expected=jnp.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        ),
    ),
    'rotation about x- and y-axis by pi/2 (90 degrees)': dict(
        a=jnp.pi / 2,
        b=jnp.pi / 2,
        c=0.0,
        expected=jnp.array(
            [[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
        ),
    ),
    'rotation about x- and z-axis by pi/2 (90 degrees)': dict(
        a=jnp.pi / 2,
        b=0.0,
        c=jnp.pi / 2,
        expected=jnp.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
    ),
    'rotation about y- and z-axis by pi/2 (90 degrees)': dict(
        a=0.0,
        b=jnp.pi / 2,
        c=jnp.pi / 2,
        expected=jnp.array(
            [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        ),
    ),
    'rotation about x-, y- and z-axis by pi/2 (90 degrees)': dict(
        a=jnp.pi / 2,
        b=jnp.pi / 2,
        c=jnp.pi / 2,
        expected=jnp.array(
            [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
        ),
    ),
})
def test_rotation_euler(
    a: float, b: float, c: float, expected: Float[Array, '... 3']
) -> None:
  xyz = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
  result = xyz @ e3x.so3.rotation_euler(a, b, c)
  assert jnp.allclose(result, expected, atol=1e-5)


@pytest.fixture(name='random_euler_angles')
def fixture_random_euler_angles(
    num: int = 10,
) -> Tuple[Float[Array, 'num'], Float[Array, 'num'], Float[Array, 'num']]:
  euler_angles = jax.random.uniform(
      jax.random.PRNGKey(0), shape=(num, 3), minval=-jnp.pi, maxval=jnp.pi
  )
  return jnp.split(euler_angles, 3, axis=-1)  # pytype: disable=bad-return-type  # jnp-type


def test_rotation_euler_has_determinant_1(
    random_euler_angles: Tuple[
        Float[Array, '...'], Float[Array, '...'], Float[Array, '...']
    ],
) -> None:
  a, b, c = random_euler_angles
  rot = e3x.so3.rotation_euler(a=a, b=b, c=c)
  assert jnp.allclose(jnp.linalg.det(rot), 1.0, atol=1e-5)


def test_rotation_euler_is_orthogonal(
    random_euler_angles: Tuple[
        Float[Array, '...'], Float[Array, '...'], Float[Array, '...']
    ],
) -> None:
  a, b, c = random_euler_angles
  rot = e3x.so3.rotation_euler(a=a, b=b, c=c)
  assert jnp.allclose(rot @ jnp.swapaxes(rot, -1, -2), jnp.eye(3), atol=1e-5)


def test_rotation_euler_is_periodic_with_2pi(
    random_euler_angles: Tuple[
        Float[Array, '...'], Float[Array, '...'], Float[Array, '...']
    ],
) -> None:
  a, b, c = random_euler_angles
  rot1 = e3x.so3.rotation_euler(a=a, b=b, c=c)
  rot2 = e3x.so3.rotation_euler(a=a + 2 * jnp.pi, b=b, c=c)
  rot3 = e3x.so3.rotation_euler(a=a, b=b + 2 * jnp.pi, c=c)
  rot4 = e3x.so3.rotation_euler(a=a, b=b, c=c + 2 * jnp.pi)
  assert jnp.allclose(rot1, rot2, atol=1e-5)
  assert jnp.allclose(rot1, rot3, atol=1e-5)
  assert jnp.allclose(rot1, rot4, atol=1e-5)


def test_euler_angles_from_rotation(
    random_euler_angles: Tuple[
        Float[Array, '...'], Float[Array, '...'], Float[Array, '...']
    ],
) -> None:
  expected_a, expected_b, expected_c = random_euler_angles
  expected_rot = e3x.so3.rotation_euler(
      a=expected_a, b=expected_b, c=expected_c
  )
  a, b, c = e3x.so3.euler_angles_from_rotation(expected_rot)
  rot = e3x.so3.rotation_euler(a=a, b=b, c=c)
  # Instead of directly comparing the Euler angles, they are indirectly compared
  # via rotation matrices constructed from Euler angles. This is done because
  # multiple different values are equivalent, e.g. an angle x is equivalent to
  # an angle x + 2pi, which makes directly comparing the angles for equivalence
  # relatively complicated.
  assert jnp.allclose(rot, expected_rot, atol=1e-5)


@pytest.mark.parametrize(
    'expected_rot',
    [
        jnp.asarray([
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ]),
        jnp.asarray([
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]),
        jnp.asarray([
            [0.0, 0.0, 1.0],
            [jnp.cos(0.3), -jnp.sin(0.3), 0.0],
            [jnp.sin(0.3), jnp.cos(0.3), 0.0],
        ]),
        jnp.asarray([
            [0.0, 0.0, -1.0],
            [jnp.cos(0.7), -jnp.sin(0.7), 0.0],
            [-jnp.sin(0.7), -jnp.cos(0.7), 0.0],
        ]),
    ],
)
def test_euler_angles_from_rotation_when_degenerate(
    expected_rot: Float[Array, '3 3']
) -> None:
  a, b, c = e3x.so3.euler_angles_from_rotation(expected_rot)
  rot = e3x.so3.rotation_euler(a=a, b=b, c=c)
  assert jnp.allclose(rot, expected_rot, atol=1e-5)


def test_random_rotation() -> None:
  rot1, rot2 = e3x.so3.random_rotation(jax.random.PRNGKey(0), num=2)
  assert not jnp.allclose(rot1, rot2, atol=1e-5)


def test_random_rotation_with_no_perturbation_is_identity() -> None:
  rot = e3x.so3.random_rotation(jax.random.PRNGKey(0), perturbation=0.0)
  assert jnp.allclose(rot, jnp.eye(3), atol=1e-5)


@pytest.mark.parametrize('seed', [0, 7, 13, 42])
def test_random_rotation_has_determinant_1(seed: int) -> None:
  rot = e3x.so3.random_rotation(jax.random.PRNGKey(seed))
  assert jnp.allclose(jnp.linalg.det(rot), 1.0, atol=1e-5)


@pytest.mark.parametrize('seed', [0, 7, 13, 42])
def test_random_rotation_is_orthogonal(seed: int) -> None:
  rot = e3x.so3.random_rotation(jax.random.PRNGKey(seed))
  assert jnp.allclose(rot @ jnp.swapaxes(rot, -1, -2), jnp.eye(3), atol=1e-5)


@pytest.mark.parametrize('max_degree', [0, 1, 2, 3])
@pytest.mark.parametrize('cartesian_order', [True, False])
def test_wigner_d(
    max_degree: int, cartesian_order: bool, num: int = 10
) -> None:
  rot_key, r_key = jax.random.split(jax.random.PRNGKey(0), 2)
  rot = e3x.so3.random_rotation(rot_key, num=num)
  wigner_d = e3x.so3.wigner_d(
      rot, max_degree=max_degree, cartesian_order=cartesian_order
  )
  r = jax.random.normal(r_key, (num, 3))  # Random vectors.
  r_rot = jnp.einsum('...a,...ab->...b', r, rot)  # Rotate.
  ylm = e3x.so3.spherical_harmonics(  # From non-rotated vectors.
      r, max_degree=max_degree, cartesian_order=cartesian_order
  )
  ylm_rot = e3x.so3.spherical_harmonics(  # From rotated vectors.
      r_rot, max_degree=max_degree, cartesian_order=cartesian_order
  )
  # Rotate output from non-rotated vectors.
  ylm_wigner_d = jnp.einsum('...a,...ab->...b', ylm, wigner_d)
  assert jnp.allclose(ylm_wigner_d, ylm_rot, atol=1e-5)


def test_wigner_d_has_nan_safe_derivatives(
    max_degree: int = 2,
) -> None:
  finfo = jnp.finfo(jnp.float32)
  # Note: This is NOT a valid rotation matrix, but works for the test.
  x = jnp.asarray([
      [0.0, 0.0, 0.0],
      [finfo.tiny, 0.0, 0.0],
      [finfo.eps, 0.0, 0.0],
  ])
  func = lambda x: e3x.so3.wigner_d(x, max_degree=max_degree)
  for y in e3x.ops.evaluate_derivatives(func, x, max_order=4):
    assert jnp.all(jnp.isfinite(y))
