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

"""Utility functions for symbolic maths with irreps."""

import collections
from typing import Iterable, Tuple
import sympy as sp
from sympy.physics.wigner import clebsch_gordan as _complex_cg


def _spherical_harmonics(l: int, m: int) -> sp.Poly:
  """Real Cartesian spherical harmonics.

  Computes a symbolic expression for the spherical harmonics of degree l and
  order m (as polynomial) with sympy. Note: The spherical harmonics computed
  here use Racah's normalization (also known as Schmidt's semi-normalization):
              ∫ Ylm(r)·Yl'm'(r) dΩ = 4π/(2l+1)·δ(l,l')·δ(m,m')
  (the integral runs over the unit sphere Ω and δ is the delta function).

  Args:
    l: Degree of the spherical harmonic.
    m: Order of the spherical harmonic.

  Returns:
    A sympy.Poly object with a symbolic expression for the spherical harmonic
    of degree l and order m.
  """

  def a(m: int, x: sp.Symbol, y: sp.Symbol) -> sp.Symbol:
    a = sp.S(0)
    for p in range(m + 1):
      a += sp.binomial(m, p) * x**p * y ** (m - p) * sp.cos((m - p) * sp.pi / 2)
    return a

  def b(m: int, x: sp.Symbol, y: sp.Symbol) -> sp.Symbol:
    b = sp.S(0)
    for p in range(m + 1):
      b += sp.binomial(m, p) * x**p * y ** (m - p) * sp.sin((m - p) * sp.pi / 2)
    return b

  def pi(l: int, m: int, x: sp.Symbol, y: sp.Symbol, z: sp.Symbol) -> sp.Symbol:
    pi = sp.S(0)
    r2 = x**2 + y**2 + z**2
    for k in range((l - m) // 2 + 1):
      pi += (
          (-1) ** k
          * sp.S(2) ** (-l)
          * sp.binomial(l, k)
          * sp.binomial(2 * l - 2 * k, l)
          * sp.factorial(l - 2 * k)
          / sp.factorial(l - 2 * k - m)
          * z ** (l - 2 * k - m)
          * r2**k
      )
    return sp.sqrt(sp.factorial(l - m) / sp.factorial(l + m)) * pi

  x, y, z = sp.symbols('x y z')
  if m > 0:
    ylm = sp.sqrt(2) * pi(l, m, x, y, z) * a(m, x, y)
  elif m < 0:
    ylm = sp.sqrt(2) * pi(l, -m, x, y, z) * b(-m, x, y)
  else:
    ylm = pi(l, m, x, y, z)

  return sp.Poly(sp.simplify(ylm), x, y, z)


def _clebsch_gordan(
    l1: int, m1: int, l2: int, m2: int, l3: int, m3: int
) -> sp.Symbol:
  """Clebsch-Gordan coefficients for real spherical harmonics.

  Computes a symbolic expression for the Clebsch-Gordan coefficient
  <l1 m1|l2 m2|l3 m3> of the real spherical harmonics with sympy.

  Args:
    l1: Degree of first spherical harmonic.
    m1: Order of first spherical harmonic.
    l2: Degree of second spherical harmonic.
    m2: Order of second spherical harmonic.
    l3: Degree of third spherical harmonic.
    m3: Order of third spherical harmonic.

  Returns:
    A SymPy symbolic expression for the corresponding Clebsch-Gordan
    coefficient.
  """

  def a_b(m: int) -> Tuple[sp.Symbol, sp.Symbol]:
    if m < 0:
      if m % 2 == 0:
        return sp.I / sp.sqrt(2), -sp.I / sp.sqrt(2)
      else:
        return sp.I / sp.sqrt(2), sp.I / sp.sqrt(2)
    elif m > 0:
      if m % 2 == 0:
        return 1 / sp.sqrt(2), 1 / sp.sqrt(2)
      else:
        return -1 / sp.sqrt(2), 1 / sp.sqrt(2)
    else:  # m == 0
      tmp = 1 / sp.S(2)
      return tmp, tmp

  a1, b1 = a_b(m1)
  a2, b2 = a_b(m2)
  a3, b3 = a_b(m3)
  cg = sp.I ** (l1 + l2 - l3) * (
      a1 * a2 * a3.conjugate() * _complex_cg(l1, l2, l3, m1, m2, m3)
      + a1 * a2 * b3.conjugate() * _complex_cg(l1, l2, l3, m1, m2, -m3)
      + a1 * b2 * a3.conjugate() * _complex_cg(l1, l2, l3, m1, -m2, m3)
      + a1 * b2 * b3.conjugate() * _complex_cg(l1, l2, l3, m1, -m2, -m3)
      + b1 * a2 * a3.conjugate() * _complex_cg(l1, l2, l3, -m1, m2, m3)
      + b1 * a2 * b3.conjugate() * _complex_cg(l1, l2, l3, -m1, m2, -m3)
      + b1 * b2 * a3.conjugate() * _complex_cg(l1, l2, l3, -m1, -m2, m3)
      + b1 * b2 * b3.conjugate() * _complex_cg(l1, l2, l3, -m1, -m2, -m3)
  )
  return sp.simplify(cg)


def _rotate_xyz_polynomial(p: sp.Poly) -> sp.Poly:
  """Rotate a polynomial in x, y, and z coordinates.

  Given a polynomial in x, y, and z coordinates, this function computes the
  "rotated" polynomial, where the coordinates are rotated by the rotation matrix
  ┌               ┐
  │ r00  r01  r02 │
  │ r10  r11  r12 │
  │ r20  r21  r22 │
  └               ┘.

  Args:
    p: The sympy.Poly object representing the polynomial to rotate.

  Returns:
    A sympy.Poly object with a symbolic expression for the rotated polynomial.
  """
  # (Unrotated) x, y, z coordinates.
  x, y, z = sp.symbols('x y z')
  # Rotation matrix entries.
  r00, r01, r02, r10, r11, r12, r20, r21, r22 = sp.symbols(
      'r00 r01 r02 r10 r11 r12 r20 r21 r22'
  )
  # Rotated x, y, z coordinates.
  xs = r00 * x + r01 * y + r02 * z
  ys = r10 * x + r11 * y + r12 * z
  zs = r20 * x + r21 * y + r22 * z
  # Substitute x, y, z with rotated coordinates xs, ys, zs.
  x_, y_, z_ = sp.symbols('x_ y_ z_')  # (Necessary) temporary coordinates.
  p = p.subs([(x, x_), (y, y_), (z, z_)]).subs([(x_, xs), (y_, ys), (z_, zs)])
  # Return as a polynomial in the (unrotated) x, y, z coordinates.
  return sp.Poly(sp.simplify(p), x, y, z)


def _polynomial_dot_product(
    p1: sp.Poly,
    p2: sp.Poly,
    gens: Iterable[sp.Symbol] = sp.symbols(
        'r00 r01 r02 r10 r11 r12 r20 r21 r22'
    ),
) -> sp.Poly:
  """Polynomial dot product.

  Args:
    p1: First polynomial.
    p2: Second polynomial.
    gens: Generators of the returned polynomial (for computing entries of
      Wigner-D matrices, these should be the entries of the rotation matrix).

  Returns:
    A sympy.Poly object with a symbolic expression for the dot product expressed
    as a polynomial of the given generator expressions.
  """

  def trinomial_coefficient(a, b, c):
    return sp.factorial(a + b + c) / (
        sp.factorial(a) * sp.factorial(b) * sp.factorial(c)
    )

  # Initialize result to 0.
  result = sp.S(0)
  # Early out if degrees do not match.
  degree = p1.total_degree()
  if degree != p2.total_degree():
    return result
  # Calculate dot product.
  p1 = p1.as_dict()
  p2 = collections.defaultdict(lambda: sp.S(0), p2.as_dict())
  for key in p1:  # key = (a, b, c)
    result += p1[key] * p2[key] / trinomial_coefficient(*key)
  result *= sp.S(2) ** degree / sp.binomial(2 * degree, degree)
  # Return result as polynomial.
  return sp.Poly(sp.simplify(result), *gens)
