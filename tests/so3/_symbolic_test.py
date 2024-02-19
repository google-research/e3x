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

import e3x
import pytest
import sympy as sp
from sympy.abc import x, y, z


@pytest.mark.parametrize(
    'l, m, expected',
    [
        (0, 0, sp.poly('1', x, y, z)),
        (1, -1, sp.poly('y', x, y, z)),
        (1, 0, sp.poly('z', x, y, z)),
        (1, 1, sp.poly('x', x, y, z)),
        (2, -2, sp.poly('sqrt(3) * x * y', x, y, z)),
        (2, -1, sp.poly('sqrt(3) * y * z', x, y, z)),
        (2, 0, sp.poly('z**2 - x**2 / 2 - y**2 / 2', x, y, z)),
        (2, 1, sp.poly('sqrt(3) * x * z', x, y, z)),
        (2, 2, sp.poly('sqrt(3) / 2 * (x**2 - y**2)', x, y, z)),
        (
            4,
            0,
            sp.poly(
                '3/8 * x**4 + 3 / 4 * x**2 * y**2 - 3 * x**2 * z**2 + 3 / 8 *'
                ' y**4 - 3 * y**2 * z**2 + z**4',
                x,
                y,
                z,
            ),
        ),
    ],
)
def test__spherical_harmonics(l: int, m: int, expected: sp.Poly) -> None:
  assert e3x.so3._symbolic._spherical_harmonics(l, m) == expected


@pytest.mark.parametrize(
    'l1, m1, l2, m2, l3, m3, expected',
    [
        (0, 0, 0, 0, 0, 0, sp.S(1)),
        (1, 1, 1, 1, 0, 0, sp.sqrt(3) / 3),
        (1, 1, 1, 1, 1, 1, sp.S(0)),
        (1, -1, 1, 1, 1, 0, -sp.sqrt(2) / 2),
        (2, 0, 2, 0, 0, 0, sp.sqrt(5) / 5),
    ],
)
def test__clebsch_gordan(
    l1: int, m1: int, l2: int, m2: int, l3: int, m3: int, expected: sp.Symbol
) -> None:
  assert e3x.so3._symbolic._clebsch_gordan(l1, m1, l2, m2, l3, m3) == expected


@pytest.mark.parametrize(
    'polynomial, expected',
    [
        (sp.poly('1', x, y, z), sp.poly('1', x, y, z)),
        (
            sp.poly('x', x, y, z),
            sp.poly('r00 * x + r01 * y + r02 * z', x, y, z),
        ),
        (
            sp.poly('y', x, y, z),
            sp.poly('r10 * x + r11 * y + r12 * z', x, y, z),
        ),
        (
            sp.poly('z', x, y, z),
            sp.poly('r20 * x + r21 * y + r22 * z', x, y, z),
        ),
        (
            sp.poly('x**2', x, y, z),
            sp.poly(
                '(r00 * x + r01 * y + r02 * z) * (r00 * x + r01 * y + r02 * z)',
                x,
                y,
                z,
            ),
        ),
        (
            sp.poly('y**2', x, y, z),
            sp.poly(
                '(r10 * x + r11 * y + r12 * z) * (r10 * x + r11 * y + r12 * z)',
                x,
                y,
                z,
            ),
        ),
        (
            sp.poly('z**2', x, y, z),
            sp.poly(
                '(r20 * x + r21 * y + r22 * z) * (r20 * x + r21 * y + r22 * z)',
                x,
                y,
                z,
            ),
        ),
        (
            sp.poly('x*y', x, y, z),
            sp.poly(
                '(r00 * x + r01 * y + r02 * z) * (r10 * x + r11 * y + r12 * z)',
                x,
                y,
                z,
            ),
        ),
        (
            sp.poly('x*z', x, y, z),
            sp.poly(
                '(r00 * x + r01 * y + r02 * z) * (r20 * x + r21 * y + r22 * z)',
                x,
                y,
                z,
            ),
        ),
        (
            sp.poly('y*z', x, y, z),
            sp.poly(
                '(r10 * x + r11 * y + r12 * z) * (r20 * x + r21 * y + r22 * z)',
                x,
                y,
                z,
            ),
        ),
    ],
)
def test__rotate_xyz_polynomial(polynomial: sp.Poly, expected: sp.Poly) -> None:
  assert e3x.so3._symbolic._rotate_xyz_polynomial(polynomial) == expected


@pytest.mark.parametrize(
    'p1, p2, expected',
    [
        (
            sp.poly('1', x, y, z),
            sp.poly('1', x, y, z),
            sp.poly('1', sp.symbols('r00 r01 r02 r10 r11 r12 r20 r21 r22')),
        ),
        (
            sp.poly('x', x, y, z),
            sp.poly('x', x, y, z),
            sp.poly('1', sp.symbols('r00 r01 r02 r10 r11 r12 r20 r21 r22')),
        ),
        (
            sp.poly('x', x, y, z),
            sp.poly('y', x, y, z),
            sp.poly('0', sp.symbols('r00 r01 r02 r10 r11 r12 r20 r21 r22')),
        ),
        (
            sp.poly('x', x, y, z),
            sp.poly('r00 * x + r01 * y + r02 * z', x, y, z),
            sp.poly('r00', sp.symbols('r00 r01 r02 r10 r11 r12 r20 r21 r22')),
        ),
        (
            sp.poly('sqrt(3)*x*y', x, y, z),
            sp.poly('sqrt(3)*x*y', x, y, z),
            sp.poly('1', sp.symbols('r00 r01 r02 r10 r11 r12 r20 r21 r22')),
        ),
        (
            sp.poly('sqrt(3)*x*y', x, y, z),
            sp.poly('sqrt(2)*x*y', x, y, z),
            sp.poly(
                'sqrt(2/3)', sp.symbols('r00 r01 r02 r10 r11 r12 r20 r21 r22')
            ),
        ),
    ],
)
def test__polynomial_dot_product(
    p1: sp.Poly, p2: sp.Poly, expected: sp.Poly
) -> None:
  assert e3x.so3._symbolic._polynomial_dot_product(p1, p2) == expected
