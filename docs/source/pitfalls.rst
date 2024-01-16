Pitfalls
========

.. _OrderingOfIrreps:

Ordering of irreps
------------------

Depending on the task you are trying to solve, the ordering of irreps may be
crucial for correctness. As explained :ref:`here <IrrepFeatures>`, features in
E3x consist of irreps of degrees :math:`\ell = 0, \dots, L` (where :math:`L` is
the maximum degree), with each irrep consisting of :math:`2\ell+1` numbers, all
stacked together in a single array. Following the common conventions for
`spherical harmonics <https://en.wikipedia.org/wiki/Spherical_harmonics>`_, we
refer to the :math:`2\ell+1` components of a single irrep as "orders" :math:`m`
and label them from :math:`m=-\ell, -\ell+1, \dots, \ell-1, \ell`. E3x supports two
modes for the ordering of irreps. Both go from irreps of small to large degrees
:math:`\ell`, but differ in the way the orders :math:`m` are arranged. The first
convention, referred to as "Cartesian order", goes from largest to smallest
:math:`\lvert m\rvert`, alternating between positive and negative values:

:math:`{}^{\ell=0}_{m=\pm 0},\; {}^{\ell=1}_{m=+1},\; {}^{\ell=1}_{m=-1},\; {}^{\ell=1}_{m=\pm 0},\; {}^{\ell=2}_{m=+2},\; {}^{\ell=2}_{m=-2},\; {}^{\ell=2}_{m=+1},\; {}^{\ell=2}_{m=-1},\; {}^{\ell=2}_{m=\pm 0},\; \dots`

The second convention goes from smallest to largest :math:`m`:

:math:`{}^{\ell=0}_{m=\pm 0},\; {}^{\ell=1}_{m=-1},\; {}^{\ell=1}_{m=\pm 0},\; {}^{\ell=1}_{m=+1},\; {}^{\ell=2}_{m=-2},\; {}^{\ell=2}_{m=-1},\; {}^{\ell=2}_{m=\pm 0},\; {}^{\ell=2}_{m=+1},\; {}^{\ell=2}_{m=+2},\; \dots`

By default, E3x uses Cartesian order. This convention may appear less intuitive
than the second one, but it has the advantage that irreps of degree
:math:`\ell=1` correspond to Cartesian (pseudo)vectors in the usual
:math:`x,y,z`-order (whereas with the second convention, the order would be
:math:`y,z,x`). Consequently, for predicting vectors (or using them as input
quantities), Cartesian order is more convenient. All operations where the order
of irreps matters take a boolean ``cartesian_order`` keyword, which can be set
to ``True`` or ``False`` to switch between the available conventions. However,
instead of passing this keyword to all operations, we recommend changing the
default behavior for *all* operations by calling
``e3x.Config.set_cartesian_order(<bool>)`` at the start of your script (see
:class:`Config <e3x.config.Config>` for details). This is less error prone, as
inadvertently mixing operations with different conventions would lead to
non-equivariant outputs.

The ordering of irreps may also be relevant for predicting other quantities. For
example, the
`multipole moments <https://en.wikipedia.org/wiki/Spherical_multipole_moments>`_
of molecules calculated with different *ab initio* codes often follow a specific
ordering convention. When trying to predict such quantities, it is necessary to
convert to the same convention (either by re-ordering the outputs of E3x or the
target values) before e.g. calculating the loss function. Otherwise, it may be
impossible to solve the prediction task, because the rotational behavior of the
individual irrep components would be inconsistent.

Spherical harmonics
-------------------

The `spherical harmonics <https://en.wikipedia.org/wiki/Spherical_harmonics>`_
are used extensively in E3x, e.g. to convert unit vectors to
:math:`\mathrm{SO3}`-features. In E3x,
`real spherical harmonics  <https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics>`_
are used instead of the complex formulation for efficiency reasons. Further,
E3x supports different normalization schemes for the spherical harmonics (see
:func:`spherical_harmonics <e3x.so3.irreps.spherical_harmonics>` for details).
If the spherical harmonics have different values than what you would expect,
please make sure that you are using the correct normalization scheme for your
desired application. Per default, the spherical harmonics in E3x use Racah's
normalization (not the more common orthonormal formulation!), which leads to
unit :math:`(2\ell+1)`-vectors for each irrep of degree :math:`\ell`.
