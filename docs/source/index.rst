E3x documentation
=================

E3x is a `JAX <https://jax.readthedocs.io/>`_ library for constructing
efficient :math:`\mathrm{E}(3)`-equivariant deep learning architectures built on
top of `Flax <https://flax.readthedocs.io/>`_.

To learn how E3x works, what :math:`\mathrm{E}(3)`-equivariance means, and for
what it is useful, please have a look at the :ref:`Overview`, which also
introduces notation used throughout the documentation. To learn how to use E3x,
please refer to the :ref:`Examples`, which show how to solve common tasks with
simple toy problems. If you encounter any difficulties or problems when working
with E3x, make sure to check the :ref:`Pitfalls` for common mistakes and sources
of error. More details on the mathematical theory behind E3x can be found in
`this paper <https://arxiv.org/abs/2401.07595>`_.

E3x is available on `github <https://github.com/google-research/e3x>`_. To
install E3x, simply run

::

  python -m pip install --upgrade e3x


.. _Quickstart:

.. toctree::
   :maxdepth: 1
   :caption: Quickstart

   overview
   pitfalls

.. _Examples:

.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples/tetracubes
   examples/moment_of_inertia
   examples/md17_ethanol

.. _Useful recipes and tricks:

.. toctree::
   :maxdepth: 1
   :caption: Useful recipes and tricks

   changing_defaults
   neighbor_lists
   basis_functions
   constructing_cartesian_tensors


.. _API reference:

.. toctree::
   :maxdepth: 3
   :caption: API reference

   api
