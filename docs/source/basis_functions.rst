Basis functions
===============

A common pattern used in :math:`\mathrm{E}(3)`-equivariant architectures is the
"featurization" of Euclidean vectors :math:`\vec{v}`. This is typically done by
expanding the vectors in radial-spherical basis functions. For example, the
:class:`MessagePass <e3x.nn.modules.MessagePass>` expects an expansion of all
relative displacement vectors between source and target nodes in in basis
functions as input. In the `tetracubes example <examples/tetracubes.html>`_,
basis functions are used to "featurize" the positions of the four cubes making
up a tetracube shape.

We recommended to use the convenience wrapper function
:func:`basis <e3x.nn.wrappers.basis>` to construct basis function (see its
documentation for more details). This wrapper function can be combined with a
multitude of different radial functions implemented in E3x, e.g.
:func:`exponential_bernstein <e3x.nn.functions.bernstein.exponential_bernstein>`
or :func:`sinc <e3x.nn.functions.trigonometric.sinc>`, as well as cutoff or
damping functions such as
:func:`smooth_cutoff <e3x.nn.functions.cutoff.smooth_cutoff>` or
:func:`smooth_damping <e3x.nn.functions.cutoff.smooth_cutoff>`. For a hands-on
example of how :func:`basis <e3x.nn.wrappers.basis>` can be used in a real
architecture, please refer to the
`MD17 ethanol example <examples/md17_ethanol.html>`_.
