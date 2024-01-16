Changing the default behavior of E3x
====================================

Functions and modules in E3x often assume specific conventions, which, in some
cases, may be even necessary for the correct interaction with other components.
Examples include the :ref:`ordering of irreps <OrderingOfIrreps>`, whether
higher-level modules such as :class:`TensorDense <e3x.nn.modules.TensorDense>`
or :class:`MessagePass <e3x.nn.modules.MessagePass>` use the
:class:`Tensor <e3x.nn.modules.Tensor>` or the
:class:`FusedTensor <e3x.nn.modules.FusedTensor>` module to
:ref:`couple irreps <CouplingIrreps>`, or the normalization used when computing
:func:`spherical_harmonics <e3x.so3.irreps.spherical_harmonics>`. While the
behavior of all functions and modules can be tuned individually by providing
corresponding keyword arguments, it may be convenient to change the default
behavior globally (e.g. to avoid subtle bugs that may appear when the
:ref:`ordering of irreps <OrderingOfIrreps>` is inconsistent across different
operations). Default behaviors can be changed with the
:class:`Config <e3x.config.Config>` class (see its documentation for a full
overview of changeable behaviors). For example, to change the default
normalization of
:func:`spherical_harmonics <e3x.so3.irreps.spherical_harmonics>` from
``'racah'`` to ``'orthonormal'`` you could simply call

.. jupyter-execute::

  import e3x
  e3x.Config.set_normalization('orthonormal')

at the start of your script.