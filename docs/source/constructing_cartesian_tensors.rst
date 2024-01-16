Constructing Cartesian tensors
==============================

Background
^^^^^^^^^^

In physics and engineering,
`Cartesian tensors <https://en.wikipedia.org/wiki/Cartesian_tensor>`_ play an
important role, often in a "stimulus-response" way, when some quantities in a
system of interest have a "directional dependence". For example, the
`moment of intertia tensor <https://en.wikipedia.org/wiki/Moment_of_inertia>`_
:math:`\mathbf{I} \in \mathbb{R}^{3\times3}` of a physical object relates its
angular velocity :math:`\boldsymbol{\omega} \in \mathbb{R}^{3}` to its angular
momentum :math:`\mathbf{J} \in \mathbb{R}^{3}` and rotational kinetic energy
:math:`K \in \mathbb{R}`:

.. math::
  \mathbf{J} &= \mathbf{I} \cdot \boldsymbol{\omega}\\
  K &= \frac{1}{2} \boldsymbol{\omega} \cdot \mathbf{I} \cdot \boldsymbol{\omega}\,.

The moment of inertia tensor :math:`\mathbf{I}` is a tensor of degree (often
also called "rank" or "order") :math:`2`. In general, a Cartesian tensor of
degree :math:`\ell` is a :math:`3\times3\times\dots\times3`
(repeated :math:`\ell` times) array of numbers (a tensor of degree
:math:`\ell=0` is just a single number) and can take :math:`m \leq \ell` vectors
and return a new tensor of degree :math:`\ell - m`.


Formal definition of Cartesian tensors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`Cartesian tensors <https://en.wikipedia.org/wiki/Cartesian_tensor>`_ are
tensors on vector spaces :math:`V` with a given scalar product -- in our case,
on :math:`V=\mathbb{R}^3` with the usual scalar product. The tensors of degree
:math:`\ell` can be defined as the multilinear functions
:math:`V\times V \times ... \times V\rightarrow \mathbb{R}`, where we have
:math:`\ell` variables in :math:`V`. We also allow :math:`\ell=0` variables, in
that case, the function of no variable/tensor of degree :math:`0` is just a
single number. (Since we have a fixed scalar product, we do not have to
distinguish between covariant and contravariant indices, so instead of the
":math:`(p,q)`-tensors" that are used in settings without a specified scalar
product, Cartesian tensors only have one degree :math:`\ell`, corresponding to
:math:`p+q=\ell` in the other settings.) In our case :math:`V=\mathbb{R}^3` and
the
`multilinear functions <https://en.wikipedia.org/wiki/Multilinear_map#Coordinate_representation>`_
of :math:`\ell` variables can be given by an :math:`\ell`-dimensional array of
:math:`3^\ell` numbers. There is a natural operation of rotations/reflections
:math:`g\in \mathrm{O}(3)` on these multilinear functions :math:`f`, its result
:math:`g f` is uniquely defined by the property
:math:`(g f)(g v_1, ..., g v_l) := f(v_1,...,v_l)`. Therefore we can also say
what a covariant map from e.g. point configurations to tensors is, which gives
us the notion of covariant tensor-valued features.

We already said that tensors of degree :math:`0` are just numbers. Since we have
a fixed scalar product, we can also identify vectors :math:`v \in V` with their
linear functions :math:`L(w) := \langle w, v \rangle` and vice versa, so
Cartesian tensors of degree :math:`1` can be identified with vectors, and under
this identification, also the operations of :math:`\mathrm{O}(3)` match.
Similarly, a matrix :math:`A` or a linear map :math:`A:V\rightarrow V` can be
identified with the tensor of degree :math:`2` given by
:math:`T(v,w) := \langle v, A w \rangle`.

Under this identification, symmetric matrices correspond to tensors of degree
:math:`2` that satisfy :math:`T(v,w)=T(w,v)`. In general, a tensor is called
`symmetric <https://en.wikipedia.org/wiki/Symmetric_tensor>`_ if one can change
any two variables without affecting the value. As is known from symmetric
matrices and quadratic forms, a symmetric tensor :math:`T` of degree :math:`2`
is already determined by its values :math:`T(v,v)`.

Traceless symmetric tensors
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The notion of the trace of a matrix :math:`\mathrm{Tr}(a_{ij}) := \sum_i a_{ii}`
can also be generalized to tensors: For any tensor
:math:`t_{{i_1}{i_2}...{i_l}}` of degree :math:`\ell` and any two indices
:math:`1\leq n < m \leq \ell`, we can sum over all :math:`\ell`-tuples of
indices in which :math:`i_n = i_m`, this gives a tensor of degree
:math:`\ell-2`. A tensor is called traceless if all of these tensors of degree
:math:`\ell-2` are zero.

It follows directly from the definitions that any :math:`g\in \mathrm{O}(3)`
maps symmetric tensors again to symmetric tensors, and one can show that it also
maps traceless tensors to traceless tensors, so the subspace of symmetric
traceless tensors is also a representation of any :math:`\mathrm{O}(3)`. One can
show that its dimension is :math:`2\ell+1` and that this is in fact the
irreducible representation of degree :math:`\ell` and parity :math:`(-1)^\ell`.

Therefore, the :ref:`irreps <Irreps>` used as features in E3x are mathematically
equivalent to symmetric traceless (pseudo)tensors of the corresponding degrees,
meaning they transform in the same manner under rotations (and reflections),
while being much more compact (requiring only :math:`2\ell+1` instead of
:math:`3^\ell` numbers to store the same information). For this reason, all
operations in E3x work with the :math:`(2\ell+1)`-sized representation of
(pseudo)tensors. When predicting a tensorial quantity with a neural network,
however, it is often convenient to convert the output features back to a
:math:`3\times3\times\dots\times3` array of numbers. E3x contains convenience
functions that allow an easy back-and-forth transformation between the different
representations. Let's assume the output features predicted by our neural
network are :math:`\mathbf{x} \in \mathbb{R}^{2\times (L+1)^2\times F}` and
we are interested in a specific tensor
:math:`\mathbf{a} = \mathbf{x}_0^{(3_-)}` (see :ref:`here <FeatureSlicing>` for
a reminder on the notation used here and feature slicing).

.. jupyter-execute::
  :hide-code:

  import jax
  import jax.numpy as jnp
  import e3x
  jnp.set_printoptions(precision=3, suppress=True)

We can extract :math:`\mathbf{a}` with the following code snippet:

.. jupyter-execute::

  # Draw random features (in a real application,
  # these would be the output of a neural network).
  x = jax.random.normal(jax.random.PRNGKey(0), (2, 16, 4))

  p = 1  # parity (0=even, 1=odd)
  l = 3  # degree
  f = 0  # feature index
  irrep = x[p, l**2:(l+1)**2, f]  # a in irrep form
  print(irrep)

To convert :math:`\mathbf{a}` to the corresponding traceless symmetric
:math:`3 \times 3 \times 3` tensor, we can use

.. jupyter-execute::

  tensor = e3x.so3.irreps_to_tensor(irrep, degree=l)  # a in tensor form
  print(tensor)

To go back to the original irrep representation, we can use

.. jupyter-execute::

  irrep_from_tensor = e3x.so3.tensor_to_irreps(tensor, degree=l)
  print(irrep_from_tensor)

For some applications (e.g., predicting
`multipole moments <https://en.wikipedia.org/wiki/Multipole_expansion>`_), this
simple conversion may already be sufficient.

General tensors of degree :math:`2`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

However, we may be interested in general tensors, not only traceless symmetric
ones. For this, it is helpful to think of the desired tensor of degree
:math:`\ell` as a repeated tensor product of :math:`\ell` (three-dimensional)
irreps :math:`\mathbb{1}`. For example, for a general tensor of degree
:math:`2`, we have (see also :ref:`coupling of irreps <CouplingIrreps>`)

.. math::
  \mathbb{1} \otimes \mathbb{1} = \mathbb{0} \oplus \mathbb{1} \oplus \mathbb{2}\,,

meaning that a general :math:`3 \times 3` tensor can be constructed from irreps
:math:`\mathbb{0}`, :math:`\mathbb{1}`, and :math:`\mathbb{2}`.
First, we collect these irreps from the features. We have to make sure that the
parity of the irreps is even/odd when the degree of the (proper) tensor we want
to predict is even/odd, otherwise, we would not construct a tensor, but a
pseudotensor. For degree :math:`2`, this means we need irreps with even parity.

.. jupyter-execute::

  tensor_components = []
  for l in (0, 1, 2):
    tensor_components.append(x[0, l**2:(l+1)**2, f])
    print(f'l={l}\n', tensor_components[-1], '\n')

So far so good, but we'd like to have a :math:`3 \times 3` output. To convert
the irreps to the correct shape, we can use
`Clebsch-Gordan coefficients <https://en.wikipedia.org/wiki/Clebsch%E2%80%93Gordan_coefficients>`_.

.. jupyter-execute::

  cg = e3x.so3.clebsch_gordan(1, 1, 2)
  for l in range(3):
    tensor_components[l] = jnp.einsum('...n,lmn->lm',
        tensor_components[l],
        cg[1:4, 1:4, l**2:(l+1)**2]
    )
    print(f'l={l}\n', tensor_components[l], '\n')

In this form, we can clearly see that the irreps :math:`\mathbb{0}`,
:math:`\mathbb{1}`, and :math:`\mathbb{2}` contribute the trace,
antisymmetric component, and symmetric traceless component. The final
:math:`3 \times 3` tensor is simply the sum of all these components.

.. jupyter-execute::

  tensor = sum(tensor_components)
  print(tensor)

Depending on what is known about the quantity of interest, it may be even better
to remove certain irreps. For example, if we know that the :math:`3 \times 3`
tensor we want to predict is symmetric, we could simply remove the contribution
from :math:`\mathbb{1}` to always receive symmetric tensors. See also
`here <examples/moment_of_inertia.html>`_ for a full worked out example of
predicting tensorial quantities using this recipe.

General tensors of higher degrees
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The general method described above also works for tensors of higher degrees, but
more irreps are necessary and the conversion with Clebsch-Gordan coefficients is
slightly more complicated. For example, for a degree :math:`3` tensor, we have:

.. math::
  \mathbb{1} \otimes \mathbb{1} \otimes \mathbb{1}
    &= (\mathbb{1} \otimes \mathbb{1}) \otimes \mathbb{1} \\
    &= (\mathbb{0} \oplus \mathbb{1} \oplus \mathbb{2}) \otimes \mathbb{1} \\
    &= (\mathbb{0} \otimes \mathbb{1}) \oplus (\mathbb{1} \otimes \mathbb{1}) \oplus (\mathbb{2} \otimes \mathbb{1}) \\
    &= (\mathbb{1}) \oplus (\mathbb{0} \oplus \mathbb{1} \oplus \mathbb{2}) \oplus (\mathbb{1} \oplus \mathbb{2} \oplus \mathbb{3}) \\
    &= \mathbb{0} \oplus \mathbb{1} \oplus \mathbb{1} \oplus \mathbb{1} \oplus \mathbb{2} \oplus \mathbb{2} \oplus \mathbb{3}
This means that in total, we need the following seven irreps to represent an
arbitrary :math:`3\times3\times3` tensor: :math:`1 \times \mathbb{0}`,
:math:`3 \times \mathbb{1}`, :math:`2 \times \mathbb{2}`, and
:math:`1 \times \mathbb{3}` (as a sanity check, it can be helpful to confirm
that this corresponds to :math:`3\times3\times3=27` individual numbers). To
convert the irreps to the desired :math:`3\times3\times3` shape, it is necessary
to remember which "coupling path" (see above) led to it (differences between
coupling paths are underlined). We have:

.. list-table:: irreps of :math:`3\times3\times3` tensor
   :header-rows: 1
   :align: center
   :widths: auto

   * - #
     - irrep
     - coupling path
   * - 1
     - :math:`\mathbb{0}`
     - :math:`((\mathbb{1}\otimes\mathbb{1})\rightarrow\underline{\mathbb{1}})\otimes\mathbb{1}\rightarrow\underline{\mathbb{0}}`
   * - 2
     - :math:`\mathbb{1}`
     - :math:`((\mathbb{1}\otimes\mathbb{1})\rightarrow\underline{\mathbb{0}})\otimes\mathbb{1}\rightarrow\underline{\mathbb{1}}`
   * - 3
     - :math:`\mathbb{1}`
     - :math:`((\mathbb{1}\otimes\mathbb{1})\rightarrow\underline{\mathbb{1}})\otimes\mathbb{1}\rightarrow\underline{\mathbb{1}}`
   * - 4
     - :math:`\mathbb{1}`
     - :math:`((\mathbb{1}\otimes\mathbb{1})\rightarrow\underline{\mathbb{2}})\otimes\mathbb{1}\rightarrow\underline{\mathbb{1}}`
   * - 5
     - :math:`\mathbb{2}`
     - :math:`((\mathbb{1}\otimes\mathbb{1})\rightarrow\underline{\mathbb{1}})\otimes\mathbb{1}\rightarrow\underline{\mathbb{2}}`
   * - 6
     - :math:`\mathbb{2}`
     - :math:`((\mathbb{1}\otimes\mathbb{1})\rightarrow\underline{\mathbb{2}})\otimes\mathbb{1}\rightarrow\underline{\mathbb{2}}`
   * - 7
     - :math:`\mathbb{3}`
     - :math:`((\mathbb{1}\otimes\mathbb{1})\rightarrow\underline{\mathbb{2}})\otimes\mathbb{1}\rightarrow\underline{\mathbb{3}}`

Let's first collect the seven irreps we need (different feature indices are used
for irreps with the same degree, because they should be independent). Since the
tensor we want to predict has degree :math:`3` (odd), we also want irreps with
odd parity (irreps with even parity would give us a pseudotensor).

.. jupyter-execute::

  tensor_components = []
  for l, f in [(0, 0), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (3, 0)]:
    tensor_components.append(x[1, l**2:(l+1)**2, f])
    print(f'l={l} (feature channel {f})\n', tensor_components[-1], '\n')

Now, the irreps can be converted to shape :math:`3\times3\times3` based on their
coupling path (see table above):

.. jupyter-execute::

  coupling_paths = [
    (1, 0), #1
    (0, 1), #2
    (1, 1), #3
    (2, 1), #4
    (1, 2), #5
    (2, 2), #6
    (2, 3), #7
  ]

  cg = e3x.so3.clebsch_gordan(2, 1, 3)
  for i, (l1, l2) in enumerate(coupling_paths):
    tensor_components[i] = jnp.einsum('...p,lmn,nop->...lmo',
        tensor_components[i],
        cg[1:4, 1:4, l1**2:(l1+1)**2],
        cg[l1**2:(l1+1)**2, 1:4, l2**2:(l2+1)**2]
    )
    print(f'irrep #{i} (l={l2})\n', tensor_components[i], '\n')

Again, the different irreps correspond to different contributions to the final
:math:`3\times3\times3`, which is obtained by summing over all contributions.

.. jupyter-execute::

  tensor = sum(tensor_components)
  print(tensor)

Tensors with even higher degrees can be constructed analogously.