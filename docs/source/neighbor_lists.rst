Neighbor/index lists
====================

When working with data that can be represented as point clouds (e.g. molecules,
polygen meshes), it is often necessary to define neighbor/index lists, which
specify which points interact with each other (e.g. in a
:class:`MessagePass <e3x.nn.modules.MessagePass>` layer, the index lists
determine which nodes pass messages between each other). E3x supports two kinds
of index lists, which we call "sparse" and "dense", for use with
":ref:`indexed operations <IndexedOps>`". Indexed operations can be thought of
as operating according to a graph, with index lists specifying edges/node
connectivity.

For example, consider the (undirected) graph::

  0───1───3
       ╲ ╱
        2


With sparse index lists, this connectivity could be encoded with two arrays, the
"destination indices" (``dst_idx``) and the "source indices" (``src_idx``) as::

  dst_idx = [0, 1, 1, 1, 2, 2, 3, 3]
  src_idx = [1, 0, 2, 3, 1, 3, 1, 2]

You can read this as: "The node with index ``src_idx[i]`` connects to the node
with index ``dst_idx[i]``." Note that in this example, each "edge" is specified
twice, once with node :math:`a` as source and node :math:`b` as destination, and
once with the roles reversed (:math:`b` is the destination, :math:`a` is the
source). This is the typical setup for message-passing, because we want both
nodes to "pass messages" to each other. A directed graph (with "unidirectional
message-passing") can easily be defined by only specifying one edge-direction,
see :ref:`below <DirectedEdgesAndLoops>`.

Sparse index lists can always be padded with node indices that do not appear in
the graph (any number larger than the largest valid index may be used for
padding) without changing the results, for example::

  dst_idx = [0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4]
  src_idx = [1, 0, 2, 3, 1, 3, 1, 2, 4, 4, 4, 4]
                                     | padding |

Adding padding is often necessary to avoid frequent recompilation when using
:func:`jax.jit <jax.jit>`. With a dense index list, the same connectivity would
be encoded with "adjacency indices" as::

  adj_idx = [[1, 4, 4],
             [0, 2, 3],
             [1, 3, 4],
             [1, 2, 4]]

You can read this as: "The nodes at indices ``adj_idx[i, :]`` connect to the
node with index ``i``."

The use of padding values (``4`` in this example) with dense index lists is
necessary here, because nodes have different numbers of neighbors (e.g., node 0
has only one neighbor, but node 1 has three neighbors). Adding additional
padding does not change the results.

Depending on the use case, either sparse or dense neighborlists can be more
"natural"/efficient, so both are supported. There are also convenience functions
for converting from one format to the other (introducing padding if necessary),
see :func:`sparse_to_dense_indices <e3x.ops.indexed.sparse_to_dense_indices>`
and :func:`dense_to_sparse_indices <e3x.ops.indexed.dense_to_sparse_indices>`.

.. _DirectedEdgesAndLoops:

It is also possible to specify "directed edges" or "loops", for example::

  0◀──1──▶2◀─╮
          ╰──╯

With sparse index lists::

  dst_idx = [0, 2, 2]
  src_idx = [1, 1, 2]

With dense index lists (note the use of ``3`` as padding)::

  adj_idx = [[1, 3],
             [3, 3],
             [1, 2]]


Usage example
^^^^^^^^^^^^^

.. jupyter-execute::
  :hide-code:

  import jax
  import jax.numpy as jnp
  import e3x
  jnp.set_printoptions(precision=3, suppress=True)

Recall the first example from above with the graph::

  0───1───3
       ╲ ╱
        2

Let's imagine we have four points embedded in three-dimensional space and we
want to calculate distances between pairs of points according to the
graph connectivity specified above.

.. jupyter-execute::

  # Positions of the four points specified as x, y, z coordinates.
  positions = jnp.array([
    [-1.0, 0.0, 0.0],  # point/node 0
    [ 0.0, 0.0, 0.0],  # point/node 1
    [ 1.0, 0.0, 1.0],  # point/node 2
    [ 0.5, 0.5, 0.0],  # point/node 3
  ])

  # Sparse index list.
  dst_idx = jnp.array([0, 1, 1, 1, 2, 2, 3, 3])
  src_idx = jnp.array([1, 0, 2, 3, 1, 3, 1, 2])

  # Dense index list.
  adj_idx = jnp.array([[1, 4, 4], [0, 2, 3], [1, 3, 4], [1, 2, 4]])

Let's start with the sparse index list.

To compute the distances, we need to gather the positions from both "sources"
(using :func:`gather_src <e3x.ops.indexed.gather_src>`) and "destinations"
(using :func:`gather_dst <e3x.ops.indexed.gather_dst>`), calculate their
difference to get the "displacement vectors" between the points, and finally
calculate the norm of the displacements.

.. jupyter-execute::

  dst_positions = e3x.ops.gather_dst(positions, dst_idx=dst_idx)
  src_positions = e3x.ops.gather_src(positions, src_idx=src_idx)
  displacements = dst_positions - src_positions
  distances = e3x.ops.norm(displacements, axis=-1)

  print('dst_positions\n', dst_positions, '\n')
  print('src_positions\n', src_positions, '\n')
  print('displacements\n', displacements, '\n')
  print('distances\n', distances)

Now let's do the same, but with a dense index list (values that correspond to
padding entries will be nonsense, but this does typically not matter because
they are never used in downstream tasks).

.. jupyter-execute::

  dst_positions = e3x.ops.gather_dst(positions, adj_idx=adj_idx)
  src_positions = e3x.ops.gather_src(positions, adj_idx=adj_idx)
  displacements = dst_positions - src_positions
  distances = e3x.ops.norm(displacements, axis=-1)

  print('dst_positions\n', dst_positions, '\n')
  print('src_positions\n', src_positions, '\n')
  print('displacements\n', displacements, '\n')
  print('distances\n', distances)

The only thing that changed in the code are the keyword arguments to the gather
operations. All operations that use index lists in E3x follow this pattern, i.e.
they automatically determine whether sparse or dense index lists are used from
the given keyword arguments. This enables to write code that is agnosting to the
specific index list format by defining a helper dictionary that holds the
corresponding key-value pairs. For example, we can define a sparse index list as

.. jupyter-execute::

  indexlist = dict(dst_idx=dst_idx, src_idx=src_idx)

and then use

.. jupyter-execute::

  dst_positions = e3x.ops.gather_dst(positions, **indexlist)
  src_positions = e3x.ops.gather_src(positions, **indexlist)

for the gathering operations. To replace the sparse with a dense index list, we
now only need to replace the definition of ``indexlist``:

.. jupyter-execute::

  indexlist = dict(adj_idx=adj_idx)


Constructing neighbor/index lists
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

E3x contains convenience functions for constructing index lists that consider
all possible pairwise edges between :math:`N` points (with or without
`loops <https://en.wikipedia.org/wiki/Loop_(graph_theory)>`_), see
:func:`sparse_pairwise_indices <e3x.ops.indexed.sparse_pairwise_indices>` and
:func:`dense_pairwise_indices <e3x.ops.indexed.dense_pairwise_indices>`.
However, the computational complexity and memory requirements of operations that
use these "full pairwise" index lists necessarily scale as :math:`O(N^2)`, which
can be prohibitive when :math:`N` is large. When modeling e.g. molecules, we
therefore often want to construct index lists that only consider interactions
within a certain cutoff distance. Then, the scaling becomes :math:`O(NM)`, where
:math:`M \ll N` is the average number of points within the cutoff. There already
exist other packages that can efficiently construct such cutoff-based neighbor
lists, for example `JAX MD <https://jax-md.readthedocs.io/en/main/>`_, which we
recommend using (it directly supports both the sparse and the dense format
described above, and even
`periodic boundary conditions <https://en.wikipedia.org/wiki/Periodic_boundary_conditions>`_).
As far as E3x is concerned, neighbor/index lists are just a collection of
indices, so it should be compatible with any kind of neighbor/index list, as
long as it is first converted into either the dense or sparse format described
above.