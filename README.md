# e3x library

e3x is a JAX library for constructing efficient E(3)-equivariant deep learning
architectures built on top of [Flax](https://flax.readthedocs.io).

The goal is to provide common neural network building blocks for
E(3)-equivariant architectures to make the development of models operating on
three-dimensional data (point clouds, polygon meshes, etc.) easier.

This is not an officially supported Google product.

## Installation

Clone this repository, enter the directory and run
```console
> python -m pip install .
```
If you are a developer, you might want to also install the optional development
dependencies by running
```console
> python -m pip install .[dev]
```
instead.

### Running unit tests

Running unit tests requires installed development dependencies (see above).
```console
> pytest tests
```

### Building the documentation

Building the documentation requires installed development dependencies (see
above).
```console
> cd docs
> make html
```
