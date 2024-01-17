<div align="center">
<img src="https://raw.githubusercontent.com/google-research/e3x/main/docs/source/_static/logo.svg" alt="logo" width="200"></img>
</div>

# E3x: E(3)-Equivariant Deep Learning Made Easy

![Autopublish Workflow](https://github.com/google-research/e3x/actions/workflows/pytest_and_autopublish.yml/badge.svg)
![PyPI version](https://img.shields.io/pypi/v/e3x)
[![Documentation Status](https://readthedocs.org/projects/e3x/badge/?version=latest)](https://e3x.readthedocs.io/en/latest/?badge=latest)

[E3x](https://e3x.readthedocs.io) is a [JAX](https://jax.readthedocs.io) library
for constructing efficient E(3)-equivariant deep learning architectures built on
top of [Flax](https://flax.readthedocs.io).

The goal is to provide common neural network building blocks for
E(3)-equivariant architectures to make the development of models operating on
three-dimensional data (point clouds, polygon meshes, etc.) easier.

This is not an officially supported Google product.

## Installation

The easiest way to install E3x is via the Python Package Index (PyPI). Simply
run
```console
> python -m pip install --upgrade e3x
```
and you should be good to go.

Alternatively, you can clone this repository, enter the directory and run:
```console
> python -m pip install .
```

If you are a developer, you might want to also install the optional development
dependencies by running
```console
> python -m pip install .[dev]
```
instead.

## Documentation

Documentation for E3x, including usage examples and tutorials can be found
[here](https://e3x.readthedocs.io). For a more detailed overview over the
mathematical theory behind E3x, please refer to
[this paper](https://arxiv.org/abs/2401.07595).

## Citing E3x

If you find E3x useful and use it in your work, please cite:
```
@article{unke2024e3x,
  title={\texttt{E3x}: $\mathrm{E}(3)$-Equivariant Deep Learning Made Easy},
  author={Unke, Oliver T. and Maennel, Hartmut},
  journal={arXiv preprint arXiv:2401.07595},
  year={2024}
}
```
