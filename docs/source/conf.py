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

"""Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
import typing

package_path = os.path.abspath('../..')
sys.path.insert(0, os.path.abspath(package_path))
# Necessary to make jupyter_sphinx find the package in example code.
os.environ['PYTHONPATH'] = ':'.join(
    (package_path, os.environ.get('PYTHONPATH', ''))
)

import e3x

# -- Project information -----------------------------------------------------

project = 'e3x'
copyright = 'The e3x Authors'  # pylint: disable=redefined-builtin
author = 'The e3x Authors'
version = e3x.__version__
release = version

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.doctest',
    'sphinx_autodoc_typehints',
    'jupyter_sphinx',
    'nbsphinx',
]

# File extensions that are regarded as sources.
source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'restructuredtext',
    '.md': 'markdown',
}

# Autodoc configuration.
autodoc_default_options = {
    'autosummary': True,
}
autodoc_inherit_docstrings = False
autosummary_generate = True

# Automatic equation labeling with mathjax.
mathjax3_config = {
    'tex': {'tags': 'ams', 'useLabelIds': True},
}

# Configuration option for default values in sphinx_autodoc_typehints.
typehints_defaults = 'comma'

# Global setup for doctests (empty for now).
doctest_global_setup = """"""


def custom_typehints_formatter(annotation, _):
  """Custom typehints formatter."""
  annotation_as_str = str(annotation)
  if 'jaxtyping' in annotation_as_str:
    typename = annotation_as_str.replace('jaxtyping.', '').replace(
        'typing.', ''
    )
    if typename == "UInt32[Array, '2']":
      return ':func:`PRNGKey <jax.random.PRNGKey>`'
    else:
      return f':obj:`{typename} <jax.numpy.ndarray>`'
  elif typing.get_origin(annotation) is typing.Literal:
    args = ', '.join(f'``{repr(arg)}``' for arg in annotation.__args__)
    return f'{{{args}}}'


# Configuration option for typehints in sphinx_autodoc_typehints.
typehints_formatter = custom_typehints_formatter

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# Filter out all modules ending with "_test" when using autosummary.
autosummary_mock_imports = []
for path, _, files in os.walk('../../e3x'):
  for file in files:
    if file.endswith('_test.py'):
      autosummary_mock_imports.append(
          os.path.join(path, file)
          .split('../../')[-1]
          .replace('.py', '')
          .replace('/', '.')
      )

# links to other documentations
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'jax': ('https://jax.readthedocs.io/en/latest/', None),
    'flax': ('https://flax.readthedocs.io/en/latest/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    # "sympy": ("https://docs.sympy.org/latest/index.html", None),
    # "plotly": ("https://plotly.com/python-api-reference/index.html", None)
}
myst_url_schemes = [
    'http',
    'https',
]

# latex preamble (load custom packages for nicer math)
latex_preamble = [
    (
        '\\usepackage{amssymb}',
        '\\usepackage{amsmath}',
        '\\usepackage{amsxtra}',
    ),
]

# rst prolog to support colored text
rst_prolog = """
.. include:: <s5defs.txt>

"""

# -- Options for nsphinx -----------------------------------------------------
nbsphinx_execute_arguments = ["--InlineBackend.figure_formats={'svg', 'pdf'}"]

nbsphinx_prolog = """
{% set docname = env.doc2path(env.docname, base=None) %}

.. raw:: html

  <a href="../{{ docname }}" download>Download this example as Jupyter notebook</a>
"""

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = '_static/logo.svg'
html_theme_options = {
    'logo_only': False,
    'display_version': True,
}
html_scaled_image_link = False

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Custom css files
html_css_files = ['css/s4defs-roles.css']
