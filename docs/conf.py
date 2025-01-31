# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "PyGenAlgo"
copyright = "2024, Michalis Vrettas, PhD"
author = "Michalis Vrettas, PhD"
release = "01.70.00"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon"
]

templates_path = ["_templates"]

# The suffix of source filenames.
source_suffix = {".rst": "restructuredtext"}

# The encoding of source files.
source_encoding = "utf-8"

# The master toctree document.
master_doc = "index"

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme_path = ["_themes"]

# html_theme = "alabaster", "sphinx_rtd_theme"
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
