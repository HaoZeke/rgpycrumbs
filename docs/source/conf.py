import os
import sys

# -- Path setup --------------------------------------------------------------
sys.path.insert(0, os.path.abspath("../../rgpycrumbs"))

# -- Project information -----------------------------------------------------
project = "rgpycrumbs"
copyright = "2025, Rohit Goswami"
author = "Rohit Goswami"
html_logo = "../../branding/logo/pycrumbs_notext.svg"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx_click",  # Generates the CLI reference (Options/Args)
    "sphinxcontrib.programoutput",  # Runs 'uv run ...' for dynamic examples
    # Include autodoc since sphinx-click relies on its mocking machinery.
    "sphinx.ext.autodoc",  # Needed for mocking machinery
    "sphinx.ext.viewcode",  # Adds '[source]' links
    "sphinx.ext.intersphinx",
    "autoapi.extension",
    "sphinxcontrib.autodoc_pydantic",
    "sphinx_sitemap",
]

templates_path = ["_templates"]
exclude_patterns = []

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "eon": ("https://eondocs.org", None),
}

# -- Mocking Dependencies ----------------------------------------------------
# Necessary for the dispatch architecture.
# Allows Sphinx to pretend these modules exist to read the Click definitions
# without crashing on missing imports.
autodoc_mock_imports = [
    "numpy",
    "matplotlib",
    "scipy",
    "ase",
    "cmcrameri",
    "polars",
    "rich",
    "ovito",
    "chemfiles",
    "pyvista",
    "featomic",
    "sklearn",
    "skmatter",
    "pypotlib",
    "pyprotochemgp",
    "click",
]

# -- Options for HTML output -------------------------------------------------
html_theme = "shibuya"
html_static_path = ["_static"]

# Shibuya theme specific options
html_theme_options = {
    "github_url": "https://github.com/Haozeke/rgpycrumbs",
    # "nav_links": [
    #     {"title": "EON Tools", "url": "eon_tools"},
    # ],
}

autoapi_dirs = ["../../rgpycrumbs"]
html_baseurl = "rgpycrumbs.rgoswami.me"
