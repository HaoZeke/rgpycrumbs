import os
import sys

# -- Path setup --------------------------------------------------------------
sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------
project = "rgpycrumbs"
copyright = "2025, Rohit Goswami"
author = "Rohit Goswami"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx_click",
    "autodoc2",
    # Include autodoc since sphinx-click relies on its mocking machinery.
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

templates_path = ["_templates"]
exclude_patterns = []

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
    "nav_links": [
        {"title": "EON Tools", "url": "eon_tools"},
    ],
}

# -- Autodoc2 Configuration
autodoc2_packages = [
    {
        "path": "../../rgpycrumbs",
        # Exclude files handled by click..? (TBD)
        "exclude_files": ["rgpycrumbs/eon/plt_neb.py"],
    },
]
