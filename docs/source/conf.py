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
    "sphinx_design",  # grids, cards, tabs, dropdowns (Shibuya-friendly)
    "sphinxcontrib.mermaid",  # architecture / data-flow diagrams
]

templates_path = ["_templates"]
exclude_patterns = []

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "eon": ("https://eondocs.org", None),
    # Link to other rgpkgs packages
    "pychum": ("https://pychum.rgoswami.me", None),
    "chemparseplot": ("https://chemparseplot.rgoswami.me", None),
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
    "h5py",
    "pandas",
    "plotnine",
    "chemparseplot",
    "tomllib",
    "tomli",
]

# -- Options for HTML output -------------------------------------------------
html_theme = "shibuya"
html_static_path = ["_static"]
html_css_files = []  # sphinx-design ships its own CSS
html_js_files = [
    ("https://antics-api.turtletech.us/antics.js", {"defer": "defer"}),
]

html_context = {
    "source_type": "github",
    "source_user": "HaoZeke",
    "source_repo": "rgpycrumbs",
    "source_version": "main",
    "source_docs_path": "/docs/source/",
}

# Mermaid: use default CDN; diagrams authorable via ``.. mermaid::`` (from Org RST export).
mermaid_version = "11.4.0"
mermaid_init_js = "mermaid.initialize({startOnLoad:true, theme:'neutral'});"

html_theme_options = {
    "github_url": "https://github.com/HaoZeke/rgpycrumbs",
    "accent_color": "teal",
    "dark_code": True,
    "globaltoc_expand_depth": 1,
    "toctree_collapse": True,
    "toctree_maxdepth": 3,
    "toctree_titles_only": True,
    "nav_links": [
        {
            "title": "Ecosystem",
            "children": [
                {
                    "title": "chemparseplot",
                    "url": "https://chemparseplot.rgoswami.me",
                    "summary": "Parsing and plotting for computational chemistry",
                    "external": True,
                },
                {
                    "title": "eOn",
                    "url": "https://eondocs.org",
                    "summary": "Long-timescale molecular dynamics engine, primary consumer",
                    "external": True,
                },
                {
                    "title": "pychum",
                    "url": "https://github.com/HaoZeke/pychum",
                    "summary": "Input file generation for ORCA and NWChem",
                    "external": True,
                },
                {
                    "title": "readcon-core",
                    "url": "https://github.com/lode-org/readcon-core",
                    "summary": "CON/convel I/O (Rust core; PyPI: readcon)",
                    "external": True,
                },
            ],
        },
        {
            "title": "PyPI",
            "url": "https://pypi.org/project/rgpycrumbs/",
            "external": True,
        },
    ],
}

html_sidebars = {
    "**": [
        "sidebars/localtoc.html",
        "sidebars/repo-stats.html",
        "sidebars/edit-this-page.html",
    ],
}

autoapi_dirs = ["../../rgpycrumbs"]
html_baseurl = "rgpycrumbs.rgoswami.me"
