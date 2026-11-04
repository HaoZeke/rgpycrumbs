"""Tests for build_llms.py, run on a small repository built in a temp dir.

python -m unittest discover -s scripts/llms -p "test_*.py"
"""

import contextlib
import io
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_llms

CONFIG = """
[project]
name = "demo"
summary = "Demo adds numbers."
notes = ["Links are relative to the repository root."]
links = "relative"

[docs]
root = "docs/orgmode"
index = "index.org"
section_order = ["Start here", "Reference", "Skills"]
optional = ["notes"]
full_exclude = ["notes"]

[sections]
guide = "Start here"
api = "Reference"

[[links]]
section = "Reference"
title = "Changes"
path = "CHANGES.md"
description = "Release notes."
full = true

[skills]
dir = "skills"
section = "Skills"
"""

FILES = {
    "docs/orgmode/index.org": """
        #+title: demo

        Demo adds numbers.

        * Where to start

        - @@rst::doc:`guide`@@ :: How to add two numbers.

        * Contents

        #+begin_export rst
        .. toctree::
           :caption: Start here

           guide
           api
           notes
        #+end_export
        """,
    "docs/orgmode/guide.org": """
        #+title: Guide

        Call =add= with two numbers.

        * Adding

        #+begin_src python
        # a comment, not a heading
        print(add(1, 2))
        #+end_src

        See [[file:api.org][the API]] and [[https://example.org][the site]].

        * Nothing here
        * Last

        Done.
        """,
    "docs/orgmode/api.org": """
        #+title: API

        The =add= function returns the sum.
        """,
    "docs/orgmode/notes.org": """
        #+title: Notes

        Run logs live in a lab notebook.
        """,
    "CHANGES.md": """
        # Changes

        ## 1.0

        First release.
        """,
    "skills/demo-add/SKILL.md": """
        ---
        name: demo-add
        description: Add two numbers with demo.
        ---

        # Adding

        Call `add(1, 2)`.
        """,
}


class Repo:
    """A throwaway repository holding FILES and llms.toml."""

    def __init__(self, files=FILES, config=CONFIG):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for rel, text in files.items():
            path = self.root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(textwrap.dedent(text).lstrip(), encoding="utf-8")
        self.config = self.root / "llms.toml"
        self.config.write_text(config, encoding="utf-8")

    def run(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            status = build_llms.main(
                ["--root", str(self.root), "--config", str(self.config), *args]
            )
        return status, err.getvalue()

    def read(self, rel):
        return (self.root / rel).read_text(encoding="utf-8")

    def close(self):
        self.tmp.cleanup()


class BuildTest(unittest.TestCase):
    def setUp(self):
        self.repo = Repo()
        self.addCleanup(self.repo.close)
        status, err = self.repo.run()
        self.assertEqual(status, 0, err)
        self.index = self.repo.read("llms.txt")
        self.full = self.repo.read("llms-full.txt")

    def test_index_header_follows_llmstxt(self):
        lines = self.index.splitlines()
        self.assertEqual(lines[0], "# demo")
        self.assertEqual(lines[2], "> Demo adds numbers.")
        self.assertIn("Links are relative to the repository root.", self.index)

    def test_sections_in_configured_order_with_optional_last(self):
        headings = [line for line in self.index.splitlines() if line.startswith("## ")]
        self.assertEqual(
            headings, ["## Start here", "## Reference", "## Skills", "## Optional"]
        )

    def test_page_links_are_relative_and_described(self):
        self.assertIn(
            "- [Guide](docs/orgmode/guide.org): How to add two numbers.", self.index
        )
        self.assertIn(
            "- [API](docs/orgmode/api.org): The `add` function returns the sum.",
            self.index,
        )
        self.assertIn("- [Changes](CHANGES.md): Release notes.", self.index)
        self.assertIn(
            "- [demo-add](skills/demo-add/SKILL.md): Add two numbers with demo.",
            self.index,
        )

    def test_optional_page_is_linked_but_not_inlined(self):
        optional = self.index.split("## Optional", 1)[1]
        self.assertIn("[Notes](docs/orgmode/notes.org)", optional)
        self.assertNotIn("lab notebook", self.full)

    def test_full_text_converts_org(self):
        self.assertIn("## Guide\n\nSource: docs/orgmode/guide.org", self.full)
        self.assertIn("Call `add` with two numbers.", self.full)
        self.assertIn("### Adding", self.full)
        self.assertIn(
            "```python\n# a comment, not a heading\nprint(add(1, 2))\n```", self.full
        )
        self.assertIn("[the site](https://example.org)", self.full)
        self.assertIn("(docs/orgmode/api.org)", self.full)

    def test_heading_without_content_is_dropped(self):
        self.assertNotIn("Nothing here", self.full)
        self.assertIn("### Last", self.full)

    def test_links_and_skills_are_inlined(self):
        self.assertIn("## Changes\n\nSource: CHANGES.md", self.full)
        self.assertIn("First release.", self.full)
        self.assertIn("## Skill: demo-add", self.full)
        self.assertIn("Call `add(1, 2)`.", self.full)

    def test_check_passes_when_current_and_fails_when_stale(self):
        self.assertEqual(self.repo.run("--check")[0], 0)
        (self.repo.root / "docs/orgmode/api.org").write_text(
            "#+title: API\n\nNow it subtracts.\n", encoding="utf-8"
        )
        status, err = self.repo.run("--check")
        self.assertEqual(status, 1)
        self.assertIn("stale: llms.txt, llms-full.txt", err)

    def test_output_is_deterministic(self):
        self.repo.run()
        self.assertEqual(self.repo.read("llms.txt"), self.index)
        self.assertEqual(self.repo.read("llms-full.txt"), self.full)


class RefusalTest(unittest.TestCase):
    def test_machine_path_stops_generation(self):
        for top in ("home", "Users", "tmp"):
            with self.subTest(top=top):
                # Joined at run time so this file does not itself hold such a path.
                machine_path = "/".join(["", top, "someone", "runs"])
                files = dict(FILES)
                files["docs/orgmode/api.org"] = (
                    f"#+title: API\n\nData sits in {machine_path}.\n"
                )
                repo = Repo(files)
                self.addCleanup(repo.close)
                status, err = repo.run()
                self.assertEqual(status, 2)
                self.assertIn("machine-local path", err)
                self.assertFalse((repo.root / "llms.txt").exists())

    def test_missing_link_target_stops_generation(self):
        files = {k: v for k, v in FILES.items() if k != "CHANGES.md"}
        repo = Repo(files)
        self.addCleanup(repo.close)
        status, err = repo.run()
        self.assertEqual(status, 2)
        self.assertIn("links to CHANGES.md, which does not exist", err)


MAP_CONFIG = (
    CONFIG
    + """
[output]
index = ""

[map]
path = "docs/source/llms.txt"
site = "https://demo.example/"
generated = ["autoapi/"]
"""
)

MAP = """
    # demo

    > Demo adds numbers.

    - [Home](https://demo.example/): The front page.
    - [Guide](https://demo.example/guide.html): How to add.
    - [API](https://demo.example/api.html#add): The add function.
    - [Reference](https://demo.example/autoapi/demo/index.html): From the code.
    - [Elsewhere](https://other.example/gone.html): Another site.
    """


class SiteMapTest(unittest.TestCase):
    """A hand-written llms.txt is kept, and its links to the site are checked."""

    def repo(self, site_map):
        repo = Repo({**FILES, "docs/source/llms.txt": site_map}, MAP_CONFIG)
        self.addCleanup(repo.close)
        return repo

    def test_map_is_kept_and_only_the_full_text_written(self):
        repo = self.repo(MAP)
        status, err = repo.run()
        self.assertEqual(status, 0, err)
        self.assertFalse((repo.root / "llms.txt").exists())
        self.assertIn("## Guide", repo.read("llms-full.txt"))
        self.assertEqual(repo.read("docs/source/llms.txt"), textwrap.dedent(MAP).lstrip())
        self.assertIn("note: docs/source/llms.txt does not link notes", err)
        self.assertEqual(repo.run("--check")[0], 0)

    def test_link_to_a_missing_page_fails(self):
        repo = self.repo(MAP + "    - [Gone](https://demo.example/gone.html): Removed.\n")
        for args in ((), ("--check",)):
            with self.subTest(args=args):
                status, err = repo.run(*args)
                self.assertEqual(status, 2)
                self.assertIn(
                    "broken map link: docs/source/llms.txt:10: "
                    "https://demo.example/gone.html has no page under orgmode",
                    err,
                )

    def test_page_name(self):
        cases = {
            "": "index",
            "index.html": "index",
            "guide.html#adding": "guide",
            "api.html?highlight=add": "api",
            "tools/": "tools/index",
            "tools/eon/plt_neb": "tools/eon/plt_neb",
        }
        for rest, name in cases.items():
            with self.subTest(rest=rest):
                self.assertEqual(build_llms.page_name(rest), name)


class OrgDetailTest(unittest.TestCase):
    """Org and ox-rst constructs the rgpycrumbs manual uses."""

    def full(self, api_page):
        files = {**FILES, "docs/orgmode/api.org": api_page, "docs/orgmode/plot.png": ""}
        repo = Repo(files)
        self.addCleanup(repo.close)
        status, err = repo.run()
        self.assertEqual(status, 0, err)
        return repo.read("llms-full.txt")

    def test_title_markup_becomes_code(self):
        full = self.full("#+title: The ~add~ function\n\nIt adds.\n")
        self.assertIn("## The `add` function\n", full)

    def test_plain_file_link_to_an_image(self):
        full = self.full("#+title: API\n\nfile:plot.png\n")
        self.assertIn("![plot.png](docs/orgmode/plot.png)", full)

    def test_rst_block_numbered_list_and_repeated_title(self):
        page = """\
            #+title: API

            #+begin_export rst
            API
            ===

            #. First
            #. Second
            #+end_export
            """
        full = self.full(textwrap.dedent(page))
        self.assertIn("1. First\n1. Second", full)
        self.assertNotIn("**API**", full)


if __name__ == "__main__":
    unittest.main()
