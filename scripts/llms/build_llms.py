#!/usr/bin/env python3
"""Generate llms.txt and llms-full.txt from the documentation sources.

llms.txt follows the layout at https://llmstxt.org: an H1 title, a
blockquote summary, a few notes, then H2 sections whose list items link
to pages. llms-full.txt has the same header followed by every page as
Markdown, so one file holds the whole manual.

The page list comes from the toctree directives of the docs index, so a
page added to a toctree appears here on the next run. A page summary is
taken, in order, from the [descriptions] table of llms.toml, from a
description list on the index page, from a ``#+description:`` keyword,
and from the first sentence of the page.

Generation fails with exit status 2 when a [[links]] path does not
exist, or when the output would carry an absolute path into a temp,
home or scratch directory; fix the page or list it in
docs.full_exclude.

A project that keeps a hand-written llms.txt sets ``output.index = ""``
so only llms-full.txt is written, and names its file in [map]. Every
link in that file under the site address must then resolve to a page
of the manual, or generation fails with exit status 2; pages the file
does not link are printed as notes.

    python scripts/llms/build_llms.py          # rewrite the outputs
    python scripts/llms/build_llms.py --check  # exit 1 when one is stale

Standard library only. Needs Python 3.11 or newer for tomllib.
"""

from __future__ import annotations

import argparse
import posixpath
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import tomllib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONFIG = HERE / "llms.toml"

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}
HEADING_CAP = 6
# Absolute paths that only exist on one machine. The look-behind skips
# URLs and relative paths that contain the same directory names.
DEFAULT_FORBID = (r"(?<![\w.~$-])/(?:tmp|home|workspace|scratch|Users)/",)


# ---------------------------------------------------------------------------
# Configuration and URLs


@dataclass
class Page:
    name: str
    path: Path | None
    title: str
    section: str
    url: str
    description: str = ""
    optional: bool = False
    text: str = ""
    doc_title: str = ""


@dataclass
class Context:
    """What the converters need to resolve links from one source file."""

    root: Path
    docs_root: Path
    source: Path
    raw_base: str
    tree_base: str
    page_url: dict[str, str] = field(default_factory=dict)
    page_title: dict[str, str] = field(default_factory=dict)
    page_template: str | None = None

    def current_page(self) -> str:
        try:
            rel = self.source.relative_to(self.docs_root)
        except ValueError:
            return ""
        return rel.with_suffix("").as_posix()

    def repo_rel(self, path: Path) -> str | None:
        try:
            return path.resolve().relative_to(self.root.resolve()).as_posix()
        except ValueError:
            return None

    def doc_target(self, target: str) -> str:
        target = target.strip()
        if target.startswith("/"):
            name = target.lstrip("/")
        else:
            base = posixpath.dirname(self.current_page())
            name = posixpath.normpath(posixpath.join(base, target))
        return name.removesuffix(".org").removesuffix(".rst")

    def doc_link(self, content: str) -> str:
        text, target = split_title_target(content)
        name = self.doc_target(target)
        title = text or self.page_title.get(name) or target
        url = self.page_url.get(name)
        if url is None and self.page_template is not None:
            url = format_page_url(self.page_template, name)
        return f"[{title}]({url})" if url else title

    def file_url(self, resolved: Path) -> str | None:
        """URL of a file or directory inside the repository, else None."""
        rel = self.repo_rel(resolved)
        if rel is None:
            return None
        if resolved.is_dir():
            return self.tree_base + rel
        return self.raw_base + rel


def split_title_target(content: str) -> tuple[str, str]:
    m = re.fullmatch(r"\s*(.*?)\s*<([^<>]+)>\s*", content, re.S)
    if m:
        return m.group(1), m.group(2)
    return "", content.strip()


def format_page_url(template: str, name: str) -> str:
    if name == "index":
        page = ""
    elif name.endswith("/index"):
        page = name[: -len("index")]
    else:
        page = name
    return template.format(page=page)


def page_name(rest: str) -> str:
    """Page name for a site address with the site prefix removed."""
    rest = rest.split("#", 1)[0].split("?", 1)[0].removesuffix(".html")
    if rest in {"", "index"}:
        return "index"
    if rest.endswith("/"):
        return rest + "index"
    return rest


def load_config(path: Path) -> dict:
    with path.open("rb") as fh:
        return tomllib.load(fh)


def github_bases(repo: str, branch: str) -> tuple[str, str]:
    """Base URLs for raw file contents and for directory listings."""
    repo = repo.rstrip("/")
    slug = repo.removeprefix("https://github.com/")
    return (
        f"https://raw.githubusercontent.com/{slug}/{branch}/",
        f"{repo}/tree/{branch}/",
    )


# ---------------------------------------------------------------------------
# Small text helpers


def indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" \t"))


def dedent(lines: list[str]) -> list[str]:
    widths = [indent_of(line) for line in lines if line.strip()]
    if not widths:
        return [""] * len(lines)
    cut = min(widths)
    return [line[cut:] if line.strip() else "" for line in lines]


def trim_blank(lines: list[str]) -> list[str]:
    start, end = 0, len(lines)
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return lines[start:end]


def fence(lang: str, body: list[str], prefix: str = "") -> list[str]:
    ticks = "```"
    while any(line.lstrip().startswith(ticks) for line in body):
        ticks += "`"
    head = f"{prefix}{ticks}{lang}"
    return [
        head,
        *[f"{prefix}{line}" if line else "" for line in body],
        f"{prefix}{ticks}",
    ]


def fenced(lines: list[str]) -> list[bool]:
    """True for each line that is a code fence or sits inside one."""
    mask = []
    ticks = None
    for line in lines:
        stripped = line.strip()
        if ticks is None:
            m = re.match(r"(`{3,})", stripped)
            if m:
                ticks = m.group(1)
            mask.append(ticks is not None)
        else:
            mask.append(True)
            if stripped.startswith(ticks) and not stripped.strip("`"):
                ticks = None
    return mask


def tidy(lines: list[str]) -> str:
    """Strip trailing spaces and collapse runs of blank lines outside fences."""
    lines = [line.rstrip() for line in lines]
    out: list[str] = []
    for line, in_fence in zip(lines, fenced(lines), strict=True):
        if not in_fence and not line and out and not out[-1]:
            continue
        out.append(line)
    return "\n".join(trim_blank(out)) + "\n"


def drop_empty_headings(lines: list[str]) -> list[str]:
    """Remove ATX headings with nothing under them before the next peer."""
    while True:
        kinds: list[tuple[str, int]] = []
        for line, in_fence in zip(lines, fenced(lines), strict=True):
            m = None if in_fence else re.match(r"(#{1,6})\s", line)
            if m:
                kinds.append(("heading", len(m.group(1))))
            else:
                kinds.append(("text" if line.strip() else "blank", 0))
        drop = set()
        for i, (kind, level) in enumerate(kinds):
            if kind != "heading":
                continue
            j = i + 1
            while j < len(kinds) and kinds[j][0] == "blank":
                j += 1
            if j == len(kinds) or (kinds[j][0] == "heading" and kinds[j][1] <= level):
                drop.add(i)
        if not drop:
            return lines
        lines = [line for i, line in enumerate(lines) if i not in drop]


MD_LINK_RE = re.compile(
    r"(?P<bang>!?)\[(?P<text>[^\]]*)\]\((?P<target>[^)\s]+)(?P<title>\s+\"[^\"]*\")?\)"
)


def rebase_links(lines: list[str], ctx: Context) -> list[str]:
    """Point relative Markdown links at the repository, not the source file."""

    def fix(m: re.Match[str]) -> str:
        target = m.group("target")
        if re.match(r"[A-Za-z][\w+.-]*:", target) or target.startswith(("#", "/", "<")):
            return m.group(0)
        path_part, _, frag = target.partition("#")
        url = ctx.file_url((ctx.source.parent / path_part).resolve())
        if url is None:
            return m.group(0)
        url += f"#{frag}" if frag else ""
        return f"{m.group('bang')}[{m.group('text')}]({url}{m.group('title') or ''})"

    return [
        line if in_fence else MD_LINK_RE.sub(fix, line)
        for line, in_fence in zip(lines, fenced(lines), strict=True)
    ]


def code_span(text: str) -> str:
    ticks = "`"
    while ticks in text:
        ticks += "`"
    pad = " " if text.startswith("`") or text.endswith("`") else ""
    return f"{ticks}{pad}{text}{pad}{ticks}"


class Shelf:
    """Placeholder store that keeps converted spans away from later passes."""

    def __init__(self) -> None:
        self.items: list[str] = []

    def put(self, text: str) -> str:
        self.items.append(text)
        return f"\x00{len(self.items) - 1}\x00"

    def restore(self, text: str) -> str:
        pattern = re.compile(r"\x00(\d+)\x00")
        while pattern.search(text):
            text = pattern.sub(lambda m: self.items[int(m.group(1))], text)
        return text


def plain(text: str, keep_code: bool = False) -> str:
    """Markdown inline text with links and emphasis (and code) markers removed."""
    shelf = Shelf()
    text = re.sub(r"`+[^`]*`+", lambda m: shelf.put(m.group(0)), text)
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"(\*\*|\*|~~)(.+?)\1", r"\2", text)
    text = shelf.restore(text)
    if not keep_code:
        text = re.sub(r"`+ ?([^`]*?) ?`+", r"\1", text)
    return " ".join(text.split())


ABBREVIATIONS = (
    "e.g.",
    "i.e.",
    "et al.",
    "cf.",
    "vs.",
    "etc.",
    "Eq.",
    "Fig.",
    "Ref.",
    "Sec.",
    "No.",
)


def first_sentence(text: str) -> str:
    text = " ".join(text.split())
    for m in re.finditer(r"[.!?](?=\s+[A-Z0-9`(\[*]|\s*$)", text):
        end = m.end()
        head = text[:end]
        if any(head.endswith(a) for a in ABBREVIATIONS):
            continue
        if re.search(r"\b[A-Z]\.$", head):
            continue
        return head
    return text


# ---------------------------------------------------------------------------
# reStructuredText (export blocks)

DIRECTIVE_RE = re.compile(r"^\.\.\s+([A-Za-z][\w:+-]*)::(?:\s+(.*))?$")
OPTION_RE = re.compile(r"^:([\w-]+):(?:\s+(.*))?$")
ADORN_RE = re.compile(r"^([=\-`:'\"~^_*+#<>.])\1{2,}\s*$")
ROLE_RE = re.compile(r":(?P<role>[A-Za-z][\w:+-]*):`(?P<body>[^`]+)`")

ADMONITIONS = {
    "note": "Note",
    "tip": "Tip",
    "hint": "Hint",
    "important": "Important",
    "warning": "Warning",
    "caution": "Caution",
    "attention": "Attention",
    "danger": "Danger",
    "error": "Error",
    "seealso": "See also",
    "todo": "To do",
}
VERSION_NOTES = {
    "versionadded": "Added in version",
    "versionchanged": "Changed in version",
    "deprecated": "Deprecated since version",
}
DROPPED_DIRECTIVES = {
    "toctree",
    "contents",
    "bibliography",
    "only",
    "raw",
    "index",
    "highlight",
    "currentmodule",
    "module",
    "meta",
    "tabularcolumns",
    "autosummary",
    "automodule",
    "autoclass",
    "autofunction",
    "automethod",
    "autoattribute",
    "autodata",
    "autoexception",
    "autoapimodule",
    "autoapiclass",
    "autoapifunction",
    "rust:crate",
}
CONTAINERS = {"grid", "tab-set", "container", "card-carousel", "class", "rst-class"}
TITLED_CONTAINERS = {"grid-item-card", "card", "tab-item", "dropdown", "topic", "sidebar"}


@dataclass
class RstNode:
    kind: str
    text: str = ""
    lines: list[str] = field(default_factory=list)
    name: str = ""
    arg: str = ""
    options: dict[str, str] = field(default_factory=dict)
    raw: list[str] = field(default_factory=list)


def rst_nodes(lines: list[str]) -> list[RstNode]:
    """Split dedented rst into titles, directives, literal blocks and text."""
    nodes: list[RstNode] = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        m = DIRECTIVE_RE.match(line)
        if m and indent_of(line) == 0:
            start = i
            name, arg = m.group(1).lower(), (m.group(2) or "").strip()
            i += 1
            options: dict[str, str] = {}
            while i < n and lines[i].strip() and indent_of(lines[i]) > 0:
                om = OPTION_RE.match(lines[i].strip())
                if not om:
                    break
                options[om.group(1)] = (om.group(2) or "").strip()
                i += 1
            body: list[str] = []
            while i < n and (not lines[i].strip() or indent_of(lines[i]) > 0):
                body.append(lines[i])
                i += 1
            nodes.append(
                RstNode(
                    "directive",
                    name=name,
                    arg=arg,
                    options=options,
                    lines=dedent(trim_blank(body)),
                    raw=trim_blank(lines[start:i]),
                )
            )
            continue
        if line.startswith("..") and indent_of(line) == 0:
            i += 1
            while i < n and (not lines[i].strip() or indent_of(lines[i]) > 0):
                i += 1
            continue
        if (
            ADORN_RE.match(line)
            and i + 2 < n
            and lines[i + 1].strip()
            and ADORN_RE.match(lines[i + 2])
        ):
            nodes.append(RstNode("title", text=lines[i + 1].strip()))
            i += 3
            continue
        if (
            i + 1 < n
            and not ADORN_RE.match(line)
            and ADORN_RE.match(lines[i + 1])
            and len(lines[i + 1].strip()) >= len(line.strip())
        ):
            nodes.append(RstNode("title", text=line.strip()))
            i += 2
            continue
        para: list[str] = []
        while i < n and lines[i].strip():
            para.append(lines[i])
            i += 1
        literal = para and para[-1].rstrip().endswith("::")
        if literal:
            last = para[-1].rstrip()
            para[-1] = (
                last[:-2].rstrip()
                if last[:-2].endswith(" ") or last == "::"
                else last[:-1]
            )
            if not para[-1].strip():
                para.pop()
        if para:
            nodes.append(RstNode("text", lines=para))
        if literal:
            block: list[str] = []
            j = i
            while j < n and (not lines[j].strip() or indent_of(lines[j]) > 0):
                block.append(lines[j])
                j += 1
            block = trim_blank(block)
            if block:
                nodes.append(RstNode("literal", lines=dedent(block)))
                i = j
    return nodes


def rst_role(role: str, body: str, ctx: Context) -> str:
    role = role.lower()
    if role == "doc":
        return ctx.doc_link(body)
    if role == "math":
        return f"${body}$"
    if role.startswith("cite") or role.startswith("footcite"):
        keys = [k.strip() for k in body.split(",") if k.strip()]
        return "[" + "; ".join(f"@{k}" for k in keys) + "]"
    if role == "eq":
        return f"({body})"
    text, target = split_title_target(body)
    if role in {"ref", "numref", "term", "keyword", "download", "any"}:
        return text or target
    shown = text or target.lstrip("~!")
    if target.startswith("~") and not text:
        shown = shown.rsplit(".", 1)[-1]
    return code_span(shown)


def rst_inline(text: str, ctx: Context) -> str:
    shelf = Shelf()
    text = re.sub(
        r"``(.+?)``", lambda m: shelf.put(code_span(m.group(1))), text, flags=re.S
    )
    text = ROLE_RE.sub(
        lambda m: shelf.put(rst_role(m.group("role"), m.group("body"), ctx)), text
    )
    text = re.sub(
        r"`([^`<]+?)\s*<([^`>]+)>`__?",
        lambda m: shelf.put(
            f"[{m.group(1)}]({m.group(2)})" if "://" in m.group(2) else m.group(1)
        ),
        text,
    )
    text = re.sub(r"`([^`]+)`__?", lambda m: m.group(1), text)
    text = re.sub(r"`([^`]+)`", lambda m: shelf.put(f"*{m.group(1)}*"), text)
    text = re.sub(r"\\([*`_|\\])", r"\1", text)
    return shelf.restore(text)


def rst_to_md(lines: list[str], ctx: Context, page_title: str = "") -> list[str]:
    out: list[str] = []
    for node in rst_nodes(dedent(lines)):
        if node.kind == "title":
            text = rst_inline(node.text, ctx)
            if page_title and plain(text).lower() == plain(page_title).lower():
                continue
            out += [f"**{text}**", ""]
        elif node.kind == "text":
            converted = rst_inline("\n".join(node.lines), ctx).split("\n")
            out += [re.sub(r"^(\s*)#\.\s", r"\g<1>1. ", line) for line in converted]
            out.append("")
        elif node.kind == "literal":
            out += [*fence("text", node.lines), ""]
        else:
            out += [*rst_directive(node, ctx, page_title), ""]
    return out


def blockquote(label: str, body: list[str]) -> list[str]:
    body = trim_blank(body)
    if body and body[0] and not body[0].startswith(("```", "- ", "* ", "$$", "|")):
        lines = [f"**{label}:** {body[0]}", *body[1:]]
    else:
        lines = [f"**{label}:**", *([""] if body else []), *body]
    return [f"> {line}" if line else ">" for line in lines]


def rst_directive(node: RstNode, ctx: Context, page_title: str) -> list[str]:
    name, arg, opts, body = node.name, node.arg, node.options, node.lines
    if name in DROPPED_DIRECTIVES:
        return []
    if name == "math":
        chunks: list[list[str]] = [[arg]] if arg else []
        current: list[str] = []
        for line in [*body, ""]:
            if line.strip():
                current.append(line)
            elif current:
                chunks.append(current)
                current = []
        out: list[str] = []
        for chunk in chunks:
            out += ["$$", *chunk, "$$", ""]
        return out
    if name in {"code-block", "code", "sourcecode"}:
        return fence(arg, body)
    if name == "mermaid":
        return fence("mermaid", body) if body else []
    if name in {"graphviz", "digraph", "graph"}:
        return fence("dot", body) if body else []
    if name in {"image", "figure"}:
        alt = opts.get("alt", "") or posixpath.basename(arg)
        out = [f"![{alt}]({arg})"]
        caption = trim_blank(body)
        if name == "figure" and caption:
            first: list[str] = []
            for line in caption:
                if not line.strip():
                    break
                first.append(line.strip())
            out += ["", f"*{rst_inline(' '.join(first), ctx)}*"]
        return out
    if name in ADMONITIONS or name == "admonition" or name in VERSION_NOTES:
        if name == "admonition":
            label, lead = rst_inline(arg, ctx), []
        elif name in VERSION_NOTES:
            label, lead = f"{VERSION_NOTES[name]} {arg}".strip(), []
        else:
            label, lead = ADMONITIONS[name], ([arg, ""] if arg else [])
        return blockquote(label, rst_to_md(lead + body, ctx, page_title))
    if name == "rubric":
        return [f"**{rst_inline(arg, ctx)}**"]
    if name in CONTAINERS:
        return rst_to_md(body, ctx, page_title)
    if name in TITLED_CONTAINERS:
        out = [f"**{rst_inline(arg, ctx)}**", ""] if arg else []
        link = opts.get("link")
        if link:
            kind = opts.get("link-type", "url")
            if kind == "doc":
                out += [ctx.doc_link(link), ""]
            elif kind == "url" or "://" in link:
                out += [f"<{link}>", ""]
        return out + rst_to_md(body, ctx, page_title)
    if name == "include":
        target = (ctx.source.parent / arg).resolve()
        rel = ctx.repo_rel(target)
        return [f"(Included file: `{rel or arg}`.)"]
    return fence("rst", node.raw)


# ---------------------------------------------------------------------------
# Org

KEYWORD_RE = re.compile(r"^\s*#\+([A-Za-z_]+):\s*(.*)$")
BEGIN_RE = re.compile(r"^(\s*)#\+begin_(\w+)\b(.*)$", re.I)
HEADING_RE = re.compile(r"^(\*+)\s+(.*?)\s*$")
TAGS_RE = re.compile(r"\s+(:[\w@#%:]+:)\s*$")
BULLET_RE = re.compile(r"^(\s*)([-+]|\*(?=\s)|\d+[.)])\s+(.*)$")
TABLE_RE = re.compile(r"^\s*\|")
FIXED_RE = re.compile(r"^\s*:(\s|$)")
DRAWER_RE = re.compile(r"^\s*:([A-Za-z_-]+):\s*$")
LATEX_ENV_RE = re.compile(r"^\s*\\begin\{([A-Za-z*]+)\}")
FOOTDEF_RE = re.compile(r"^\[fn:([\w-]+)\]\s*(.*)$")
SRC_LANGS = {"jupyter-python": "python", "elisp": "emacs-lisp", "sh": "shell"}

# \x00 delimits Shelf placeholders; \x01 and \x02 bracket description-list terms.
PRE = r"(?:^|(?<=[\s\-('\"{\x00\x01]))"
POST = r"(?=[\s\-.,:!?;'\")}\[\\\x00\x02]|$)"
TERM_RE = re.compile(r"\x01(.*?)\x02", re.S)


def bold_term(m: re.Match[str]) -> str:
    term = m.group(1).strip()
    return term if re.fullmatch(r"\*\*.*\*\*", term) else f"**{term}**"


def emphasis(marker: str) -> re.Pattern[str]:
    m = re.escape(marker)
    body = r"(?P<body>[^\s" + m + r"]|[^\s][^\n]*?(?:\n[^\n]*?)?[^\s])"
    return re.compile(PRE + m + body + m + POST, re.M)


VERBATIM = (emphasis("="), emphasis("~"))
# An Org plain link, ``file:path`` without brackets; ox-rst inlines an
# image this way.
PLAIN_FILE_RE = re.compile(r"(?<![\w\[:/])file:[^\s\[\]()<>]+")
BOLD, ITALIC, STRIKE = emphasis("*"), emphasis("/"), emphasis("+")
LINK_RE = re.compile(
    r"\[\[(?P<target>(?:[^\]\\]|\\.)+)\](?:\[(?P<desc>(?:[^\]]|\](?!\]))+)\])?\]"
)


def org_link(target: str, desc: str | None, ctx: Context) -> str:
    target = target.replace("\\[", "[").replace("\\]", "]")
    if re.match(r"(https?|ftp)://|mailto:", target):
        return f"[{desc}]({target})" if desc else f"<{target}>"
    if target.startswith("doi:"):
        url = "https://doi.org/" + target[4:]
        return f"[{desc or target}]({url})"
    m = re.match(r"(?:cite[a-z]*|citep|citet):(.*)", target)
    if m:
        keys = [
            k.strip().lstrip("&@") for k in re.split(r"[,;]", m.group(1)) if k.strip()
        ]
        return "[" + "; ".join(f"@{k}" for k in keys) + "]"
    is_file = target.startswith(("file:", "./", "../", "/", "~/"))
    path_part = target.removeprefix("file:").split("::", 1)[0]
    if (
        not is_file
        and re.search(r"\.[A-Za-z0-9]{1,5}$", path_part)
        and ":" not in path_part
    ):
        is_file = True
    if is_file and path_part:
        resolved = (ctx.source.parent / path_part).resolve()
        docs_rel = None
        try:
            docs_rel = resolved.relative_to(ctx.docs_root.resolve())
        except ValueError:
            pass
        if docs_rel is not None and resolved.suffix == ".org":
            name = docs_rel.with_suffix("").as_posix()
            title = desc or ctx.page_title.get(name, name)
            url = ctx.page_url.get(name)
            if url is None and ctx.page_template is not None:
                url = format_page_url(ctx.page_template, name)
            return f"[{title}]({url})" if url else title
        url = ctx.file_url(resolved)
        if url is not None:
            if resolved.suffix.lower() in IMAGE_SUFFIXES and not desc:
                return f"![{resolved.name}]({url})"
            return f"[{desc or ctx.repo_rel(resolved)}]({url})"
        return desc or path_part
    if re.match(r"[a-z][\w+.-]*:", target) and not target.startswith(
        ("id:", "elisp:", "shell:", "help:")
    ):
        return f"[{desc or target}]({target})"
    return desc or target.lstrip("#*")


def org_inline(text: str, ctx: Context) -> str:
    shelf = Shelf()

    def verbatim(m: re.Match[str]) -> str:
        return shelf.put(code_span(m.group("body")))

    text = re.sub(
        r"src_[\w-]+(?:\[[^\]]*\])?\{([^}]*)\}",
        lambda m: shelf.put(code_span(m.group(1))),
        text,
    )
    for pattern in VERBATIM:
        text = pattern.sub(verbatim, text)

    def snippet(m: re.Match[str]) -> str:
        backend, body = m.group(1).lower(), m.group(2)
        if backend == "rst":
            return shelf.put(rst_inline(body, ctx))
        if backend in {"md", "markdown", "gfm"}:
            return shelf.put(body)
        return ""

    text = re.sub(r"@@([\w-]+):(.*?)@@", snippet, text, flags=re.S)
    text = re.sub(
        r"\\\((.+?)\\\)", lambda m: shelf.put(f"${m.group(1)}$"), text, flags=re.S
    )
    text = re.sub(
        r"\\\[(.+?)\\\]", lambda m: shelf.put(f"$${m.group(1)}$$"), text, flags=re.S
    )
    text = re.sub(
        r"(?<![\w$\\])\$(?=[^\s$.,;])([^$\n]*?[^\s$.,\\])\$(?![\w$])",
        lambda m: shelf.put(m.group(0)),
        text,
    )
    text = ROLE_RE.sub(
        lambda m: shelf.put(rst_role(m.group("role"), m.group("body"), ctx)), text
    )

    def cite(m: re.Match[str]) -> str:
        keys = [k.strip().lstrip("@") for k in m.group(1).split(";") if k.strip()]
        return shelf.put("[" + "; ".join(f"@{k}" for k in keys) + "]")

    text = re.sub(r"\[cite(?:/[\w/-]+)?:([^\]]+)\]", cite, text)
    text = LINK_RE.sub(
        lambda m: shelf.put(
            org_link(
                m.group("target"),
                m.group("desc") and org_inline(m.group("desc"), ctx),
                ctx,
            )
        ),
        text,
    )
    text = PLAIN_FILE_RE.sub(lambda m: shelf.put(org_link(m.group(0), None, ctx)), text)
    text = re.sub(
        r"\[fn:[\w-]*:([^\]]*)\]",
        lambda m: shelf.put(f" ({org_inline(m.group(1).strip(), ctx)})"),
        text,
    )
    text = re.sub(r"\[fn:([\w-]+)\]", lambda m: shelf.put(f"[^{m.group(1)}]"), text)
    text = re.sub(r"<<<?([^<>]+?)>>>?", r"\1", text)
    text = BOLD.sub(lambda m: shelf.put(f"**{m.group('body')}**"), text)
    text = ITALIC.sub(lambda m: shelf.put(f"*{m.group('body')}*"), text)
    text = STRIKE.sub(lambda m: shelf.put(f"~~{m.group('body')}~~"), text)
    text = re.sub(r"\\nbsp(\{\})?", " ", text)
    text = re.sub(r"\\\\\s*$", "  ", text, flags=re.M)
    return shelf.restore(text)


def org_title(text: str) -> str:
    """The ``#+title:`` of a page, with verbatim and code markup as code spans."""
    for line in text.splitlines():
        m = KEYWORD_RE.match(line)
        if m and m.group(1).lower() == "title":
            title = m.group(2).strip()
            for pattern in VERBATIM:
                title = pattern.sub(lambda v: code_span(v.group("body")), title)
            return title
    return ""


def org_keyword(text: str, key: str) -> str:
    for line in text.splitlines():
        m = KEYWORD_RE.match(line)
        if m and m.group(1).lower() == key:
            return m.group(2).strip()
    return ""


def org_table(lines: list[str], ctx: Context) -> list[str]:
    rows: list[list[str]] = []
    for line in lines:
        s = line.strip()
        if re.fullmatch(r"\|[-+:|]*\|?", s) and "-" in s:
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if any(cells) and all(re.fullmatch(r"<[lrc]?\d*>", c) or not c for c in cells):
            continue
        rows.append([org_inline(c, ctx).replace("|", "\\|") for c in cells])
    if not rows:
        return []
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    out = ["| " + " | ".join(rows[0]) + " |", "|" + "|".join(["---"] * width) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return out


def src_header(args: str) -> tuple[str, dict[str, str]]:
    tokens = args.split()
    lang = ""
    if tokens and not tokens[0].startswith((":", "-", "+")):
        lang = tokens.pop(0)
    params: dict[str, str] = {}
    key = None
    for tok in tokens:
        if tok.startswith(":"):
            key = tok[1:].lower()
            params[key] = ""
        elif key is not None:
            params[key] = (params[key] + " " + tok).strip()
    return SRC_LANGS.get(lang.lower(), lang.lower()), params


def unescape_org_block(lines: list[str]) -> list[str]:
    return [re.sub(r"^(\s*),(\*|#\+|,)", r"\1\2", line) for line in lines]


def org_to_md(text: str, ctx: Context, level: int, page_title: str = "") -> list[str]:
    """Convert one org document body. ``level`` is the Markdown level of ``*``."""
    lines = text.splitlines()
    out: list[str] = []
    para: list[str] = []
    caption = ""
    skip_below: int | None = None

    def flush() -> None:
        if para:
            converted = TERM_RE.sub(bold_term, org_inline("\n".join(para), ctx))
            out.extend(converted.split("\n"))
            para.clear()

    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        hm = HEADING_RE.match(line)
        if hm:
            depth = len(hm.group(1))
            if skip_below is not None and depth > skip_below:
                i += 1
                continue
            skip_below = None
            title = hm.group(2)
            tags = ""
            tm = TAGS_RE.search(title)
            if tm:
                tags, title = tm.group(1), title[: tm.start()]
            if ":noexport:" in tags or title.startswith("COMMENT"):
                skip_below = depth
                i += 1
                continue
            title = re.sub(r"^(TODO|DONE|NEXT|WAIT|HOLD)\s+", "", title)
            title = re.sub(r"^\[#[A-Z]\]\s+", "", title)
            flush()
            md_level = level + depth - 1
            converted = org_inline(title.strip(), ctx)
            out += [
                "",
                f"{'#' * min(md_level, HEADING_CAP)} {converted}"
                if md_level <= HEADING_CAP
                else f"**{converted}**",
                "",
            ]
            i += 1
            continue
        if skip_below is not None:
            i += 1
            continue
        bm = BEGIN_RE.match(line)
        if bm:
            flush()
            pad, kind, args = bm.group(1), bm.group(2).lower(), bm.group(3).strip()
            end_re = re.compile(r"^\s*#\+end_" + re.escape(kind) + r"\b", re.I)
            j = i + 1
            while j < n and not end_re.match(lines[j]):
                j += 1
            body = lines[i + 1 : j]
            i = j + 1
            out += emit_block(kind, args, body, pad, caption, ctx, level, page_title)
            caption = ""
            continue
        km = KEYWORD_RE.match(line)
        if km:
            flush()
            if km.group(1).lower() == "caption":
                caption = km.group(2).strip()
            i += 1
            continue
        if re.match(r"^\s*#(\s|$)", line):
            flush()
            i += 1
            continue
        dm = DRAWER_RE.match(line)
        if dm and dm.group(1).upper() != "END":
            j = i + 1
            while j < n and not re.match(r"^\s*:END:\s*$", lines[j], re.I):
                if HEADING_RE.match(lines[j]):
                    break
                j += 1
            if j < n and re.match(r"^\s*:END:\s*$", lines[j], re.I):
                flush()
                i = j + 1
                continue
        if TABLE_RE.match(line):
            flush()
            j = i
            while j < n and TABLE_RE.match(lines[j]):
                j += 1
            out += ["", *org_table(lines[i:j], ctx), ""]
            if caption:
                out += [f"*{org_inline(caption, ctx)}*", ""]
                caption = ""
            i = j
            continue
        if FIXED_RE.match(line):
            flush()
            j = i
            while j < n and FIXED_RE.match(lines[j]):
                j += 1
            body = [re.sub(r"^\s*:\s?", "", entry) for entry in lines[i:j]]
            out += ["", *fence("text", body), ""]
            i = j
            continue
        lm = LATEX_ENV_RE.match(line)
        if lm:
            flush()
            env = lm.group(1)
            j = i
            while j < n and ("\\end{" + env + "}") not in lines[j]:
                j += 1
            out += ["", "$$", *dedent(lines[i : j + 1]), "$$", ""]
            i = j + 1
            continue
        if re.fullmatch(r"\s*-{5,}\s*", line):
            flush()
            out += ["", "---", ""]
            i += 1
            continue
        fm = FOOTDEF_RE.match(line)
        if fm:
            flush()
            para.append(f"[^{fm.group(1)}]: {fm.group(2)}")
            i += 1
            continue
        if not line.strip():
            flush()
            out.append("")
            i += 1
            continue
        lm2 = BULLET_RE.match(line)
        if lm2:
            flush()
            pad, bullet, rest = lm2.groups()
            bullet = "-" if bullet in {"+", "*"} else bullet.replace(")", ".")
            rest = re.sub(
                r"^\[([ X-])\]\s", lambda m: "[x] " if m.group(1) == "X" else "[ ] ", rest
            )
            dl = re.match(r"(.*?)\s+::(?:\s+(.*)|\s*$)", rest)
            if dl and not rest.startswith("::"):
                rest = f"\x01{dl.group(1)}\x02: " + (dl.group(2) or "")
                para.append(f"{pad}{bullet} {rest}".rstrip())
            else:
                para.append(f"{pad}{bullet} {rest}")
            i += 1
            continue
        para.append(line)
        i += 1
    flush()
    return out


def emit_block(
    kind: str,
    args: str,
    body: list[str],
    pad: str,
    caption: str,
    ctx: Context,
    level: int,
    page_title: str,
) -> list[str]:
    body = dedent(body)
    cap = [f"*{org_inline(caption, ctx)}*", ""] if caption else []
    if kind == "src":
        lang, params = src_header(args)
        if params.get("exports", "").startswith("none"):
            return []
        return ["", *cap, *fence(lang, unescape_org_block(trim_blank(body)), pad), ""]
    if kind == "example":
        return ["", *cap, *fence("text", unescape_org_block(trim_blank(body)), pad), ""]
    if kind == "comment":
        return []
    if kind == "export":
        backend = (args.split() or [""])[0].lower()
        if backend == "rst":
            converted = rst_to_md(body, ctx, page_title)
            return ["", *[f"{pad}{line}" if line else "" for line in converted], ""]
        if backend in {"md", "markdown", "gfm", "ascii", "text"}:
            return ["", *body, ""]
        if backend == "latex" and any("\\begin{" in line for line in body):
            return ["", "$$", *trim_blank(body), "$$", ""]
        return []
    inner = org_to_md("\n".join(body), ctx, level, page_title)
    if kind == "quote":
        return ["", *[f"> {line}" if line else ">" for line in trim_blank(inner)], ""]
    if kind in {"center", "verse"}:
        return ["", *inner, ""]
    return ["", *blockquote(kind.capitalize(), inner), ""]


# ---------------------------------------------------------------------------
# Markdown, rustdoc and other sources


def shift_headings(lines: list[str], top: int) -> list[str]:
    """Shift ATX headings so the shallowest one lands on level ``top``."""
    levels = []
    in_fence = False
    for line in lines:
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        elif not in_fence:
            m = re.match(r"(#{1,6})\s", line)
            if m:
                levels.append(len(m.group(1)))
    if not levels:
        return lines
    delta = top - min(levels)
    out = []
    in_fence = False
    for line in lines:
        shifted = line
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        elif not in_fence:
            m = re.match(r"(#{1,6})(\s.*)", line)
            if m:
                new = len(m.group(1)) + delta
                shifted = (
                    "#" * new + m.group(2)
                    if new <= HEADING_CAP
                    else f"**{m.group(2).strip()}**"
                )
        out.append(shifted)
    return out


def markdown_page(text: str, top: int) -> tuple[str, list[str]]:
    """Return the H1 title (if any) and the body with headings shifted."""
    lines = text.splitlines()
    if lines and lines[0].strip() == "---":
        end = next((k for k in range(1, len(lines)) if lines[k].strip() == "---"), None)
        if end is not None:
            lines = lines[end + 1 :]
    lines = [line for line in lines if not re.fullmatch(r"\s*(<[^>]+>\s*)+", line)]
    title = ""
    for k, line in enumerate(lines):
        if not line.strip():
            continue
        m = re.match(r"#\s+(.*)", line)
        if m:
            title = m.group(1).strip()
            lines = lines[k + 1 :]
        break
    return title, shift_headings(lines, top)


def rustdoc_page(text: str) -> list[str]:
    """Crate-level ``//!`` docs, with intra-doc links reduced to code spans."""
    lines = [
        re.sub(r"^\s*//!\s?", "", line)
        for line in text.splitlines()
        if re.match(r"^\s*//!", line)
    ]
    out = []
    for line, in_fence in zip(lines, fenced(lines), strict=True):
        if in_fence:
            out.append(line)
            continue
        plain_line = re.sub(
            r"\[(`[^`\]]+`)\]\((?:[a-z]+@)?[\w:]+(?:\(\))?\)", r"\1", line
        )
        out.append(re.sub(r"\[(`[^`\]]+`)\](?![(\[])", r"\1", plain_line))
    return out


def frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Parse the flat YAML frontmatter used by SKILL.md files."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    meta: dict[str, str] = {}
    for k in range(1, len(lines)):
        line = lines[k]
        if line.strip() == "---":
            return meta, "\n".join(lines[k + 1 :])
        m = re.match(r"([A-Za-z][\w-]*):\s*(.*)$", line)
        if m and indent_of(line) == 0:
            value = m.group(2).strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                value = value[1:-1]
            meta[m.group(1)] = value
    return {}, text


# ---------------------------------------------------------------------------
# Page discovery


def org_toctrees(text: str) -> list[tuple[str, list[str], bool]]:
    """Return (caption, entries, hidden) for each toctree in rst export blocks."""
    found: list[tuple[str, list[str], bool]] = []
    lines = text.splitlines()
    i, n = 0, len(lines)
    while i < n:
        bm = BEGIN_RE.match(lines[i])
        if (
            bm
            and bm.group(2).lower() == "export"
            and bm.group(3).strip().lower().startswith("rst")
        ):
            j = i + 1
            while j < n and not re.match(r"^\s*#\+end_export\b", lines[j], re.I):
                j += 1
            for node in rst_nodes(dedent(lines[i + 1 : j])):
                if node.kind == "directive" and node.name == "toctree":
                    entries = [entry.strip() for entry in node.lines if entry.strip()]
                    found.append(
                        (
                            node.options.get("caption", ""),
                            entries,
                            "hidden" in node.options,
                        )
                    )
            i = j + 1
        else:
            i += 1
    return found


def index_descriptions(text: str, ctx: Context) -> dict[str, str]:
    """Description-list items (``- Term :: text``) on the index page."""
    found: dict[str, str] = {}
    lines = text.splitlines()
    for k, line in enumerate(lines):
        m = re.match(r"^(\s*)[-+]\s+(.*?)\s+::\s+(.*)$", line)
        if not m:
            continue
        pad = len(m.group(1))
        desc = [m.group(3)]
        for nxt in lines[k + 1 :]:
            if not nxt.strip() or indent_of(nxt) <= pad or BULLET_RE.match(nxt):
                break
            desc.append(nxt.strip())
        term = plain(org_inline(m.group(2), ctx)).lower()
        found[term] = " ".join(org_inline(" ".join(desc), ctx).split())
    return found


def summary_from_markdown(lines: list[str]) -> str:
    """First sentence of the first top-level prose paragraph.

    A sentence ending in a colon introduces a list or a code block and
    says little on its own, so its paragraph is passed over. Indented
    lines continue list items and are never prose.
    """
    paragraphs: list[str] = []
    para: list[str] = []
    for line, in_fence in zip(lines, fenced(lines), strict=True):
        stripped = line.strip()
        prose = (
            not in_fence
            and stripped
            and indent_of(line) == 0
            and not re.match(
                r"(#|>|\||\$\$|!\[|\*\*[^*]+\*\*$|[-*+]\s|\d+[.)]\s|Source: )", stripped
            )
        )
        if prose:
            para.append(stripped)
            continue
        if para:
            paragraphs.append(" ".join(para))
            para = []
    if para:
        paragraphs.append(" ".join(para))
    for text in paragraphs:
        sentence = first_sentence(text)
        if not sentence.endswith(":"):
            return plain(sentence, keep_code=True)
    return ""


class Builder:
    def __init__(self, root: Path, config: dict) -> None:
        self.root = root
        self.cfg = config
        project = config["project"]
        docs = config.get("docs", {})
        self.branch = project.get("branch", "main")
        links = project.get("links", "raw")
        if links == "relative":
            self.raw_base = self.tree_base = ""
        elif links == "raw":
            self.raw_base, self.tree_base = github_bases(project["repo"], self.branch)
        else:
            raise SystemExit(f"project.links must be 'raw' or 'relative', not {links!r}")
        self.docs_root = root / docs.get("root", "docs/orgmode")
        self.index = docs.get("index", "index.org")
        self.page_template = docs.get("page_url")
        self.default_section = docs.get("default_section", "Docs")
        self.section_order = list(docs.get("section_order", []))
        self.skip = set(docs.get("skip", []))
        self.optional = set(docs.get("optional", []))
        self.exclude = set(docs.get("exclude", []))
        self.full_exclude = set(docs.get("full_exclude", []))
        self.sources = docs.get("sources", {})
        self.sections = config.get("sections", {})
        self.descriptions = config.get("descriptions", {})
        self.forbid = [
            re.compile(p) for p in config.get("guard", {}).get("forbid", DEFAULT_FORBID)
        ]
        self.warnings: list[str] = []
        self.pages: list[Page] = []

    def url_for(self, rel: str) -> str:
        return self.raw_base + rel

    def page_url(self, name: str, path: Path | None) -> str:
        if self.page_template is not None:
            return format_page_url(self.page_template, name)
        if path is None:
            return ""
        return self.url_for(path.relative_to(self.root).as_posix())

    def context(self, source: Path) -> Context:
        return Context(
            root=self.root,
            docs_root=self.docs_root,
            source=source,
            raw_base=self.raw_base,
            tree_base=self.tree_base,
            page_url={p.name: p.url for p in self.pages},
            page_title={p.name: p.title for p in self.pages},
            page_template=self.page_template,
        )

    def discover(self) -> None:
        index_path = self.docs_root / self.index
        index_name = Path(self.index).with_suffix("").as_posix()
        seen: set[str] = set()
        order: list[tuple[str, Path | None, str]] = []

        def resolve(entry: str, current: str) -> tuple[str, str] | None:
            title, target = split_title_target(entry)
            if "://" in target or target == "self":
                return None
            if target.startswith("/"):
                name = target.lstrip("/")
            else:
                name = posixpath.normpath(
                    posixpath.join(posixpath.dirname(current), target)
                )
            return name.removesuffix(".org").removesuffix(".rst"), title

        def walk(name: str, path: Path, section: str | None) -> None:
            text = path.read_text(encoding="utf-8")
            for caption, entries, _hidden in org_toctrees(text):
                child_section = section or caption or self.default_section
                for entry in entries:
                    if any(ch in entry for ch in "*?["):
                        matches = sorted(self.docs_root.glob(entry + ".org"))
                        targets = [
                            (m.relative_to(self.docs_root).with_suffix("").as_posix(), "")
                            for m in matches
                        ]
                    else:
                        res = resolve(entry, name)
                        targets = [res] if res else []
                    for child, _title in targets:
                        if child in seen or child in self.skip or child in self.exclude:
                            continue
                        seen.add(child)
                        child_path = self.docs_root / f"{child}.org"
                        if child_path.exists():
                            order.append((child, child_path, child_section))
                            walk(child, child_path, child_section)
                        elif child in self.sources:
                            order.append(
                                (child, self.root / self.sources[child], child_section)
                            )
                        else:
                            self.warnings.append(
                                f"toctree entry {child!r} has no org source"
                            )

        seen.add(index_name)
        first = self.section_order[0] if self.section_order else "Start"
        order.append(
            (index_name, index_path, self.cfg.get("docs", {}).get("index_section", first))
        )
        walk(index_name, index_path, None)
        if self.cfg.get("docs", {}).get("include_orphans", True):
            for path in sorted(self.docs_root.rglob("*.org")):
                name = path.relative_to(self.docs_root).with_suffix("").as_posix()
                if (
                    name not in seen
                    and name not in self.exclude
                    and name not in self.skip
                ):
                    seen.add(name)
                    order.append((name, path, "Optional"))
                    self.warnings.append(
                        f"{name}.org is not in any toctree; listed under Optional"
                    )

        for name, path, section in order:
            text = path.read_text(encoding="utf-8") if path is not None else ""
            doc_title = ""
            if path is not None and path.suffix == ".org":
                doc_title = org_title(text)
            elif path is not None and path.suffix == ".md":
                doc_title = markdown_page(text, 3)[0]
            title = self.cfg.get("titles", {}).get(name, doc_title or name)
            placed = self.sections.get(name, section)
            optional = name in self.optional or placed == "Optional"
            self.pages.append(
                Page(
                    name=name,
                    path=path,
                    title=title,
                    section="Optional" if optional else placed,
                    url=self.page_url(name, path),
                    optional=optional,
                    text=text,
                    doc_title=doc_title or title,
                )
            )

    def convert(self, path: Path, text: str, top: int, title: str) -> list[str]:
        """Markdown body of one source file, its shallowest heading at ``top``.

        ``title`` is the page's own title; a repeat of it at the top of
        the body is dropped.
        """
        ctx = self.context(path)
        suffix = path.suffix
        if suffix == ".org":
            lines = org_to_md(text, ctx, top, title)
        elif suffix == ".md":
            lines = rebase_links(markdown_page(text, top)[1], ctx)
        elif suffix == ".rs":
            lines = shift_headings(rustdoc_page(text), top)
        elif suffix == ".rst":
            lines = rst_to_md(text.splitlines(), ctx, title)
        else:
            lines = text.splitlines()
        return drop_empty_headings(lines)

    def page_body(self, page: Page, top: int) -> list[str]:
        if page.path is None:
            return []
        return self.convert(page.path, page.text, top + 1, page.doc_title)

    def describe(self) -> None:
        index = self.pages[0] if self.pages else None
        from_index: dict[str, str] = {}
        if (
            index is not None
            and index.path is not None
            and self.cfg.get("docs", {}).get("descriptions_from_index", True)
        ):
            from_index = index_descriptions(index.text, self.context(index.path))
        for page in self.pages:
            desc = self.descriptions.get(page.name, "")
            if not desc:
                desc = from_index.get(page.title.lower(), "") or from_index.get(
                    page.name.lower(), ""
                )
            if not desc and page.path is not None and page.path.suffix == ".org":
                desc = org_keyword(page.text, "description")
            if not desc:
                desc = summary_from_markdown(self.page_body(page, 2))
            page.description = " ".join(desc.split())

    def extras(self) -> list[dict]:
        return list(self.cfg.get("links", []))

    def outputs(self) -> dict[str, str]:
        """Files to write, by kind; an empty name turns that output off."""
        out = self.cfg.get("output", {})
        names = {
            "index": out.get("index", "llms.txt"),
            "full": out.get("full", "llms-full.txt"),
        }
        return {kind: rel for kind, rel in names.items() if rel}

    def map_check(self) -> tuple[list[str], list[str]]:
        """Broken links in the hand-written map, and pages it does not link.

        Call after build(). A link counts when it starts with the site
        address; ``map.generated`` lists page prefixes, such as an API
        reference, that the build writes and the Org tree does not hold.
        """
        conf = self.cfg.get("map")
        if not conf:
            return [], []
        site = conf.get("site") or (self.page_template or "").split("{page}", 1)[0]
        if not site:
            raise SystemExit("map.site is not set and docs.page_url has no {page}")
        generated = tuple(conf.get("generated", []))
        known = {p.name for p in self.pages} | set(self.sources)
        linked: set[str] = set()
        broken = []
        rel = conf["path"]
        text = (self.root / rel).read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), 1):
            for m in MD_LINK_RE.finditer(line):
                url = m.group("target")
                if not url.startswith(site):
                    continue
                name = page_name(url[len(site) :])
                if name.startswith(generated):
                    continue
                linked.add(name)
                exists = (self.docs_root / f"{name}.org").exists()
                if name not in known and not exists:
                    broken.append(
                        f"{rel}:{number}: {url} has no page under {self.docs_root.name}"
                    )
        left_out = [p.name for p in self.pages if p.name not in linked]
        return broken, left_out

    def missing_links(self) -> list[str]:
        """[[links]] paths that are neither in the repository nor written by this script."""
        generated = set(self.outputs().values())
        return [
            item["path"]
            for item in self.extras()
            if "path" in item
            and item["path"] not in generated
            and not (self.root / item["path"]).exists()
        ]

    def extra_url(self, item: dict) -> str:
        if "url" in item:
            return item["url"]
        return self.url_for(item["path"])

    def skills(self) -> list[tuple[str, str, str, Path]]:
        conf = self.cfg.get("skills", {})
        directory = self.root / conf.get("dir", "skills")
        found = []
        for skill in sorted(directory.glob("*/SKILL.md")):
            meta, body = frontmatter(skill.read_text(encoding="utf-8"))
            name = meta.get("name", skill.parent.name)
            found.append((name, meta.get("description", ""), body, skill))
        return found

    # -- rendering ---------------------------------------------------------

    def header(self) -> list[str]:
        project = self.cfg["project"]
        out = [
            f"# {project['name']}",
            "",
            f"> {' '.join(project['summary'].split())}",
            "",
        ]
        for note in project.get("notes", []):
            out += [note.strip(), ""]
        return out

    def llms_txt(self) -> str:
        sections: dict[str, list[str]] = {}

        def add(section: str, line: str) -> None:
            sections.setdefault(section, []).append(line)

        for page in self.pages:
            line = f"- [{page.title}]({page.url})"
            add(page.section, f"{line}: {page.description}" if page.description else line)
        for item in self.extras():
            if item.get("index", True) is False:
                continue
            line = f"- [{item['title']}]({self.extra_url(item)})"
            desc = item.get("description", "")
            add(item.get("section", "Docs"), f"{line}: {desc}" if desc else line)
        conf = self.cfg.get("skills", {})
        for name, desc, _body, path in self.skills():
            rel = path.relative_to(self.root).as_posix()
            line = f"- [{name}]({self.url_for(rel)})"
            add(conf.get("section", "Skills"), f"{line}: {desc}" if desc else line)
        out = self.header()
        names = [s for s in self.section_order if s in sections and s != "Optional"]
        names += [s for s in sections if s not in names and s != "Optional"]
        if "Optional" in sections:
            names.append("Optional")
        for section in names:
            out += [f"## {section}", "", *sections[section], ""]
        return tidy(out)

    def llms_full(self) -> str:
        project = self.cfg["project"]
        out = self.header()
        note = project.get("full_note")
        if note:
            out += [note.strip(), ""]
        kept = [p for p in self.pages if p.name not in self.full_exclude]
        ordered = [p for p in kept if not p.optional] + [p for p in kept if p.optional]
        extras = [item for item in self.extras() if item.get("full")]
        early = [item for item in extras if item.get("order", "after") == "before"]
        late = [item for item in extras if item.get("order", "after") != "before"]
        for item in early:
            out += self.render_extra(item)
        for page in ordered:
            out += [
                f"## {page.title}",
                "",
                f"Source: {page.url}",
                "",
                *self.page_body(page, 2),
                "",
            ]
        for item in late:
            out += self.render_extra(item)
        conf = self.cfg.get("skills", {})
        if conf.get("full", True):
            for name, desc, body, path in self.skills():
                rel = path.relative_to(self.root).as_posix()
                lines = self.convert(path, body, 3, name)
                out += [f"## Skill: {name}", "", f"Source: {self.url_for(rel)}", ""]
                if desc:
                    out += [desc, ""]
                out += [*lines, ""]
        return tidy(out)

    def render_extra(self, item: dict) -> list[str]:
        path = self.root / item["path"]
        title = item.get("full_title", item["title"])
        lines = self.convert(path, path.read_text(encoding="utf-8"), 3, title)
        return [f"## {title}", "", f"Source: {self.extra_url(item)}", "", *lines, ""]

    def build(self) -> dict[str, str]:
        self.discover()
        self.describe()
        render = {"index": self.llms_txt, "full": self.llms_full}
        return {rel: render[kind]() for kind, rel in self.outputs().items()}

    def guard(self, outputs: dict[str, str]) -> list[str]:
        """Lines of the generated files that match a forbidden pattern."""
        problems = []
        for rel, text in outputs.items():
            source = ""
            for number, line in enumerate(text.splitlines(), 1):
                if line.startswith("Source: "):
                    source = line.removeprefix("Source: ")
                for pattern in self.forbid:
                    if pattern.search(line):
                        where = f" (from {source})" if source else ""
                        problems.append(f"{rel}:{number}{where}: {line.strip()}")
        return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--check", action="store_true", help="exit 1 if a generated file is stale"
    )
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--verbose", action="store_true", help="print discovery warnings")
    args = parser.parse_args(argv)

    builder = Builder(args.root.resolve(), load_config(args.config))
    missing = builder.missing_links()
    if missing:
        for rel in missing:
            print(
                f"{args.config.name} links to {rel}, which does not exist",
                file=sys.stderr,
            )
        return 2
    outputs = builder.build()
    if args.verbose:
        for warning in builder.warnings:
            print(f"warning: {warning}", file=sys.stderr)
    broken, left_out = builder.map_check()
    for name in left_out:
        print(f"note: {builder.cfg['map']['path']} does not link {name}", file=sys.stderr)
    if broken:
        for problem in broken:
            print(f"broken map link: {problem}", file=sys.stderr)
        return 2
    problems = builder.guard(outputs)
    if problems:
        for problem in problems:
            print(f"machine-local path: {problem}", file=sys.stderr)
        print(
            "Fix the source page, or list it under docs.full_exclude in llms.toml.",
            file=sys.stderr,
        )
        return 2
    stale = []
    for rel, text in outputs.items():
        path = builder.root / rel
        current = path.read_text(encoding="utf-8") if path.exists() else None
        if current == text:
            continue
        if args.check:
            stale.append(rel)
        else:
            path.write_text(text, encoding="utf-8")
            print(f"wrote {rel}")
    if stale:
        print(
            "stale: "
            + ", ".join(stale)
            + ". Regenerate with: python scripts/llms/build_llms.py",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
