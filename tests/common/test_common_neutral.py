"""`_common` is one set of files in both packages, so it never names either tool in what it prints or shows.

The rule (src/<pkg>/_common/__init__.py): the tool name is always a parameter. A string literal that names a tool
would tell a user of the other tool to run a command it does not have, and a docstring example that imports one
package or loads one tool's config breaks when copied in the other. The tool names come from
tools/sync_common.py, the same file in both repositories, so this test reads the same in both after the copy.
"""

import ast
import importlib.util
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PKG = next(d.name for d in sorted((REPO / "src").iterdir()) if (d / "_common").is_dir())
COMMON = REPO / "src" / PKG / "_common"


def tool_names() -> tuple[str, ...]:
    spec = importlib.util.spec_from_file_location("sync_common_for_test", REPO / "tools" / "sync_common.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return tuple(sorted(module.OTHER))


NAMES = tool_names()
WORD = re.compile(r"\b(?:" + "|".join(NAMES) + r")\b", re.IGNORECASE)
# What a docstring example must not do: import one package, load one tool, prefix a file with one tool, or quote
# one tool's command line.
EXAMPLE = re.compile(r"(?:from|import)\s+(?:{n})\b|load\(\s*[\"'](?:{n})[\"']|[\"'](?:{n})/|`(?:{n}) "
                     .format(n="|".join(NAMES)))


def sources() -> list[Path]:
    return sorted(COMMON.glob("*.py"))


def docstring_nodes(tree: ast.AST) -> set[int]:
    """The ids of the string constants that are docstrings (first statement of a module, class or function)."""
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                found.add(id(first.value))
    return found


def test_the_tool_names_are_known():
    assert PKG in NAMES and len(NAMES) == 2


@pytest.mark.parametrize("path", sources(), ids=lambda p: p.name)
def test_no_string_literal_names_a_tool(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docs = docstring_nodes(tree)
    bad = [(node.lineno, node.value) for node in ast.walk(tree)
           if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs
           and WORD.search(node.value)]
    assert bad == [], f"{path.name}: build the name from the tool parameter instead"


@pytest.mark.parametrize("path", sources(), ids=lambda p: p.name)
def test_docstring_examples_are_tool_neutral(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    ids = docstring_nodes(tree)
    docs = [node for node in ast.walk(tree) if isinstance(node, ast.Constant) and id(node) in ids]
    bad = [(node.lineno, m.group(0)) for node in docs for m in EXAMPLE.finditer(node.value)]
    assert bad == [], f"{path.name}: write <pkg>, <tool> or load(tool) in examples"
