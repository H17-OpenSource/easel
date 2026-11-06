"""No code under src/ imports an AGPL screening or detector package (nudenet, ultralytics).

They stay outside and optional: the screens run in their own venv through a subprocess with JSON over
stdin/stdout, and the detector lives inside the user's ComfyUI. The check reads the syntax tree, so `import x`,
`from x import y`, `importlib.import_module("x")` and `__import__("x")` are all caught, while a mention in a
docstring or a subprocess argument is not. The planted import in the self-test is assembled at run time.
"""

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BANNED = ("nudenet", "ultralytics")


def banned_imports(path: Path) -> list[str]:
    """`<line>: <module>` for every import of a banned package in one Python file."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    hits = []
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names = [node.module]
        elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
            func = node.func
            called = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if called in ("import_module", "__import__") and isinstance(node.args[0].value, str):
                names = [node.args[0].value]
        hits += [f"{node.lineno}: {name}" for name in names if name.split(".")[0] in BANNED]
    return hits


def test_no_agpl_import_under_src():
    found = {p.relative_to(REPO).as_posix(): banned_imports(p) for p in sorted((REPO / "src").rglob("*.py"))}
    assert {name: hits for name, hits in found.items() if hits} == {}


def test_the_check_finds_planted_imports(tmp_path):
    first, second = "nude" + "net", "ultra" + "lytics"
    planted = tmp_path / "planted.py"
    planted.write_text(
        f"import {first}\n"
        f"from {second}.models import YOLO\n"
        f"import importlib\nimportlib.import_module('{first}')\n"
        f"__import__('{second}')\n"
        f"'''a docstring that names {first} is fine'''\n"
        f"import json\n",
        encoding="utf-8",
    )
    assert banned_imports(planted) == [f"1: {first}", f"2: {second}.models", f"4: {first}", f"5: {second}"]
