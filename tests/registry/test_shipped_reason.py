"""`shipped.reason`: an exception as one line for a person, never with the absolute path its text carries."""

from __future__ import annotations

import pytest

from easel.registry.shipped import reason

SEP = chr(92)  # a backslash, built at run time so this file holds no drive-path literal


def _drive_path(*parts: str) -> str:
    return "C:" + SEP + SEP.join(parts)


@pytest.mark.parametrize("message, expected", [
    (lambda p: f"a record is a mapping at the top level: {p}", "a record is a mapping at the top level: <path>"),
    (lambda p: f"{p}: schema 'x/1' is not a easel/resource/<n> record",
     "<path>: schema 'x/1' is not a easel/resource/<n> record"),
    (lambda p: f"cannot read '{p}' twice", "cannot read '<path>' twice"),
])
def test_absolute_paths_become_a_placeholder(message, expected):
    path = _drive_path("Users", "some one", "home", "registry", "resources", "a.yaml")
    assert reason(ValueError(message(path))) == expected


def test_an_oserror_gives_its_reason_without_the_file_name(tmp_path):
    missing = tmp_path / "not-here.json"
    with pytest.raises(OSError) as caught:
        missing.read_text(encoding="utf-8")
    text = reason(caught.value)
    assert text and str(tmp_path) not in text and "not-here" not in text


def test_only_the_first_line_and_a_name_for_an_empty_message():
    assert reason(ValueError("first line\nsecond line")) == "first line"
    assert reason(KeyError()) == "KeyError"
    posix = "/" + "/".join(("home", "someone", "easel-home", "runs", "x.json"))
    assert reason(ValueError(f"bad file {posix}")) == "bad file <path>"
