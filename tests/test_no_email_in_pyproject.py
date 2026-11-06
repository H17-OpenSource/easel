"""pyproject.toml names no e-mail address: no `email` key for authors or maintainers, no address anywhere in it.

The repositories publish under a placeholder identity ("<pkg> contributors"); an address in the packaging metadata
would travel with every wheel. The address pattern is assembled at run time so this file holds none.
"""

import re
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
AT = "@"
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+" + AT + r"[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def email_problems(text: str) -> list[str]:
    """What is wrong with a pyproject text: an `email` key for a person, or an address anywhere."""
    problems = []
    project = tomllib.loads(text).get("project", {})
    for key in ("authors", "maintainers"):
        for person in project.get(key, []):
            if "email" in person:
                problems.append(f"project.{key} has an email key")
    if EMAIL.search(text):
        problems.append("an e-mail address appears in the text")
    return problems


def test_pyproject_has_no_email():
    assert email_problems((REPO / "pyproject.toml").read_text(encoding="utf-8")) == []


def test_the_authors_are_named_without_a_person():
    project = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["authors"], "authors is set"
    assert all(set(person) == {"name"} for person in project["authors"])


def test_the_check_finds_a_planted_address():
    address = "demo" + AT + "example" + ".org"
    keyed = f'[project]\nname = "demo"\nauthors = [{{name = "demo", email = "{address}"}}]\n'
    assert email_problems(keyed) == ["project.authors has an email key", "an e-mail address appears in the text"]
    commented = f'[project]\nname = "demo"\n# reach us at {address}\n'
    assert email_problems(commented) == ["an e-mail address appears in the text"]
