"""`easel init <dir>`: the layout it writes, the refusals, --force-existing, and a home the other verbs can use."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from easel import cli
from easel import init as init_mod
from easel._common import config

FOLDERS = ("registry", "runs", "recipes", "journal")


def run(argv: list[str], capsys) -> tuple[int, list[str], list[str]]:
    """Run `easel <argv>` in-process: (exit code, stdout lines, stderr lines without the frozen-clock line)."""
    capsys.readouterr()
    try:
        code = cli.main(argv)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 2
    captured = capsys.readouterr()
    err = [line for line in captured.err.splitlines() if not line.startswith("clock frozen")]
    return code, captured.out.splitlines(), err


@pytest.fixture
def place(tmp_path: Path) -> Path:
    """A scratch folder outside every repository checkout (skips on a machine whose temp folder sits in one)."""
    if init_mod.checkout_above(tmp_path.resolve()) is not None:
        pytest.skip("the temp folder lies inside a repository checkout here")
    return tmp_path


def test_init_writes_the_layout(place, frozen_clock, monkeypatch, capsys):
    monkeypatch.chdir(place)
    code, out, err = run(["init", "home"], capsys)
    assert code == 0, out + err
    home = place / "home"
    assert (home / ".gitignore").read_text(encoding="utf-8") == "*\n!README.md\n"
    for name in FOLDERS:
        assert (home / name).is_dir(), name
    assert out[0] == "easel home: home"
    assert out[1] == "created: .gitignore, README.md, registry/, runs/, recipes/, journal/, easel.toml"
    assert out[-1] == "next: easel doctor --workspace home"
    assert not any(str(place) in line for line in out + err), "the output names an absolute path"

    text = (home / "easel.toml").read_text(encoding="utf-8")
    first = text.splitlines()[0]
    assert first.startswith("# easel.toml: this instance's configuration, written by `easel init`")
    assert "2030-01-01" in first
    example = init_mod.EXAMPLE.read_text(encoding="utf-8")
    assert tomllib.loads(text) == tomllib.loads(example), "the instance file holds the example's values"
    readme = (home / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("# easel home\n")
    assert "EASEL_HOME" in readme and "2030-01-01" in readme
    for name in FOLDERS:
        assert f"| `{name}/` |" in readme


def test_the_new_home_loads_and_config_set_keeps_its_comments(place, frozen_clock, monkeypatch, capsys):
    monkeypatch.chdir(place)
    assert run(["init", "home"], capsys)[0] == 0
    cfg = config.load("easel", place / "home", strict=False)
    assert cfg.root == (place / "home").resolve()
    assert {key for _var, key in cfg.unset} == {"paths.models_root", "privacy.wordlist"}
    code, out, err = run(["config", "set", "paths.models_root", "models", "--workspace", "home"], capsys)
    assert code == 0, out + err
    assert out == ['set paths.models_root = "models" in easel.toml (found by --workspace)']
    text = (place / "home" / "easel.toml").read_text(encoding="utf-8")
    assert text.startswith("# easel.toml: this instance's configuration")


def test_tilde_is_expanded_and_printed_as_typed(place, frozen_clock, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(place))
    monkeypatch.setenv("USERPROFILE", str(place))
    code, out, _ = run(["init", "~/easel-home"], capsys)
    assert code == 0
    assert (place / "easel-home" / "easel.toml").is_file()
    assert out[-1] == "next: easel doctor --workspace ~/easel-home"


@pytest.mark.parametrize("marker", [".git", "pyproject"])
def test_refuses_a_target_inside_a_checkout(place, frozen_clock, monkeypatch, capsys, marker):
    checkout = place / "checkout"
    checkout.mkdir()
    if marker == ".git":
        (checkout / ".git").mkdir()
    else:
        (checkout / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")
        (checkout / "src").mkdir()
    monkeypatch.chdir(place)
    code, out, err = run(["init", "checkout/home"], capsys)
    assert code == 2
    assert out == []
    assert err[0].startswith("refused: checkout/home lies inside a repository checkout (checkout/")
    assert not (checkout / "home").exists()


def test_refuses_a_file(place, frozen_clock, monkeypatch, capsys):
    (place / "afile").write_text("x", encoding="utf-8")
    monkeypatch.chdir(place)
    code, _, err = run(["init", "afile"], capsys)
    assert code == 2
    assert err[0] == "refused: afile is a file, not a folder; nothing was written"


def test_refuses_a_non_empty_folder_without_force(place, frozen_clock, monkeypatch, capsys):
    folder = place / "busy"
    folder.mkdir()
    (folder / "notes.txt").write_text("mine", encoding="utf-8")
    monkeypatch.chdir(place)
    code, _, err = run(["init", "busy"], capsys)
    assert code == 2
    assert err[0] == "refused: busy is not empty (1 entry) and holds no easel.toml; nothing was written"
    assert "--force-existing" in err[1]
    assert sorted(p.name for p in folder.iterdir()) == ["notes.txt"]


def test_force_existing_adds_the_missing_pieces_and_keeps_files(place, frozen_clock, monkeypatch, capsys):
    folder = place / "busy"
    (folder / "runs").mkdir(parents=True)
    (folder / "README.md").write_text("my own readme\n", encoding="utf-8")
    (folder / "easel.toml").write_text("", encoding="utf-8")  # an empty config counts as missing
    monkeypatch.chdir(place)
    code, out, err = run(["init", "busy", "--force-existing"], capsys)
    assert code == 0, out + err
    assert (folder / "README.md").read_text(encoding="utf-8") == "my own readme\n"
    assert "kept (already there, not changed): README.md, runs/" in out
    assert "created: .gitignore, registry/, recipes/, journal/, easel.toml" in out
    assert tomllib.loads((folder / "easel.toml").read_text(encoding="utf-8"))["schema"] == "easel/config/1"


def test_force_existing_warns_about_a_gitignore_without_star(place, frozen_clock, monkeypatch, capsys):
    folder = place / "busy"
    folder.mkdir()
    (folder / ".gitignore").write_text("*.png\n", encoding="utf-8")
    monkeypatch.chdir(place)
    code, out, _ = run(["init", "busy", "--force-existing"], capsys)
    assert code == 0
    assert any(line.startswith("warn: the existing .gitignore has no `*` line") for line in out)
    assert (folder / ".gitignore").read_text(encoding="utf-8") == "*.png\n"


def test_an_existing_home_is_never_overwritten(place, frozen_clock, monkeypatch, capsys):
    monkeypatch.chdir(place)
    assert run(["init", "home"], capsys)[0] == 0
    before = (place / "home" / "easel.toml").read_bytes()
    for extra in ([], ["--force-existing"]):
        code, out, err = run(["init", "home", *extra], capsys)
        assert code == 2
        assert err == ["refused: home is already an easel home (it holds easel.toml); nothing was changed",
                       "use it: easel doctor --workspace home, or set EASEL_HOME to it"]
    assert (place / "home" / "easel.toml").read_bytes() == before


def test_a_missing_example_is_a_refusal(place, frozen_clock, monkeypatch, capsys):
    monkeypatch.setattr(init_mod, "EXAMPLE", place / "nowhere" / "easel.example.toml")
    monkeypatch.chdir(place)
    code, _, err = run(["init", "home"], capsys)
    assert code == 2
    assert err[0].startswith("refused: easel.example.toml is not next to this install")
    assert not (place / "home").exists()


def test_a_path_with_a_space_is_quoted_in_the_next_line(place, frozen_clock, monkeypatch, capsys):
    monkeypatch.chdir(place)
    code, out, _ = run(["init", "my home"], capsys)
    assert code == 0
    assert out[-1] == 'next: easel doctor --workspace "my home"'
