"""_common.config: discovery, overrides, [env_aliases], ${VAR}, the two always-on rules, config show and set.

Every test uses its own minimal schema under tmp_path (or none), never the repo's schemas/config.schema.json.
"""

import json
from pathlib import Path

import pytest

from easel import cli
from easel._common import config

PKG = "easel"

SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "schema": {"type": "string"},
        "instance": {"type": "object", "additionalProperties": False,
                     "properties": {"name": {"type": "string"}, "adopted": {"type": "boolean", "default": False}}},
        "paths": {"type": "object", "additionalProperties": False,
                  "properties": {"root": {"type": "string", "default": "."}, "models_root": {"type": "string"},
                                 "runs": {"type": "string", "default": "runs"},
                                 "journal": {"type": "string", "default": "journal"}}},
        "comfyui": {"type": "object", "additionalProperties": False,
                    "properties": {"url": {"type": "string", "default": "http://127.0.0.1:8188"},
                                   "start_args": {"type": "array", "items": {"type": "string"}}}},
        "limits": {"type": "object", "additionalProperties": False,
                   "properties": {"disk_floor_gb": {"type": "number", "default": 100},
                                  "max_concurrent_gpu_jobs": {"type": "integer"}}},
        "train": {"type": "object", "additionalProperties": False,
                  "properties": {"venv_dir": {"type": "string"}, "python": {"type": "string"}}},
        "env_aliases": {"type": "object", "additionalProperties": {"type": "string"}},
    },
}

TOML = """schema = "easel/config/1"

[instance]
name = "home"   # a comment that config set must keep

[paths]
root = "."
models_root = "${DEMO_MODELS_ROOT}"
journal = "journal"

[comfyui]
url = "http://127.0.0.1:8188"

[limits]
disk_floor_gb = 100

[env_aliases]
"DEMO_OLD_MODELS" = "paths.models_root"
"""


@pytest.fixture
def schema_file(tmp_path) -> Path:
    path = tmp_path / "config.schema.json"
    path.write_text(json.dumps(SCHEMA), encoding="utf-8")
    return path


@pytest.fixture
def instance(tmp_path) -> Path:
    home = tmp_path / "easel-home"
    home.mkdir()
    (home / "easel.toml").write_bytes(TOML.encode("utf-8"))
    return home


@pytest.fixture
def repo_schema(monkeypatch, schema_file):
    """Point the CLI at the test schema (the CLI always uses the repo's schema path)."""
    monkeypatch.setattr(config, "default_schema_path", lambda: schema_file)
    return schema_file


def run(argv, capsys):
    code = cli.main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))
    return path


# --- discovery ------------------------------------------------------------------------------------------------

def test_discovery_order(tmp_path):
    named = {k: write(tmp_path / k / "easel.toml", f'[instance]\nname = "{k}"\n') for k in ("ws", "home", "up", "user")}
    deep = tmp_path / "up" / "a" / "b"
    deep.mkdir(parents=True)
    env = {"EASEL_HOME": str(tmp_path / "home")}
    user = tmp_path / "user"

    def name(**kw):
        cfg = config.load(PKG, schema_file=None, user_dir=user, **kw)
        return cfg.values["instance"]["name"], cfg.found_by

    assert name(workspace=tmp_path / "ws", env=env, cwd=deep) == ("ws", "--workspace")
    assert name(workspace=named["ws"], env=env, cwd=deep) == ("ws", "--workspace")  # a file path works too
    assert name(env=env, cwd=deep) == ("home", "EASEL_HOME")
    assert name(env={}, cwd=deep) == ("up", "the current folder or a parent")
    assert name(env={}, cwd=tmp_path) == ("user", "the user config folder")
    nothing = config.load(PKG, schema_file=None, env={}, cwd=tmp_path, user_dir=tmp_path / "none")
    assert nothing.file is None and nothing.found_by == "nothing" and "no easel.toml found" in nothing.not_found_line()


def test_an_explicit_place_without_a_file_is_exit_2(tmp_path):
    with pytest.raises(config.ConfigError) as caught:
        config.load(PKG, workspace=tmp_path / "missing", schema_file=None, env={})
    assert caught.value.code == 2
    with pytest.raises(config.ConfigError) as caught:
        config.load(PKG, schema_file=None, env={"EASEL_HOME": str(tmp_path / "missing")})
    assert caught.value.code == 2 and "EASEL_HOME" in caught.value.lines[0]


def test_the_tool_name_is_a_parameter(tmp_path):
    write(tmp_path / "othertool.toml", '[review]\nport = 8741\n')  # any tool name works: none is built in
    env = {"OTHERTOOL_REVIEW__PORT": "8000", "EASEL_REVIEW__PORT": "1"}
    cfg = config.load("othertool", workspace=tmp_path, schema_file=None, env=env)
    assert cfg.file.name == "othertool.toml"
    assert cfg.get("review.port") == 8000 and cfg.sources["review.port"] == "env OTHERTOOL_REVIEW__PORT"


# --- overrides, defaults, expansion -----------------------------------------------------------------------------

def test_env_overrides_aliases_and_defaults(instance, schema_file, tmp_path):
    models = str(tmp_path / "models")
    env = {"DEMO_MODELS_ROOT": "unused", "DEMO_OLD_MODELS": models, "EASEL_LIMITS__DISK_FLOOR_GB": "50",
           "EASEL_PATHS__RUNS": "123", "EASEL_COMFYUI__START_ARGS": '["--offline"]', "EASEL_HOME": "ignored-no-dunder"}
    cfg = config.load(PKG, workspace=instance, schema_file=schema_file, env=env)
    assert cfg.get("paths.models_root") == models
    assert cfg.sources["paths.models_root"] == "env alias DEMO_OLD_MODELS"
    assert cfg.get("limits.disk_floor_gb") == 50 and cfg.sources["limits.disk_floor_gb"] == "env EASEL_LIMITS__DISK_FLOOR_GB"
    assert cfg.get("paths.runs") == "123"  # the schema says string: the text is kept
    assert cfg.get("comfyui.start_args") == ["--offline"]
    assert cfg.get("instance.adopted") is False and cfg.sources["instance.adopted"] == "default"
    assert cfg.sources["comfyui.url"] == "file"
    assert cfg.path("paths.runs") == instance.resolve() / "123"
    assert cfg.path("paths.models_root") == Path(models)


def test_tool_env_beats_the_alias(instance):
    env = {"DEMO_OLD_MODELS": "from-alias", "EASEL_PATHS__MODELS_ROOT": "from-tool-env"}
    cfg = config.load(PKG, workspace=instance, schema_file=None, env=env)
    assert cfg.get("paths.models_root") == "from-tool-env"


def test_var_expansion_and_escape(tmp_path):
    write(tmp_path / "easel.toml", '[paths]\nmodels_root = "${DEMO_MODELS_ROOT}/sd"\nnote = "$${DEMO_MODELS_ROOT}"\n')
    cfg = config.load(PKG, workspace=tmp_path, schema_file=None, env={"DEMO_MODELS_ROOT": "models"})
    assert cfg.get("paths.models_root") == "models/sd"
    assert cfg.get("paths.note") == "${DEMO_MODELS_ROOT}"


def test_unset_var_is_fail_in_doctor_and_exit_2_elsewhere(instance, schema_file):
    with pytest.raises(config.ConfigError) as caught:
        config.load(PKG, workspace=instance, schema_file=schema_file, env={})
    line = "FAIL: ${DEMO_MODELS_ROOT} unset (paths.models_root)"
    assert caught.value.code == 2 and caught.value.lines == [line]
    lenient = config.load(PKG, workspace=instance, schema_file=schema_file, env={}, strict=False)
    assert lenient.unset_lines() == [line]  # what doctor prints as its FAIL line
    with pytest.raises(config.ConfigError) as caught:
        lenient.get("paths.models_root")
    assert caught.value.lines == [line]
    assert lenient.get("comfyui.url") == "http://127.0.0.1:8188"  # other keys still read


def test_schema_rejects_unknown_keys_and_wrong_types(tmp_path, schema_file):
    write(tmp_path / "easel.toml", '[limits]\ndisk_floor_gb = "lots"\n[nosuch]\nx = 1\n')
    with pytest.raises(config.ConfigError) as caught:
        config.load(PKG, workspace=tmp_path, schema_file=schema_file, env={})
    text = "\n".join(caught.value.lines)
    assert caught.value.code == 2 and "limits.disk_floor_gb" in text and "nosuch" in text


def test_venv_dir_ending_exe_is_rejected_with_or_without_schema(tmp_path, schema_file):
    write(tmp_path / "easel.toml", '[train]\nvenv_dir = "runtime/venv/Scripts/python.exe"\n')
    for schema in (None, schema_file):
        with pytest.raises(config.ConfigError) as caught:
            config.load(PKG, workspace=tmp_path, schema_file=schema, env={})
        assert caught.value.code == 2 and "train.venv_dir" in caught.value.lines[0]


# --- shipped files and absolute paths ---------------------------------------------------------------------------

def drive_path() -> str:
    return "Q" + ":" + "\\" + "models"  # built at run time: no literal absolute path in this file


def posix_path() -> str:
    return "/" + "home" + "/" + "someone" + "/models"


def unc_path() -> str:
    return "\\" * 4 + "server" + "\\" * 2 + "share"  # TOML-escaped; the value is a network share path


def test_absolute_path_detection():
    hits = config.absolute_paths(f"a = '{drive_path()}'\nb = \"{posix_path()}\"\nc = \"runs/x\"\n"
                                 f'd = "http://127.0.0.1:8188"\ne = "{unc_path()}"\n')
    assert [n for n, _ in hits] == [1, 2, 5]


def test_shipped_file_with_an_absolute_path_is_exit_3(tmp_path):
    repo = tmp_path / "checkout"
    (repo / "src" / PKG).mkdir(parents=True)
    write(repo / "pyproject.toml", "[project]\nname = 'demo'\n")
    shipped = write(repo / "examples" / "demo" / "easel.toml", f"[paths]\nmodels_root = '{drive_path()}'\n")
    with pytest.raises(config.ConfigError) as caught:
        config.load(PKG, workspace=shipped.parent, schema_file=None, env={})
    assert caught.value.code == 3 and "REFUSED" in caught.value.lines[0] and "easel.toml:2" in caught.value.lines[0]
    example = write(tmp_path / "loose" / "easel.example.toml", f"[paths]\nmodels_root = '{drive_path()}'\n")
    with pytest.raises(config.ConfigError):
        config.load(PKG, workspace=example, schema_file=None, env={})
    instance = write(tmp_path / "home" / "easel.toml", f"[paths]\nmodels_root = '{drive_path()}'\n")
    assert config.load(PKG, workspace=instance, schema_file=None, env={}).get("paths.models_root") == drive_path()


# --- config set -------------------------------------------------------------------------------------------------

def test_config_set_writes_validated_value_and_keeps_comments(instance, repo_schema, monkeypatch, capsys):
    monkeypatch.setenv("DEMO_MODELS_ROOT", "models")
    code, out, err = run(["config", "set", "limits.disk_floor_gb", "50", "--workspace", str(instance)], capsys)
    assert code == 0, err
    assert out == "set limits.disk_floor_gb = 50 in easel.toml (found by --workspace)\n"
    text = (instance / "easel.toml").read_text(encoding="utf-8")
    assert "disk_floor_gb = 50\n" in text and "# a comment that config set must keep" in text
    assert text.replace("disk_floor_gb = 50", "disk_floor_gb = 100") == TOML  # nothing else changed
    assert (instance / ".lock").exists()  # written under the instance's store lock
    code, _, _ = run(["config", "set", "instance.name", "studio", "--workspace", str(instance)], capsys)
    assert code == 0
    assert 'name = "studio"  # a comment that config set must keep' in (instance / "easel.toml").read_text("utf-8")
    assert config.load(PKG, workspace=instance, schema_file=repo_schema).get("limits.disk_floor_gb") == 50


@pytest.mark.parametrize("key, value", [("limits.nosuch", "1"), ("nosuch.key", "1"), ("limits.disk_floor_gb", "lots"),
                                        ("limits.max_concurrent_gpu_jobs", "1.5"),
                                        ("train.venv_dir", "venv/python.exe")])
def test_config_set_refuses_and_leaves_the_file_unchanged(instance, repo_schema, key, value, capsys):
    before = (instance / "easel.toml").read_bytes()
    code, out, err = run(["config", "set", key, value, "--workspace", str(instance)], capsys)
    assert code == 2 and out == ""
    assert err.startswith(f"refused: {key} = {value}; easel.toml is unchanged")
    assert (instance / "easel.toml").read_bytes() == before


def test_config_set_adds_keys_and_sections(tmp_path):
    write(tmp_path / "easel.toml", "[paths]\nroot = \".\"\n\n[limits]\ndisk_floor_gb = 100\n")
    config.set_value(PKG, "paths.runs", "runs", workspace=tmp_path, schema_file=None, env={})
    config.set_value(PKG, "train.python", drive_path() + "\\python.exe", workspace=tmp_path, schema_file=None, env={})
    text = (tmp_path / "easel.toml").read_text(encoding="utf-8")
    assert '[paths]\nroot = "."\nruns = "runs"\n' in text
    assert text.endswith(f"[train]\npython = '{drive_path()}\\python.exe'\n")  # a literal string: no doubled \
    cfg = config.load(PKG, workspace=tmp_path, schema_file=None, env={})
    assert cfg.get("train.python") == drive_path() + "\\python.exe" and cfg.get("paths.runs") == "runs"


def test_config_set_falls_back_to_a_full_rewrite_for_multiline_values(tmp_path):
    write(tmp_path / "easel.toml", '# my notes\n[comfyui]\nstart_args = [\n  "--offline",\n]\n')
    _, value, kept, _ = config.set_value(PKG, "comfyui.start_args", '["--cpu"]', workspace=tmp_path, schema_file=None,
                                      env={})
    assert value == ["--cpu"] and kept is False
    assert config.load(PKG, workspace=tmp_path, schema_file=None, env={}).get("comfyui.start_args") == ["--cpu"]


def test_edit_toml_text_keeps_crlf_and_the_comment_column():
    text = '[comfyui]\r\nurl = "http://127.0.0.1:8188"   # upstream default\r\n'
    edited = config.edit_toml_text(text, "comfyui.url", "http://127.0.0.1:8000")
    assert edited == '[comfyui]\r\nurl = "http://127.0.0.1:8000"   # upstream default\r\n'
    assert config.edit_toml_text('a = "x#y"  # note\n', "a", "z") == 'a = "z"    # note\n'  # same column


def test_config_set_keeps_a_crlf_file_crlf(tmp_path):
    """A Windows-edited instance file stays CRLF through `config set` (it is read as bytes, never as text)."""
    original = b'[instance]\r\nname = "home"   # kept\r\n\r\n[limits]\r\ndisk_floor_gb = 20\r\n'
    (tmp_path / "easel.toml").write_bytes(original)
    _, _, kept, _ = config.set_value(PKG, "limits.disk_floor_gb", "50", workspace=tmp_path, schema_file=None, env={})
    raw = (tmp_path / "easel.toml").read_bytes()
    assert kept and raw == original.replace(b"disk_floor_gb = 20", b"disk_floor_gb = 50")
    config.set_value(PKG, "paths.runs", "runs", workspace=tmp_path, schema_file=None, env={})  # a new section
    raw = (tmp_path / "easel.toml").read_bytes()
    assert raw.count(b"\n") == raw.count(b"\r\n") and b'[paths]\r\nruns = "runs"\r\n' in raw


def test_config_set_full_rewrite_keeps_crlf_too(tmp_path):
    (tmp_path / "easel.toml").write_bytes(b'# my notes\r\n[comfyui]\r\nstart_args = [\r\n  "--cpu",\r\n]\r\n')
    _, _, kept, _ = config.set_value(PKG, "comfyui.start_args", "[]", workspace=tmp_path, schema_file=None, env={})
    raw = (tmp_path / "easel.toml").read_bytes()
    assert kept is False and raw.count(b"\n") == raw.count(b"\r\n") > 0
    assert config.load(PKG, workspace=tmp_path, schema_file=None, env={}).get("comfyui.start_args") == []


def test_config_set_without_an_instance_is_exit_2(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "nohome"))
    code, _, err = run(["config", "set", "comfyui.url", "x"], capsys)
    assert code == 2 and "no easel.toml found" in err


# --- config show ------------------------------------------------------------------------------------------------

def test_config_show_prints_values_with_sources(instance, repo_schema, monkeypatch, capsys):
    monkeypatch.setenv("DEMO_MODELS_ROOT", "models")
    monkeypatch.setenv("EASEL_COMFYUI__URL", "http://127.0.0.1:8000")
    code, out, err = run(["config", "show", "--workspace", str(instance)], capsys)
    assert code == 0, err
    lines = out.splitlines()
    assert lines[0].startswith("file: ") and "found by --workspace" in lines[0]
    by_key = {line.split(" = ", 1)[0]: line for line in lines[2:]}
    assert by_key["comfyui.url"].endswith("# env EASEL_COMFYUI__URL")
    assert '"http://127.0.0.1:8000"' in by_key["comfyui.url"]
    assert by_key["instance.name"].endswith("# file")
    assert by_key["paths.runs"].endswith("# default")
    assert by_key["paths.models_root"].split("#")[0].strip().endswith('"models"')
    code, out, _ = run(["config", "show", "paths", "--workspace", str(instance)], capsys)
    assert code == 0 and all(line.startswith("paths.") for line in out.splitlines()[2:])
    code, _, err = run(["config", "show", "nosuch", "--workspace", str(instance)], capsys)
    assert code == 2 and "no value at nosuch" in err


def test_config_show_redact_paths_hides_every_absolute_path(instance, repo_schema, monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("DEMO_MODELS_ROOT", str(tmp_path / "elsewhere" / "models"))
    monkeypatch.setenv("EASEL_PATHS__RUNS", str(instance / "runs"))
    code, out, _ = run(["config", "show", "--redact-paths", "--workspace", str(instance)], capsys)
    assert code == 0
    assert str(tmp_path) not in out and tmp_path.as_posix() not in out
    assert out.splitlines()[:2] == ["file: easel.toml (found by --workspace)", "root: <root>"]
    assert 'paths.models_root = "<models_root>"' in out
    assert 'paths.runs = "runs"' in out


def test_config_show_reports_unset_vars_and_exits_2(instance, repo_schema, capsys):
    code, out, err = run(["config", "show", "--workspace", str(instance)], capsys)
    assert code == 2
    assert "FAIL: ${DEMO_MODELS_ROOT} unset (paths.models_root)" in err
    assert "unset: ${DEMO_MODELS_ROOT}" in out
