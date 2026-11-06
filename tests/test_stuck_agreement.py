"""The STUCK template has one owner (`_common/journal.py`); docs/STUCK.md, the pitfall issue form and the formats
page are generated from it and the schemas. `tools/gen_formats_doc.py --check` proves the files on disk are in
step (CI runs the same command).
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TOOL = REPO / "tools" / "gen_formats_doc.py"


def test_generated_docs_agree_with_the_template_and_schemas():
    if not TOOL.is_file():
        pytest.skip("tools/gen_formats_doc.py not written yet")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in (str(REPO / "src"), os.environ.get("PYTHONPATH")) if p),
           "PYTHONIOENCODING": "utf-8"}
    done = subprocess.run([sys.executable, str(TOOL), "--check"], cwd=REPO, env=env, capture_output=True,
                          text=True, encoding="utf-8", timeout=300)
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.strip().splitlines()[-1].startswith("in step: "), done.stdout
