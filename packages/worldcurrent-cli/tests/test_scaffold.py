import importlib
import sys

import pytest

from worldcurrent.cli.scaffold import TEMPLATES, check_plugin, new_plugin


@pytest.mark.parametrize("group", sorted(TEMPLATES))
def test_every_template_passes_its_own_conformance_check(group, tmp_path, monkeypatch):
    if group in ("policy", "agent"):                       # these templates drive MuJoCo planar reach
        pytest.importorskip("worldcurrent.envs.mujoco", reason="worldcurrent-bench's MuJoCo environments")
    root = new_plugin(group, f"demo-{group.replace('_', '-')}", tmp_path)
    monkeypatch.syspath_prepend(str(root / "src"))
    module = next((root / "src").iterdir()).name
    plugin = importlib.import_module(f"{module}.plugin")
    cls = next(v for v in vars(plugin).values() if isinstance(v, type) and v.__module__ == plugin.__name__)
    rep = check_plugin(group, f"{module}.plugin:{cls.__name__}")
    assert rep.ok, rep.summary()
    assert f'"worldcurrent.{TEMPLATES[group][0]}"' in (root / "pyproject.toml").read_text()
    sys.modules.pop(f"{module}.plugin", None)
