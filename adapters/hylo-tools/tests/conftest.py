"""Fixtures for the tool tests (helpers in tools_testkit.py)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))     # tests import tools_testkit (pytest runs with importlib mode)

import tools_testkit as tk  # noqa: E402


@pytest.fixture(scope="session")
def libero_frame(tmp_path_factory) -> Path:
    """The first frame (step 0) of the LIBERO episode strip in the docs, 387 x 387."""
    from PIL import Image
    if not tk.STRIP.exists():
        pytest.skip(f"missing {tk.STRIP}")
    out = tmp_path_factory.mktemp("libero") / "libero0.png"
    Image.open(tk.STRIP).convert("RGB").crop((36, 135, 423, 522)).save(out)
    return out


@pytest.fixture(scope="session")
def scene_frame(tmp_path_factory) -> Path:
    """A rendered tabletop scene: scene.png, scene.npz (depth, K, T_world_cam, seg, names), scene.xml."""
    pytest.importorskip("mujoco")
    out = tmp_path_factory.mktemp("scene")
    proc = subprocess.run([sys.executable, str(tk.HERE / "scene_frame.py"), str(out)], env=tk.gl_env(),
                          capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.skip(f"cannot render the test scene (needs MuJoCo offscreen GL): {proc.stderr[-400:]}")
    return out
