import json

import numpy as np
from PIL import Image

from worldcurrent.datasources.video.sources import FrameFolderSource


def test_frame_folders_become_image_only_episodes(tmp_path):
    for e, ok in (("ep-a", True), ("ep-b", None)):
        d = tmp_path / e
        d.mkdir()
        for k in range(6):
            Image.fromarray(np.full((32, 48, 3), 40 * k, np.uint8)).save(d / f"{k:03d}.png")
        meta = {"instruction": "Open the drawer", "camera": "wrist"}
        if ok is not None:
            meta["success"] = ok
        (d / "episode.json").write_text(json.dumps(meta))
    eps = list(FrameFolderSource(str(tmp_path), every=2, size=16).episodes())
    assert [e.episode_id for e in eps] == ["ep-a", "ep-b"]
    a = eps[0]
    assert a.task_id == "video/open_the_drawer" and a.success is True and eps[1].success is None
    assert len(a.frames) == 3 and a.frames[0].images["wrist"].shape == (16, 16, 3)
    assert a.frames[0].action is None and a.frames[0].state == {}
