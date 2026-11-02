import numpy as np
import pytest

from hylo.manip import ColourSegmenter, EnvSegmenter, MarkSegmenter, Table, Tools
from hylo.manip.geometry import apply, pose, rotz
from hylo.manip.perception import fit_box, icp, label_score, ransac_table, remove_outliers

import manip_testkit as tk

RED, BLUE = (0.9, 0.1, 0.1, 1), (0.1, 0.2, 0.9, 1)


@pytest.fixture
def scene(tmp_path, arm_spec):
    objects = {"red_block": ((0.45, 0.05, 0.025), (0.025, 0.018, 0.025), RED, 0.4),
               "blue_cube": ((0.55, -0.15, 0.02), (0.02, 0.02, 0.02), BLUE, -0.2),
               "red_cylinder_stand_in": ((0.35, -0.2, 0.03), (0.02, 0.02, 0.03), RED, 0.0)}
    ex = tk.StubExecutor(tmp_path, objects, arm_spec, hw=(150, 200), noise=0.001)
    return ex, Tools(arm_spec, ex)


def test_ransac_table_with_noise_and_clutter():
    rng = np.random.default_rng(0)
    plane = np.c_[rng.uniform(0, 1, (4000, 2)), 0.72 + rng.normal(0, 0.002, 4000)]
    clutter = np.r_[tk.box_points((0.5, 0.5, 0.77), (0.05, 0.05, 0.05)), rng.uniform([0, 0, 0.3], [1, 1, 1.2], (800, 3))]
    t = ransac_table(np.r_[plane, clutter], rng)
    assert abs(t.height - 0.72) < 0.002 and t.normal[2] > 0.999
    assert np.allclose(t.extent, [[0, 1], [0, 1]], atol=0.03)
    wall = np.c_[np.full(5000, 0.0), rng.uniform(0, 1, (5000, 2))]          # a bigger vertical plane is ignored
    assert abs(ransac_table(np.r_[plane, wall], rng).height - 0.72) < 0.003


@pytest.mark.parametrize("yaw", [0.0, 0.3, -1.2, 1.5])
def test_box_fit_accuracy(yaw):
    rng = np.random.default_rng(1)
    pts = tk.box_points((0.4, -0.1, 0.03), (0.04, 0.025, 0.03), yaw=yaw, spacing=0.004)
    pts = pts[(pts - [0.4, -0.1, 0.03]) @ [np.cos(yaw + 2.0), np.sin(yaw + 2.0), 0] < 0.01]   # one side hidden
    pts = pts + rng.normal(0, 0.001, pts.shape)
    b = fit_box(pts, Table(0.0, np.r_[0, 0, 1.0]))
    assert np.allclose(b.size, [0.08, 0.05, 0.06], atol=0.006)
    assert np.allclose(b.center, [0.4, -0.1, 0.03], atol=0.004)
    err = (b.yaw - yaw + np.pi / 2) % np.pi - np.pi / 2
    assert abs(err) < np.radians(3) and -np.pi / 2 < b.yaw <= np.pi / 2
    assert np.allclose(b.pose[:3, 3], b.center)


def test_icp_recovers_pose_from_partial_view():
    rng = np.random.default_rng(2)
    model = tk.box_points((0, 0, 0), (0.05, 0.03, 0.02), spacing=0.004)
    T_true = pose(rotz(1.1), [0.5, 0.2, 0.02])
    seen = apply(T_true, model)
    seen = seen[seen[:, 0] > 0.47] + rng.normal(0, 0.0007, (int((seen[:, 0] > 0.47).sum()), 3))
    f = icp(model, seen)
    assert f.rmse < 0.002 and f.inliers > 0.95
    moved = apply(f.T, model)
    assert np.abs(moved.mean(0) - apply(T_true, model).mean(0)).max() < 0.004


def test_label_matching():
    assert label_score("the red block", "red_block") == 1.0
    assert label_score("red blocks", "red block") == 1.0
    assert label_score("red block", "red cube") == pytest.approx(1 / 3)
    assert label_score("bowl", "red block") == 0.0


def test_outlier_removal():
    rng = np.random.default_rng(3)
    pts = np.r_[rng.normal(0, 0.01, (500, 3)), [[0.3, 0.3, 0.3], [-0.2, 0.1, 0.4]]]
    out = remove_outliers(pts)
    assert len(out) >= 480 and not np.any(np.linalg.norm(out, axis=1) > 0.2)


def test_env_segmentation_objects_and_boxes(scene):
    ex, tools = scene
    view = tools.look()
    assert view.points.shape[1] == 3 and set(view.cams) == {"front", "side", "back"}
    table = tools.table(view)
    assert abs(table.height) < 0.002
    masks = tools.find(view, ["red block", "the blue cube"])
    assert {m.label for m in masks} == {"red block", "the blue cube"}
    assert all(m.score == 1.0 and view.segments[m.instance] in ("red block", "blue cube") for m in masks)
    dets = {d.label: d for d in tools.objects(view, masks, table)}
    b = tools.box(dets["red block"].points, table)
    assert np.allclose(b.center[:2], [0.45, 0.05], atol=0.006)
    assert np.allclose(b.size, [0.05, 0.036, 0.05], atol=0.008)
    assert abs((b.yaw - 0.4 + np.pi / 2) % np.pi - np.pi / 2) < np.radians(6)
    loose = tools.find(view, "red cube")                     # no exact label: best word overlap
    assert loose and all(m.score < 1.0 for m in loose)


def test_colour_segmentation_splits_instances(scene):
    ex, _ = scene
    tools = Tools(ex.spec, ex, segmenter=ColourSegmenter())
    view = tools.look()
    table = tools.table(view)
    dets = tools.objects(view, tools.find(view, "red"), table)
    centres = sorted(tuple(np.round(tools.box(d.points, table).center[:2], 2)) for d in dets)
    assert len(dets) == 2
    assert np.allclose(centres, [(0.35, -0.2), (0.45, 0.05)], atol=0.012)
    assert tools.find(view, "a thing") == []


def test_mark_segmenter_with_a_stub_chooser(scene):
    ex, _ = scene

    def choose(rgb, regions, label):        # stands in for a VLM: pick the bluest region
        blue = [rgb[m][:, 2].mean() - rgb[m][:, 0].mean() for m in regions]
        return [int(np.argmax(blue))] if blue and max(blue) > 80 else []

    tools = Tools(ex.spec, ex, segmenter=MarkSegmenter(choose))
    view = tools.look()
    table = tools.table(view)
    dets = tools.objects(view, tools.find(view, "the blue thing"), table)
    assert len(dets) == 1
    assert np.allclose(tools.box(dets[0].points, table).center[:2], [0.55, -0.15], atol=0.01)


def test_default_segmenter_falls_back_to_colour(scene):
    ex, tools = scene
    view = tools.look()
    for c in view.cams.values():
        c.seg = None
    masks = tools.find(view, "blue cube")
    assert masks and all(m.instance is None for m in masks)
    assert isinstance(EnvSegmenter().segment(view, ["blue cube"]), list)
