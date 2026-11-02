# hylo: worker-side (runs in a tool worker's own environment; never imported by Hylo packages)
"""Blender-side script of the render worker: blender -b --factory-startup --python render_blender.py -- scene.json

Builds the exported MuJoCo geometry (meshes in geom frames, per-frame poses, camera path) as a Cycles scene on the
CPU: principled materials (image textures where the MJCF had them), a soft three-area-light rig, a sky gradient, a
ground plane when the scene has none, filmic (AgX) colour, denoising when available; renders one PNG per frame.
Runs inside Blender's own Python (bpy, numpy); never imported by Hylo.
"""

from __future__ import annotations

import json
import math
import sys
import time

import bpy
import numpy as np
from mathutils import Matrix, Vector


def args() -> dict:
    argv = sys.argv[sys.argv.index("--") + 1:]
    with open(argv[0]) as f:
        return json.load(f)


def setup_render(cfg: dict) -> bpy.types.Scene:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    r = sc.render
    r.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = int(cfg["samples"])
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.max_bounces = 6
    sc.cycles.caustics_reflective = sc.cycles.caustics_refractive = False
    try:
        sc.cycles.use_denoising = bool(cfg.get("denoise", True))
        sc.cycles.denoiser = "OPENIMAGEDENOISE"
    except Exception:  # noqa: BLE001
        sc.cycles.use_denoising = False
    r.threads_mode, r.threads = "FIXED", int(cfg.get("threads", 8))
    r.resolution_x, r.resolution_y, r.resolution_percentage = int(cfg["width"]), int(cfg["height"]), 100
    r.image_settings.file_format = "PNG"
    r.image_settings.color_mode = "RGB"
    r.film_transparent = False
    for vt in ("AgX", "Filmic", "Standard"):
        try:
            sc.view_settings.view_transform = vt
            break
        except TypeError:
            continue
    try:
        sc.view_settings.look = "AgX - Medium High Contrast" if sc.view_settings.view_transform == "AgX" else "None"
    except TypeError:
        pass
    return sc


def world(sc: bpy.types.Scene) -> None:
    w = bpy.data.worlds.new("sky")
    sc.world = w
    w.use_nodes = True
    nt = w.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    nt.links.new(tc.outputs["Generated"], sep.inputs[0])     # the ray direction: Z = 0 at the horizon, 1 up
    nt.links.new(sep.outputs["Z"], ramp.inputs["Fac"])
    ramp.color_ramp.elements[0].position, ramp.color_ramp.elements[0].color = 0.0, (0.50, 0.58, 0.72, 1)
    ramp.color_ramp.elements[1].position, ramp.color_ramp.elements[1].color = 0.4, (0.28, 0.45, 0.80, 1)
    nt.links.new(ramp.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = 0.4
    nt.links.new(bg.outputs[0], out.inputs[0])


def material(g: dict) -> bpy.types.Material:
    mat = bpy.data.materials.new(g["name"] + "_mat")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    rgba = g["rgba"]
    base = (rgba[0], rgba[1], rgba[2], 1.0)
    bsdf.inputs["Base Color"].default_value = [c ** 2.2 for c in base[:3]] + [1.0]   # MuJoCo rgba is sRGB-ish
    bsdf.inputs["Roughness"].default_value = float(g.get("roughness", 0.5))
    bsdf.inputs["Metallic"].default_value = float(g.get("metallic", 0.0))
    if rgba[3] < 1.0:
        bsdf.inputs["Alpha"].default_value = float(rgba[3])
    if g.get("emission", 0) > 0 and "Emission Strength" in bsdf.inputs:
        bsdf.inputs["Emission Color"].default_value = bsdf.inputs["Base Color"].default_value
        bsdf.inputs["Emission Strength"].default_value = float(g["emission"])
    if g.get("texture"):
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(g["texture"])
        tex.interpolation = "Cubic"
        uvn = nt.nodes.new("ShaderNodeUVMap")
        nt.links.new(uvn.outputs["UV"], tex.inputs["Vector"])
        mix = nt.nodes.new("ShaderNodeMix")
        mix.data_type, mix.blend_type = "RGBA", "MULTIPLY"
        mix.inputs["Factor"].default_value = 1.0
        nt.links.new(tex.outputs["Color"], mix.inputs[6])
        mix.inputs[7].default_value = [c ** 2.2 for c in base[:3]] + [1.0]
        nt.links.new(mix.outputs[2], bsdf.inputs["Base Color"])
    return mat


def mesh_object(g: dict, arrays) -> bpy.types.Object:
    v = arrays[f"v_{g['key']}"]
    f = arrays[f"f_{g['key']}"]
    me = bpy.data.meshes.new(g["name"])
    me.vertices.add(len(v))
    me.vertices.foreach_set("co", v.astype(np.float32).ravel())
    me.loops.add(f.size)
    me.loops.foreach_set("vertex_index", f.astype(np.int32).ravel())
    me.polygons.add(len(f))
    me.polygons.foreach_set("loop_start", np.arange(0, f.size, 3, dtype=np.int32))
    me.polygons.foreach_set("loop_total", np.full(len(f), 3, dtype=np.int32))
    uvk = f"uv_{g['key']}"
    if uvk in arrays:
        uv = arrays[uvk]
        layer = me.uv_layers.new(name="UVMap")
        layer.data.foreach_set("uv", uv[f.ravel()].astype(np.float32).ravel())
    me.update(calc_edges=True)
    me.validate()
    if g["smooth"] is True:
        me.shade_smooth()
    elif g["smooth"] == "angle":
        try:
            me.shade_smooth()
            me.set_sharp_from_angle(angle=math.radians(35))
        except Exception:  # noqa: BLE001
            pass
    else:
        me.shade_flat()
    me.materials.append(material(g))
    ob = bpy.data.objects.new(g["name"], me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def ground(cfg: dict) -> None:
    c, r = cfg["centre"], cfg["radius"]
    me = bpy.data.meshes.new("ground")
    s = max(r * 12, 4.0)
    me.from_pydata([(-s, -s, 0), (s, -s, 0), (s, s, 0), (-s, s, 0)], [], [(0, 1, 2, 3)])
    mat = bpy.data.materials.new("ground_mat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    col = cfg.get("floor_rgba") or (0.80, 0.80, 0.78, 1)
    bsdf.inputs["Base Color"].default_value = [x ** 2.2 for x in col[:3]] + [1.0]
    bsdf.inputs["Roughness"].default_value = 0.7
    me.materials.append(mat)
    ob = bpy.data.objects.new("ground", me)
    ob.location = (c[0], c[1], cfg["floor"] - 1e-3)
    bpy.context.scene.collection.objects.link(ob)


def lights(cfg: dict) -> None:
    c, r = Vector(cfg["centre"]), max(cfg["radius"], 0.1)
    rig = [("key", (1.2, -1.0, 2.2), 260.0, 1.2), ("fill", (-1.4, -0.6, 1.4), 70.0, 2.0),
           ("rim", (-0.3, 1.5, 1.8), 140.0, 1.0)]
    for name, off, power, size in rig:
        ld = bpy.data.lights.new(name, "AREA")
        ld.shape, ld.size = "DISK", size * r * 1.5
        d = 2.5 * r
        ld.energy = power * (d / 1.0) ** 2 * 0.25
        ob = bpy.data.objects.new(name, ld)
        ob.location = c + Vector(off) * d
        ob.rotation_euler = (c - ob.location).to_track_quat("-Z", "Y").to_euler()
        bpy.context.scene.collection.objects.link(ob)


def camera(cfg: dict) -> bpy.types.Object:
    cd = bpy.data.cameras.new("cam")
    cd.sensor_fit = "VERTICAL"
    cd.angle_y = math.radians(cfg["fovy"])
    cd.clip_start = 0.005
    cd.clip_end = 1000
    ob = bpy.data.objects.new("cam", cd)
    bpy.context.scene.collection.objects.link(ob)
    bpy.context.scene.camera = ob
    return ob


def main() -> None:
    cfg = args()
    t0 = time.time()
    sc = setup_render(cfg)
    world(sc)
    arrays = dict(np.load(cfg["geometry"]))
    objs = [mesh_object(g, arrays) for g in cfg["geoms"]]
    if not cfg["has_plane"]:
        ground(cfg)
    lights(cfg)
    cam = camera(cfg)
    poses, cams = arrays["poses"], arrays["cams"]
    print(f"[hylo-render] scene built in {time.time() - t0:.2f} s: {len(objs)} geoms, {len(cams)} frames", flush=True)
    for i in range(len(cams)):
        for ob, T in zip(objs, poses[i]):
            ob.matrix_world = Matrix(T.tolist())
        cam.matrix_world = Matrix(cams[i].tolist())
        sc.render.filepath = f"{cfg['frames_dir']}/{i:05d}.png"
        t1 = time.time()
        bpy.ops.render.render(write_still=True)
        print(f"[hylo-render] frame {i} {time.time() - t1:.2f} s", flush=True)


if __name__ == "__main__":
    main()
