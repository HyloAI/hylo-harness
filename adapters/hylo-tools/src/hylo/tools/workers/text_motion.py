# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "torch>=2.5",
#   "kimodo @ git+https://github.com/nv-tlabs/kimodo@e1c219c99b4b94ba48be1fb6a1cbc64ecee4c954",
#   "mujoco>=3.2", "PyOpenGL>=3.1", "imageio>=2.30", "imageio-ffmpeg>=0.5", "pillow>=10", "numpy>=1.26",
# ]
# [tool.uv]
# index-strategy = "unsafe-best-match"
# python-preference = "only-managed"   # kimodo builds a C++ extension: needs Python headers (uv-managed Pythons have them)
# [[tool.uv.index]]
# url = "https://download.pytorch.org/whl/cpu"
# ///
"""Text -> humanoid motion with NVIDIA Kimodo on the CPU, plus keypoint-IK retargeting to a MuJoCo humanoid.

params: prompt (str, or [str] for consecutive segments), duration (s, or [s] per segment; default 4),
        model (soma | g1 | a Kimodo model name; default soma = Kimodo-SOMA-RP-v1.2), steps (denoising, default 100),
        seed (0), cfg_weight ([text, constraint], default model's), postprocess (foot-skate cleanup; default true,
        never for g1), retarget (bool, default true: SOMA motion -> built-in humanoid qpos), render (bool, default
        true: stick-figure and robot MP4 + PNG strip; robot views need MUJOCO_GL=osmesa or egl)
outputs: motion.npz: joints (T, J, 3) metres, z up, x forward (MuJoCo convention), names (J,), parents (J,), fps,
         foot_contacts (T, 4) [left heel, left toe, right heel, right toe], global_rot_mats (T, J, 3, 3) in the same
         frame, local_rot_mats (T, J, 3, 3) (Kimodo skeleton convention), root (T, 3);
         g1 only: qpos (T, 36) for Kimodo's Unitree G1 MJCF (free base + 29 joints), qpos_names;
         robot.npz (retarget): qpos (T, nq), joint_names, err (T,), xml (the humanoid MJCF text)
"""

from __future__ import annotations

import os

import numpy as np

from _common import hf_revision, main, setup_torch, versions

LICENCE_CODE = "Apache-2.0 (nv-tlabs/kimodo)"
LICENCE_WEIGHTS = "NVIDIA Open Model License"
TEXT_ENCODER = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp(-supervised), base meta-llama/Meta-Llama-3-8B-Instruct"
ALIASES = {"soma": "Kimodo-SOMA-RP-v1.2", "g1": "Kimodo-G1-RP-v1", "soma-seed": "Kimodo-SOMA-SEED-v1.1",
           "g1-seed": "Kimodo-G1-SEED-v1"}
# Kimodo: y up, z forward -> MuJoCo / Hylo: z up, x forward (rows: new axes in old coordinates)
K2M = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def load(params):
    os.environ.setdefault("TEXT_ENCODER_MODE", "local")     # no text-encoder server; LLM2Vec in this process
    os.environ["TEXT_ENCODER_DEVICE"] = "cpu"
    torch = setup_torch()
    return {"torch": torch, "models": {}, "text_encoder": None}


def _model(st, name):
    from kimodo.model.load_model import load_model
    from kimodo.model.registry import get_model_info
    name = ALIASES.get(name, name)
    if name not in st["models"]:
        import time
        t0 = time.perf_counter()
        model, key = load_model(name, device="cpu", default_family="Kimodo", return_resolved_name=True,
                                text_encoder=st["text_encoder"], text_encoder_fp32=True)
        if st["text_encoder"] is None:
            st["text_encoder"] = model.text_encoder        # shared by the other Kimodo models
        info = get_model_info(key)
        st["models"][name] = (model, key, info.repo_id if info else name)
        st["model_load_s"] = round(time.perf_counter() - t0, 3)
    return st["models"][name]


def _frames(rot):
    return np.einsum("ij,...jk,lk->...il", K2M, rot, K2M)


def handle(st, req):
    torch = st["torch"]
    prompt = req.get("prompt")
    if not prompt:
        raise ValueError("text_motion: pass a prompt")
    prompts = [prompt] if isinstance(prompt, str) else list(prompt)
    dur = req.get("duration", 4.0)
    durs = [float(dur)] * len(prompts) if isinstance(dur, (int, float)) else [float(x) for x in dur]
    st.pop("model_load_s", None)
    model, key, repo = _model(st, req.get("model", "soma"))
    if "model_load_s" in st:
        req.timing["model_load_s"] = st["model_load_s"]     # first request of a worker: weights (and download)
    fps = float(model.fps)
    frames = [max(8, int(round(d * fps))) for d in durs]
    is_g1 = "g1" in key
    post = bool(req.get("postprocess", True)) and not is_g1
    seed = int(req.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)
    kw = {}
    if req.get("cfg_weight") is not None:
        kw["cfg_weight"] = req.get("cfg_weight")
    import time
    t_gen = time.perf_counter()
    with torch.inference_mode():
        out = model(prompts if len(prompts) > 1 else prompts[0], frames if len(frames) > 1 else frames[0],
                    num_denoising_steps=int(req.get("steps", 100)), multi_prompt=len(prompts) > 1, num_samples=1,
                    post_processing=post, return_numpy=True, progress_bar=lambda x, **_: x, **kw)
    req.timing["generate_s"] = round(time.perf_counter() - t_gen, 3)
    first = {k: (v[0] if hasattr(v, "shape") and v.ndim > 0 and v.shape[0] == 1 else v) for k, v in out.items()}
    sk = model.skeleton
    if hasattr(sk, "somaskel77") and first["posed_joints"].shape[1] == len(sk.somaskel77.bone_order_names):
        sk = sk.somaskel77                                   # SOMA-30 models output the 77-joint skeleton
    names = list(sk.bone_order_names)
    parents = sk.joint_parents.cpu().numpy()
    joints = first["posed_joints"] @ K2M.T
    arrays = dict(joints=joints.astype(np.float32), names=np.array(names), parents=parents, fps=np.float32(fps),
                  foot_contacts=np.asarray(first.get("foot_contacts", np.zeros((len(joints), 4)))),
                  global_rot_mats=_frames(first["global_rot_mats"]).astype(np.float32),
                  local_rot_mats=np.asarray(first["local_rot_mats"], np.float32),
                  root=(np.asarray(first["root_positions"]) @ K2M.T).astype(np.float32))
    outputs = {"frames": int(len(joints)), "fps": fps, "joints": len(names), "model": repo, "prompts": prompts,
               "durations_s": durs, "postprocess": post, "skeleton": sk.name}
    if is_g1:
        from kimodo.exports.mujoco import MujocoQposConverter
        conv = MujocoQposConverter(model.skeleton)
        tens = {k: torch.as_tensor(v)[None] if isinstance(v, np.ndarray) else v for k, v in first.items()}
        qpos = conv.dict_to_qpos(tens, "cpu")
        qpos = np.asarray(qpos.cpu().numpy() if hasattr(qpos, "cpu") else qpos)[0]
        arrays["qpos"] = qpos.astype(np.float32)
        import mujoco
        g1 = mujoco.MjModel.from_xml_path(str(conv.xml_path))
        arrays["qpos_names"] = np.array([g1.joint(j).name for j in range(g1.njnt) if g1.jnt_type[j] in (2, 3)])
        outputs["g1_xml"] = str(conv.xml_path)
    req.save_npz("motion", **arrays)

    robot = None
    if req.get("retarget", True) and not is_g1:
        from text_motion_retarget import KeypointIKRetargeter
        rt = KeypointIKRetargeter()
        t_rt = time.perf_counter()
        robot = rt(joints, names, fps)
        req.timing["retarget_s"] = round(time.perf_counter() - t_rt, 3)
        req.save_npz("robot", qpos=robot.qpos.astype(np.float32), joint_names=np.array(robot.joint_names),
                     err=robot.err.astype(np.float32), xml=np.array(robot.xml), fps=np.float32(fps))
        outputs["retarget"] = {"robot": "hylo_humanoid (built-in)", "nq": int(robot.qpos.shape[1]),
                               "mean_keypoint_err_m": round(float(robot.err.mean()), 4),
                               "max_keypoint_err_m": round(float(robot.err.max()), 4)}
    if req.get("render", True):
        _render(req, joints, parents, names, fps, robot, arrays.get("qpos"), outputs.get("g1_xml"), prompts)

    req.provenance.update({"model": repo, "revision": hf_revision(repo), "licence": f"{LICENCE_WEIGHTS} (weights); "
                           f"{LICENCE_CODE}; text encoder: Llama 3 Community License + MIT (LLM2Vec)",
                           "text_encoder": TEXT_ENCODER, "backend": "kimodo", "dtype": "float32",
                           "packages": versions("torch", "transformers", "kimodo", "mujoco")})
    return outputs


def _render(req, joints, parents, names, fps, robot, g1_qpos, g1_xml, prompts):
    from text_motion_stick import stick_video
    stride = 2 if fps > 20 else 1
    stick_video(joints, parents, str(req.path("skeleton.mp4", "skeleton_video")), fps, names=names, still=str(req.path("skeleton_strip.png")),
                stride=stride, title=" / ".join(prompts))
    if os.environ.get("MUJOCO_GL") not in ("osmesa", "egl"):
        req.provenance["render_note"] = "robot views skipped: set MUJOCO_GL=osmesa or egl"
        return
    from text_motion_retarget import render_robot
    if robot is not None:
        render_robot(robot.xml, robot.qpos, str(req.path("robot.mp4", "robot_video")), fps, stride=stride,
                     still=str(req.path("robot_strip.png")))
    if g1_qpos is not None and g1_xml:
        render_robot(g1_xml, g1_qpos, str(req.path("robot.mp4", "robot_video")), fps, stride=stride,
                     still=str(req.path("robot_strip.png")))


if __name__ == "__main__":
    raise SystemExit(main(handle=handle, load=load))
