"""Hylo tool adapters (RFC-0006 part 8): perception and motion models, each run as a worker in its own uv
environment on the CPU (a GPU only makes them faster).

- runner: worker processes, file exchange (JSON, .npz, .png, .glb, .mp4), timing, provenance, persistent workers
- segment: text prompts -> instance masks (SAM 3); `Sam3Segmenter` implements the hylo.manip segmenter interface
- depth: one image -> metric depth, points, normals, intrinsics (MoGe-2)
- gravity: one image -> gravity direction (roll, pitch) and focal length (GeoCalib)
- mesh: image + mask -> object mesh (SAM 3D Objects when access is granted; depth back-projection fallback)
- convex: mesh -> convex collision parts (CoACD) and an MJCF snippet
- hands: video -> per-frame 3D hand poses (WiLoR with MANO; MediaPipe Hands keypoints fallback)
- human: video -> human motion (GVHMR with SMPL; MediaPipe Pose 3D keypoints fallback)
- text_motion: text -> humanoid joint trajectories and a retargeting hook
- render: MuJoCo scene / trajectory -> MP4 (Blender Cycles on the CPU; MuJoCo offscreen fallback)
"""

TOOLS = {
    "segment": "hylo.tools.segment",
    "depth": "hylo.tools.depth",
    "gravity": "hylo.tools.gravity",
    "mesh": "hylo.tools.mesh",
    "convex": "hylo.tools.convex",
    "physicalize": "hylo.tools.physicalize",
    "hands": "hylo.tools.hands",
    "human": "hylo.tools.human",
    "text_motion": "hylo.tools.text_motion",
    "render": "hylo.tools.render",
}

# contract versions and what is known about these plugins (hylo.core.plugins)
META = {
    "contracts": {"commands": "hylo.commands"},
    "maturity": "implemented",
    "plugins": {
        "commands/tool": {"notes": "nine tool adapters, each run on this workstation's CPU on a recorded example; "
                                   "see harness/adapters/hylo-tools/README.md for models, licences, CPU times and status"},
    },
}
