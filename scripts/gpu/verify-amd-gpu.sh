#!/usr/bin/env bash
# After setup-amd-gpu-root.sh + reboot: checks the driver, permissions, PyTorch ROCm and GPU rendering. No root.
# Builds envs/rocm/.venv (PyTorch ROCm wheels bundle the HIP runtime; nothing system-wide).
set -uo pipefail
cd "$(dirname "$0")/../.."
ok() { echo "  [ok]   $*"; }; bad() { echo "  [FAIL] $*"; FAILED=1; }
FAILED=0
echo "driver"
for d in 63:00.0 73:00.0; do
  lspci -k -s "$d" | grep -q "Kernel driver in use: amdgpu" && ok "$d bound to amdgpu" || bad "$d has no driver (reboot? dkms status?)"
done
GPUS=$(grep -l "gfx_target_version 1201" /sys/class/kfd/kfd/topology/nodes/*/properties 2>/dev/null | wc -l)
[ "$GPUS" -ge 1 ] && ok "KFD sees $GPUS gfx1201 GPU node(s)" || bad "KFD has no gfx1201 node"
echo "permissions"
id -nG | grep -qw render && ok "user in render group" || bad "user not in render group (log out and back in)"
ls /dev/dri/renderD* >/dev/null 2>&1 && ok "render nodes: $(ls /dev/dri/renderD* | tr '\n' ' ')" || bad "no /dev/dri/renderD*"
echo "pytorch rocm (envs/rocm/.venv)"
if [ ! -x envs/rocm/.venv/bin/python ]; then
  uv venv -q envs/rocm/.venv --python 3.12
  uv pip install -q --python envs/rocm/.venv/bin/python torch --index-url https://download.pytorch.org/whl/rocm7.14
fi
envs/rocm/.venv/bin/python - <<'PY' && ok "torch sees the GPUs" || bad "torch cannot use the GPUs"
import torch
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"         gpu{i}: {p.name} {getattr(p, 'gcnArchName', '')} {p.total_memory / 2**30:.0f} GiB")
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize(); print(f"         bf16 matmul ok: {(x @ x).float().abs().mean().item():.2f}")
PY
echo "mujoco egl rendering"
MUJOCO_GL=egl .venv/bin/python - <<'PY' && ok "MuJoCo renders with EGL" || bad "MuJoCo EGL rendering failed"
import mujoco, time
m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><light pos="0 0 1"/><geom type="sphere" size=".1"/></worldbody></mujoco>')
d = mujoco.MjData(m); r = mujoco.Renderer(m, 256, 256); mujoco.mj_forward(m, d)
t = time.time(); [r.update_scene(d) or r.render() for _ in range(50)]; print(f"         {1000 * (time.time() - t) / 50:.1f} ms/frame")
PY
[ "$FAILED" = 0 ] && echo "GPU stack ready." || { echo "Some checks failed (see above)."; exit 1; }
