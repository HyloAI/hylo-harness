"""Opt-in CPU and GPU differential checks against the pinned upstream runtime.

Run from a prepared inference environment, with upstream and Hylo packages on PYTHONPATH:
python verify_upstream.py --checkpoint /path/to/checkpoint --output /path/to/results.json
No installation, source mutation, training or implicit checkpoint download is performed.
"""
import argparse
from collections import deque
import json
from pathlib import Path
import time

import numpy as np


def cpu_parity(config, upstream_type):
    from hylo.policies.simplememvla.bridge import history_from_config
    from types import SimpleNamespace
    checks = 0
    for variable in (False, True):
        cfg = SimpleNamespace(**vars(config)) if not hasattr(config, "to_dict") else SimpleNamespace(**config.to_dict())
        cfg.variable_history = variable
        memory = history_from_config(cfg)
        upstream = object.__new__(upstream_type)
        upstream.image_keys = list(memory.image_keys)
        upstream.history_image_keys = list(memory.history_image_keys)
        upstream.n_frames, upstream.stride = memory.n_frames, memory.stride
        upstream.variable_history = variable
        upstream._buffer_caps = memory.capacities
        upstream._buffers = {k: [] for k in memory.image_keys}
        upstream._frames_seen = dict.fromkeys(memory.image_keys, 0)
        for i in range(2401):
            frames = {k: np.full((2, 2, 3), i % 251, dtype=np.uint8) for k in memory.image_keys}
            memory.observe(frames)
            upstream.observe(frames)
            if i in {0, 1, 9, 10, 1189, 1190, 1191, 1300, 2400}:
                for camera in memory.history_image_keys:
                    assert memory.frame_ids(camera) == upstream._history_global_indices(camera)
                    np.testing.assert_array_equal(memory.video(camera), upstream._history_video(camera))
                keys = [(k.split(".")[-1], a, b) for k, a, b in memory.patch_keys()]
                assert keys == upstream._video_patch_keys()
                checks += 1
    return checks


def run(checkpoint, output, on_runtime=None):
    import torch
    from robomme_sim.policy import RoboMMEPolicy
    from hylo.policies.simplememvla.runtime import SimpleMemVLARuntime
    started = time.perf_counter()
    torch.manual_seed(0)
    np.random.seed(0)
    runtime = SimpleMemVLARuntime(checkpoint)
    results = dict(cpu_sampling_checks=cpu_parity(runtime.config, RoboMMEPolicy), gpu_checks=[])
    output = Path(output)
    def save():
        output.write_text(json.dumps(results, indent=2) + "\n")
    save()
    native = [RoboMMEPolicy(runtime.model, runtime.processor, runtime.unnormalize, runtime.normalize) for _ in range(2)]
    hylo = [runtime.new_session() for _ in range(2)]
    assert hylo[0].memory.feature_cache is not hylo[1].memory.feature_cache
    instructions = ["pick the red cube", "press the button"]
    states = [np.zeros(runtime.config.state_dim, dtype=np.float32), np.ones(runtime.config.state_dim, dtype=np.float32) * .01]
    milestones = {0, 16, 1190, 1206, 1300}
    for i in range(1301):
        for row in range(2):
            # Nonconstant deterministic RGB; different rows must never share cached features.
            rgb = np.empty((256, 256, 3), dtype=np.uint8)
            rgb[:, :, 0] = (i + row * 17) % 251
            rgb[:, :, 1] = np.arange(256, dtype=np.uint8)[:, None]
            rgb[:, :, 2] = np.arange(256, dtype=np.uint8)[None, :]
            frames = {k: rgb for k in runtime.config.image_keys}
            native[row].observe(frames)
            hylo[row].observe(frames)
        if i not in milestones:
            continue
        if i == 1206:
            instructions[0] = "place the red cube on the table"  # Invalidate prompt/feature cache.
        native_inputs = [p._prepare_inputs(text) for p, text in zip(native, instructions)]
        hylo_inputs = [p._prepare_inputs(text) for p, text in zip(hylo, instructions)]
        for a, b in zip(native_inputs, hylo_inputs):
            assert a.keys() == b.keys()
            for key in a:
                if key == "policy":
                    continue
                if torch.is_tensor(a[key]):
                    assert torch.equal(a[key], b[key]), key
                elif isinstance(a[key], dict):
                    assert a[key].keys() == b[key].keys()
                    assert all(torch.equal(a[key][k], b[key][k]) for k in a[key]), key
                else:
                    assert a[key] == b[key], key
        normalized = [runtime.normalize.normalize(torch.from_numpy(s)).unsqueeze(0).to(runtime.device) for s in states]
        seed = 1729 + i
        with torch.random.fork_rng(devices=[runtime.device.index or 0]):
            torch.manual_seed(seed)
            expected = runtime.batched.generate_batch(native_inputs, normalized)
        actual = runtime.infer_batch(hylo, instructions, states, seed=seed)
        for (a, text_a), (b, text_b) in zip(expected, actual):
            np.testing.assert_array_equal(a, b)
            assert text_a == text_b
        assert all(len(p.memory.feature_cache) <= 60 for p in hylo)
        results["gpu_checks"].append(dict(frame=i, batch=2, inputs_equal=True, actions_bitwise_equal=True,
                                           subtasks_equal=True, max_abs_action_error=0.0))
        save()
        print(json.dumps(results["gpu_checks"][-1]), flush=True)
    for p in hylo:
        p.reset()
        assert not p.memory.feature_cache and all(n == 0 for n in p.memory.frames_seen.values())
    results.update(complete=True, seconds=time.perf_counter() - started)
    save()
    if on_runtime is not None:
        on_runtime(runtime)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.checkpoint, args.output)
