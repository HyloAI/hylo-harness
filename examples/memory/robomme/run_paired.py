"""RoboMME paired pilot: frozen grounded pi0.5, frozen VLM, external visual memory.

This is a Hylo integration experiment, not a reproduction of SERA's Codex planner. The
benchmark builder supplies its official episode metadata, reset demonstration, and action
wrapper. Only RGB, proprioception and task_goal cross into policy/planner requests.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import importlib.metadata
import os
import re
from pathlib import Path
import sys
import tarfile
import time
import traceback
from types import SimpleNamespace
import urllib.request

import numpy as np
from PIL import Image, ImageDraw

from hylo.core.episode_memory import EpisodeScope, RecallQuery
from hylo.memory.episode import EventMemory
from hylo.memory.frames import FrameEvidence
from hylo.memory.visual import VisualPlan, VisualPoint, VisualReportMemory, render_subgoal


def encode(image):
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return base64.b64encode(stream.getvalue()).decode()


def sheet(frames, frame_ids):
    width, height = 256, 216
    grid = Image.new("RGB", (width * 4, height * ((len(frames) + 3) // 4)), "white")
    draw = ImageDraw.Draw(grid)
    for i, (frame, frame_id) in enumerate(zip(frames, frame_ids)):
        tile = Image.fromarray(np.asarray(frame, dtype=np.uint8)).resize((256, 192))
        x, y = i % 4 * width, i // 4 * height
        grid.paste(tile, (x, y + 24))
        draw.text((x + 5, y + 5), f"frame {frame_id}", fill="black")
    return grid


def json_object(text):
    """Accept one fenced/prose-wrapped JSON object; schema validation still checks its data."""
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fenced:
        return json.loads(fenced.group(1))
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in model response")
    value, _ = json.JSONDecoder().raw_decode(text[start:])
    return value


def grounding_points(data, expected_ids, frame_id):
    """Accept the requested schema or Qwen's native labeled point_2d representation."""
    if isinstance(data, list):
        rows = data
    elif "points" in data:
        rows = data["points"]
    elif "point_2d" in data:
        rows = [data]
    else:
        raise ValueError("grounder did not return a supported point representation")
    normalized = []
    for row in rows:
        if len(row) == 1 and next(iter(row)) in expected_ids:
            key = next(iter(row))
            row = {"point_2d": row[key], "label": key}
        if "point_2d" in row:
            if len(row["point_2d"]) != 2:
                raise ValueError("point_2d must contain x and y")
            point_id = row.get("label")
            if len(expected_ids) == 1:
                point_id = expected_ids[0]
            row = {"id": point_id, "x": row["point_2d"][0], "y": row["point_2d"][1]}
        normalized.append(VisualPoint.model_validate({**row, "frame_id": frame_id}))
    if {p.id for p in normalized} != set(expected_ids) or len(normalized) != len(expected_ids):
        raise ValueError("grounder could not uniquely identify requested targets")
    return tuple(normalized)


def first_stage(data, frame_id):
    """Compile an overlong proposal to its first manipulation stage, without executing it.

    This is a language/interface repair, not a completion detector. The planner must use the
    next observation to decide when to move on. Coordinate proposals are only hints; the
    subsequent single-image grounding call determines the actual current target point.
    """
    data = dict(data)
    command = re.split(r"\s+(?:then|and\s+(?=(?:pick|place|press|move|insert|release|lift)\b))\s*",
                       data["subgoal"], maxsplit=1, flags=re.IGNORECASE)[0].rstrip(". ,")
    points = data.get("points", [])
    used = [p for p in points if f"<{p['id']}>" in command]
    if not used and points and command.lower() not in {"remain static", "wait"}:
        used = [points[0]]
        command += f" at <{used[0]['id']}>"
    elif not used and re.match(r"(?:pick|grasp|place|put|press|move|insert)\b", command, flags=re.IGNORECASE):
        used = [{"id": "point_1", "frame_id": frame_id, "x": 0, "y": 0}]
        command += " at <point_1>"
    data["subgoal"], data["points"] = command, used
    return data


def ground(args, command, points, front, frame_id):
    target = re.sub(r"\s+at\s+<point_[12]>", "", command, flags=re.IGNORECASE)
    purpose = "object to contact"
    if re.match(r"(?:place|put)\b", command, flags=re.IGNORECASE):
        parts = re.split(r"\b(?:onto|into|on|in)\b", target, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) == 2:
            target = parts[1].strip()
            purpose = "placement DESTINATION, not the object currently held by the gripper"
    prompt = {
        "target_to_locate": target, "purpose": purpose, "frame_id": frame_id,
        "requested_point_ids": [p.id for p in points],
    }
    candidates = []
    payload = {"mode": "ground", "prompt": json.dumps(prompt), "images": [encode(Image.fromarray(front))]}
    if getattr(args, "rgb_regions", False) and len(points) == 1:
        from grounding import propose, filter_targets
        candidates, annotated = propose(front)
        candidates, appearance_rule = filter_targets(candidates, target, purpose, front.shape)
        if not candidates:
            return None, {"mode": "rgb_regions", "appearance_rule": appearance_rule,
                          "candidates": [], "abstention": "no matching RGB evidence"}
        if len(candidates) == 1 and appearance_rule != "unfiltered":
            c = candidates[0]
            h, w = front.shape[:2]
            return (VisualPoint(id=points[0].id, frame_id=frame_id,
                                x=round(c["x"] * 999 / (w - 1)), y=round(c["y"] * 999 / (h - 1))),), {
                "mode": "rgb_regions", "appearance_rule": appearance_rule, "candidates": candidates,
                "selected_region": c["id"], "selection": "unique_matching_rgb_region"}
        if candidates:
            prompt["candidates"] = candidates
            payload.update(mode="marks", prompt=json.dumps(prompt),
                           images=[encode(Image.fromarray(front)), encode(annotated)])
    request = urllib.request.Request(args.vision_url, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        result = json.load(response)
    # The adapter, rather than the language model, binds coordinates to the sole image sent.
    # Extra top-level explanation fields do not alter point validation and remain in the trace.
    try:
        data = json_object(result["text"])
        if candidates:
            choices = data.get("choices", data)
            selected = choices.get(points[0].id)
            if isinstance(selected, str) and selected.isdecimal():
                selected = int(selected)
            chosen = next((c for c in candidates if type(selected) is int and c["id"] == selected), None)
            result.update(mode="rgb_regions", candidates=candidates, selected_region=selected)
            if chosen is None:
                result["abstention"] = "no valid visible region selected"
                return None, result
            h, w = front.shape[:2]
            return (VisualPoint(id=points[0].id, frame_id=frame_id,
                                x=round(chosen["x"] * 999 / (w - 1)), y=round(chosen["y"] * 999 / (h - 1))),), result
        if data == {"points": []} or data == []:
            return None, result
        grounded = grounding_points(data, [p.id for p in points], frame_id)
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        result["abstention"] = f"invalid grounding output: {exc}"
        return None, result
    return grounded, result


def plan(args, task_goal, front, wrist, frame_ids, memory, *, demonstration=False):
    context = memory.context(task_goal) if args.condition == "memory_on" else {"reports": []}
    retained_ids, retained = [], []
    if args.condition == "memory_on" and getattr(args, "frame_evidence", None) is not None:
        retained_ids, retained = args.frame_evidence.retrieve(before=frame_ids[0])
    supplied_ids = retained_ids + frame_ids
    monitor = getattr(args, "execution_monitor", None)
    execution_memory = None
    if monitor is not None and args.condition == "memory_on":
        execution_memory = memory.journal.recall(RecallQuery(
            name="observed_pick_place_cycles", kind="motion.pick_place", subject=monitor.color,
            op="count", complete=False)).model_dump(mode="json")
    prompt = json.dumps({
        "task_instruction": task_goal, "phase": "demonstration" if demonstration else "execution",
        "frame_ids": supplied_ids, "recent_frame_ids": frame_ids, "current_frame_id": frame_ids[-1],
        "reset_prefix_ends_at": getattr(args, "demo_end", None),
        "memory": context,
        "execution_state": monitor.summary() if monitor is not None else None,
        "execution_memory": execution_memory,
        "request": "Record visible cues; no actions will execute yet." if demonstration else "Choose the next action.",
    })
    images = [sheet(front, frame_ids), Image.fromarray(wrist[-1]), Image.fromarray(front[-1])]
    labels = ["Chronological FRONT contact sheet. Do NOT ground x,y in this composite image.",
              f"CURRENT WRIST view, frame {frame_ids[-1]}. Contact evidence only; do NOT ground x,y here.",
              f"CURRENT FRONT view, frame {frame_ids[-1]}. Ground all points using ONLY this image, x,y in 0..999."]
    memory_images, memory_labels = [], []
    for start in range(0, len(retained), 16):
        memory_images.append(sheet(retained[start:start + 16], retained_ids[start:start + 16]))
        memory_labels.append("PAST FRONT evidence, not current state. Tile labels give chronological frame IDs.")
    images, labels = memory_images + images, memory_labels + labels
    encoded = [encode(im) for im in images]
    attempts = []
    request_prompt = prompt
    for attempt in range(3):
        payload = json.dumps({"prompt": request_prompt, "images": encoded, "image_labels": labels}).encode()
        request = urllib.request.Request(args.vision_url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=600) as response:
            result = json.load(response)
        attempts.append(result)
        raw = result["text"].strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        try:
            data = first_stage(json_object(raw), frame_ids[-1])
            parsed = VisualPlan.model_validate(data)
            if not demonstration:
                render_subgoal(parsed, current_frame_id=frame_ids[-1], width=front[-1].shape[1], height=front[-1].shape[0])
            # Validate citations in both conditions. Repair changes formatting only and never
            # adds new observations, oracle hints, or a claim that the requested action ran.
            if any(not set(c.frame_ids).issubset(supplied_ids) for c in parsed.observations):
                raise ValueError("planner cited a frame absent from its request")
            break
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            result["validation_error"] = str(exc)
            if attempt == 2:
                raise ValueError("planner output failed validation: " + json.dumps(attempts)) from exc
            request_prompt = prompt + "\nRepair the previous JSON to satisfy the schema. " + str(exc) + "\nPrevious response: " + raw
    grounding = None
    command = None
    gate_reason = None
    proposed_subgoal = parsed.subgoal
    if monitor is not None and not demonstration:
        gated, gate_reason = monitor.gate(parsed.subgoal)
        if gate_reason:
            ids_in_command = list(dict.fromkeys(re.findall(r"<(point_[12])>", gated)))
            parsed = parsed.model_copy(update={"subgoal": gated, "points": tuple(
                VisualPoint(id=point_id, frame_id=frame_ids[-1], x=0, y=0) for point_id in ids_in_command)})
    semantic = re.sub(r"<point_[12]>", "<target>", parsed.subgoal.lower())
    if parsed.points and not demonstration:
        points, grounding = ground(args, parsed.subgoal, parsed.points, front[-1], frame_ids[-1])
        if points is None:
            # Both conditions share this bounded actuator-command continuity mechanism.
            # A temporarily occluded target does not fabricate a new current-view point.
            age = frame_ids[-1] - getattr(args, "last_grounded_frame", -10000)
            if getattr(args, "last_semantic", None) == semantic and age <= 128:
                command = args.last_grounded_command
                grounding.update(fallback="continue_same_command", evidence_frame=args.last_grounded_frame, age_frames=age)
            else:
                command = "remain static"
                grounding.update(fallback="abstain_no_current_target")
        else:
            parsed = parsed.model_copy(update={"points": points})
    if command is None:
        command = "remain static" if demonstration else render_subgoal(
            parsed, current_frame_id=frame_ids[-1], width=front[-1].shape[1], height=front[-1].shape[0])
        if not demonstration and parsed.points:
            args.last_semantic, args.last_grounded_command = semantic, command
            args.last_grounded_frame = frame_ids[-1]
    if args.condition == "memory_on":
        memory.write(parsed, supplied_frame_ids=supplied_ids, step=frame_ids[-1])
    if monitor is not None and not demonstration:
        target = None
        if parsed.points and not (grounding or {}).get("fallback"):
            target = (parsed.points[-1].x * (front[-1].shape[1] - 1) / 999,
                      parsed.points[-1].y * (front[-1].shape[0] - 1) / 999)
        monitor.accept(parsed.subgoal, command, target)
    return command, {"request": json.loads(prompt), "response": parsed.model_dump(mode="json"),
                     "attempts": attempts, "grounding": grounding, "command": command,
                     "proposed_subgoal": proposed_subgoal, "execution_gate": gate_reason,
                     "snapshot": memory.journal.snapshot()}


def run_episode(args, task, episode, directory, policy, EnvRunner, pack_buffer):
    directory.mkdir(parents=True, exist_ok=False)
    runner = EnvRunner(task, str(directory), max_steps=args.max_steps, dataset_split=args.split)
    seed, _ = runner.env_builder.resolve_episode(episode)
    np.random.seed(seed)
    journal = EventMemory(str(directory / "memory.sqlite"))
    memory = VisualReportMemory(journal)
    memory.reset(EpisodeScope(task_id=task, episode_id=f"{args.split}:{episode}:{args.condition}"))
    started = time.monotonic()
    result = {"task": task, "episode": episode, "split": args.split, "seed": seed,
              "condition": args.condition, "status": "infra_error", "steps": 0}
    args.last_semantic, args.last_grounded_command, args.last_grounded_frame = None, None, -10000
    frames = []
    try:
        runner.make_env(episode)
        initial = runner.get_init_obs()
        front, wrist, states = initial["images"], initial["wrist_images"], initial["states"]
        assert len(front) == len(wrist) == len(states) and len(front) > 0
        task_goal = initial["task_goal"]
        args.execution_monitor = None
        if args.execution_gates and "cube" in task_goal and "place" in task_goal:
            from execution import UrdfChain, PickPlaceMonitor
            from mani_skill import PACKAGE_ASSET_DIR
            urdf = PACKAGE_ASSET_DIR / "robots/panda/panda_v2.urdf"
            args.execution_monitor = PickPlaceMonitor(UrdfChain(
                urdf, [f"panda_joint{i}" for i in range(1, 8)], "panda_hand_tcp"))
            result["robot_urdf_sha256"] = hashlib.sha256(urdf.read_bytes()).hexdigest()
        args.frame_evidence = FrameEvidence()
        args.demo_end = len(front) - 1
        for i, image in enumerate(front):
            args.frame_evidence.append(i, image, demonstration=i < len(front) - 1)
        result.update(task_goal=task_goal, initial_frames=len(front),
                      render_device=str(runner.env.unwrapped._render_device))
        reset_digest = hashlib.sha256(task_goal.encode())
        for stream in (front, wrist, states):
            array = np.ascontiguousarray(np.stack(stream))
            reset_digest.update(str((array.shape, array.dtype.str)).encode())
            reset_digest.update(array.tobytes())
        result["reset_observation_sha256"] = reset_digest.hexdigest()
        (directory / "initial.json").write_text(json.dumps(result, indent=2))
        # Store every public reset frame losslessly. No segmentation, depth, oracle or hidden
        # environment state is serialized into these agent-visible evidence archives.
        np.savez_compressed(directory / "reset-observations.npz", front=np.stack(front),
                            wrist=np.stack(wrist), state=np.stack(states))
        reset = policy.reset()
        if not reset.get("reset_finished"):
            raise RuntimeError("policy reset did not acknowledge completion")
        policy.add_buffer(pack_buffer(front, states, len(front) - 1))
        trace_path = directory / "planner.jsonl"
        # Last <=16 frames are used in the first action request in BOTH conditions.
        prefix_end = max(0, len(front) - 16)
        if args.condition == "memory_on":
            for start in range(0, prefix_end, 16):
                end = min(start + 16, prefix_end)
                _, record = plan(args, task_goal, front[start:end], wrist[start:end],
                                 list(range(start, end)), memory, demonstration=True)
                with trace_path.open("a") as f:
                    f.write(json.dumps(record) + "\n")
        current_front, current_wrist, current_state = front[-1], wrist[-1], states[-1]
        recent_front, recent_wrist = list(front[prefix_end:]), list(wrist[prefix_end:])
        ids = list(range(prefix_end, len(front)))
        cursor = len(front) - 1
        frames.append(np.concatenate([current_front, current_wrist], axis=1))
        round_index = 0
        while result["steps"] < args.max_steps:
            command, record = plan(args, task_goal, recent_front, recent_wrist, ids, memory)
            with trace_path.open("a") as f:
                f.write(json.dumps(record) + "\n")
            actions = policy.infer({"observation/image": current_front, "observation/wrist_image": current_wrist,
                                    "observation/state": current_state, "prompt": task_goal,
                                    "grounded_subgoal": command})["actions"]
            actions = np.asarray(actions)[:min(16, args.max_steps - result["steps"])]
            if actions.ndim != 2 or actions.shape[1] != 8 or not len(actions) or not np.isfinite(actions).all():
                raise ValueError(f"invalid joint action chunk: {actions.shape}")
            recent_front, recent_wrist, recent_states, ids = [], [], [], []
            stop = False
            for action in actions:
                obs, stop, status = runner.step(action)
                if status == "error":
                    raise RuntimeError("simulator step failed")
                current_front, current_wrist, current_state = obs
                result["steps"] += 1
                cursor += 1
                args.frame_evidence.append(cursor, current_front, demonstration=False)
                if args.execution_monitor is not None:
                    events = args.execution_monitor.observe(current_state, float(action[-1]), current_front, cursor)
                    if args.condition == "memory_on":
                        journal.append(events)
                recent_front.append(current_front)
                recent_wrist.append(current_wrist)
                recent_states.append(current_state)
                ids.append(cursor)
                frames.append(np.concatenate([current_front, current_wrist], axis=1))
                if stop:
                    result["status"] = status
                    break
            np.savez_compressed(directory / f"observations-{round_index:04d}.npz",
                                frame_ids=ids, front=np.stack(recent_front), wrist=np.stack(recent_wrist),
                                state=np.stack(recent_states), actions=actions[:len(ids)])
            print(json.dumps({"task": task, "episode": episode, "condition": args.condition,
                              "steps": result["steps"], "command": command, "stop": bool(stop)}), flush=True)
            round_index += 1
            if stop:
                break
            policy.add_buffer(pack_buffer(recent_front, recent_states))
        else:
            result["status"] = "timeout"
    except Exception as exc:
        if isinstance(exc, (ValueError, OverflowError)):
            result["status"] = "agent_error"
        result["error"] = traceback.format_exc()
        print(result["error"], flush=True)
    finally:
        runner.close_env()
        result.update(seconds=time.monotonic() - started, memory_snapshot=journal.snapshot())
        journal.close()
        (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        if frames:
            import imageio.v2 as imageio
            imageio.mimsave(directory / "rollout.mp4", frames, fps=10)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sera-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=["VideoUnmask", "PickHighlight", "PickXtimes", "PatternLock"])
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--episode-indices", nargs="+", type=int, help="Explicit, preselected metadata episode IDs")
    parser.add_argument("--max-steps", type=int, default=1300)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--conditions", nargs="+", choices=["memory_off", "memory_on"], default=["memory_off", "memory_on"])
    parser.add_argument("--vision-url", default="http://127.0.0.1:18231")
    parser.add_argument("--policy-port", type=int, default=19300)
    parser.add_argument("--rgb-regions", action="store_true", help="Use RGB region choices for single-target grounding")
    parser.add_argument("--execution-gates", action="store_true", help="Use observable Panda pick/place phase checks")
    args = parser.parse_args()
    episode_ids = args.episode_indices or list(range(args.episode_start, args.episode_start + args.episodes))
    if not episode_ids or min(episode_ids) < 0 or len(set(episode_ids)) != len(episode_ids):
        raise ValueError("episode IDs must be nonnegative and unique")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("this pilot is authorized on physical GPU 1 only; set CUDA_VISIBLE_DEVICES=1")
    sys.path.insert(0, str(args.sera_root / "examples/robomme"))
    from env_runner import EnvRunner
    from utils import pack_buffer
    import gymnasium as gym
    import robomme.env_record_wrapper.episode_config_resolver as resolver
    # The released builder does not expose device configuration. Replace only its gym
    # reference to explicitly select the CUDA-masked render device; preserve all task rules.
    resolver.gym = SimpleNamespace(make=lambda env_id, **kw: gym.make(
        env_id, sim_backend="cpu", render_backend="sapien_cuda:0", **kw))
    from openpi_client.websocket_client_policy import MMEVLAWebsocketClientPolicy
    policy = MMEVLAWebsocketClientPolicy(host="127.0.0.1", port=args.policy_port)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    manifest.update(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    protocol="Hylo pilot; shared current window, memory_on also processes reset prefix; unequal planner compute")
    source_root = Path(__file__).resolve().parents[3]
    source_files = list(Path(__file__).parent.glob("*.py"))
    for package in ("hylo-core", "hylo-memory", "hylo-manip"):
        source_files.extend((source_root / "packages" / package / "src").rglob("*.py"))
    manifest["source_sha256"] = {
        str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(source_files)
    }
    with urllib.request.urlopen(args.vision_url, timeout=30) as response:
        manifest["vision_server"] = json.load(response)
    server_source = manifest["vision_server"].get("source_sha256")
    if server_source != manifest["source_sha256"]["examples/memory/robomme/serve_vision.py"]:
        raise ValueError("vision server runs a different source revision; restart it before freezing the run")
    manifest["simulator_packages"] = {name: importlib.metadata.version(name)
                                      for name in ("torch", "numpy", "sapien", "mani-skill", "pydantic")}
    manifest_path = args.output / "protocol.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("output contains a different protocol; use a new output directory")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    archive_path = args.output / "source.tar.gz"
    if not archive_path.exists():
        with tarfile.open(archive_path, "x:gz") as archive:
            for path in sorted(source_files):
                archive.add(path, arcname=str(path.relative_to(source_root)), recursive=False)
    results = []
    for task in args.tasks:
        for episode in episode_ids:
            conditions = args.conditions if episode % 2 == 0 else list(reversed(args.conditions))
            for condition in conditions:
                args.condition = condition
                directory = args.output / task / f"{episode:02d}-{condition}"
                result_path = directory / "result.json"
                if result_path.exists():
                    result = json.loads(result_path.read_text())
                else:
                    result = run_episode(args, task, episode, directory, policy, EnvRunner, pack_buffer)
                results.append(result)
                (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
