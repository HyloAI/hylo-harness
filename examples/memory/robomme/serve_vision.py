"""Local, evaluation-only Qwen vision endpoint. Run in a compatible Transformers environment.

Uses already downloaded weights and greedy decoding; never reads simulator state. The HTTP
endpoint binds to loopback and accepts image bytes, not filesystem paths or remote image URLs.
"""
from __future__ import annotations

import argparse
import base64
import io
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer


SYSTEM = """You plan robot manipulation from visible evidence and the task instruction.
Images are labeled: a chronological FRONT contact sheet, a CURRENT WRIST view for contact,
then a CURRENT FRONT view for grounding. Use ONLY the LAST, CURRENT FRONT image for x,y points.
Separate demonstrations from the execution that follows them.
Retain important color-to-container bindings, highlighted objects, button press sequences,
object positions, and clearly observed lift/place transitions. Use stable topic names.
The observations array records ONLY new evidence in the CURRENT supplied frame_ids. Do not
copy past reports into new observations or cite old frames. Return [] if no new evidence exists.
Past reports are fallible interpretations. A requested action is NOT evidence it happened.
Do not count a held object repeatedly; count a new lift only after a visible placement/release.
Do not invent occluded contents, successful contact, counts, or missing sequence elements.
When execution_state is provided, use its proprioceptive grasp/lift evidence before changing
from pickup to placement. execution_memory reports sensor-supported completed pick/place cycles;
do not count an intended, incomplete or merely repeated action as another cycle.
When both appear, distinguish a flat colored concentric-ring landing TARGET on the tabletop
from a raised round BUTTON cap on a square base. Placement goes to the flat target, not the button.
Choose ONE immediately executable stage, continuing it until visible evidence supports a change.
Use concise verbs such as pick up, place, press, move, insert, or remain static.
Name the manipulated object in the command, for example "pick up the red cube at <point_1>",
"place the cube onto the target at <point_1>", or "press the button at <point_1>".
For spatial targets use <point_1> and optionally <point_2> in the subgoal, and provide the
corresponding CURRENT front-view target center with integer x,y normalized to 0..999.
Do not use contact-sheet coordinates. Ground the physical object that the gripper should touch.
Return ONLY JSON with this schema:
{"observations":[{"topic":"stable entity or event topic", "claim":"brief visible observation,
including uncertainty", "frame_ids":[integer], "confidence":0.0}],
"subgoal":"one short action with optional <point_1>",
"points":[{"id":"point_1","frame_id":integer,"x":integer,"y":integer}]}
Use at most six observations, cite only supplied frame IDs, and omit uncertain claims or points.
"""

GROUND_SYSTEM = """Locate the requested manipulation targets in this SINGLE current front-camera
image. Return ONLY JSON: {"points":[{"id":"point_1","x":0,"y":0}]}.
Copy each requested point id. x is horizontal left-to-right; y is vertical top-to-bottom.
Both are integer normalized coordinates 0..999. Locate the visible center of the requested object,
not the robot gripper. Ground only visible targets; return an empty points list if ambiguous.
When both appear, a landing TARGET is the flat colored concentric-ring marker on the tabletop;
a BUTTON is a raised round cap on a square base. Do not substitute the button for the landing target.
Do not include prose, markdown, observations, subgoal, or additional keys.
"""

MARK_SYSTEM = """Select the visible image REGION corresponding to the requested target.
You receive the original current FRONT image and a magnified copy with numbered yellow tags.
Black leader lines point from each tag to its candidate region. These are current RGB regions,
not hidden object labels. Pick the number whose leader points to the target, not a nearby object.
A landing target is a flat colored concentric-ring marker. A button is a raised round cap on a
square base. For placement, select the landing destination, NOT the object being carried.
Return ONLY JSON: {"choices":{"point_1":5}}. Replace 5 with the selected region number.
Use null if none of the numbered regions represents the target. Do not return coordinates.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=18231)
    parser.add_argument("--backend", choices=["transformers", "vllm"], default="transformers")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--batch-wait-ms", type=float, default=200)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.64)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 8 or not 0 <= args.batch_wait_ms <= 2000 or not 0 < args.gpu_memory_utilization < 1:
        parser.error("invalid batching or GPU memory limit")
    if args.backend != "vllm" and args.batch_size != 1:
        parser.error("request batching requires vllm")
    source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    import torch
    from PIL import Image
    if args.backend == "vllm":
        from vllm import LLM, SamplingParams
        model = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=args.gpu_memory_utilization,
                    max_model_len=16384, max_num_seqs=args.batch_size, enforce_eager=True,
                    limit_mm_per_prompt={"image": 6}, enable_prefix_caching=False, seed=7,
                    mm_processor_kwargs={"max_pixels": 1048576, "min_pixels": 65536})
    else:
        from transformers import AutoModelForImageTextToText, AutoProcessor
        processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
        model = AutoModelForImageTextToText.from_pretrained(
            args.model, local_files_only=True, dtype=torch.bfloat16,
            device_map="cuda:0", attn_implementation="sdpa",
        ).eval()

    runtime_state = {"healthy": True}
    batcher = None
    if args.backend == "vllm":
        from microbatch import Microbatcher
        def generate_batch(items):
            params = [SamplingParams(temperature=0, max_tokens=item[1], seed=7) for item in items]
            outputs = model.chat([item[0] for item in items], sampling_params=params, use_tqdm=False,
                                 chat_template_kwargs={"enable_thinking": False})
            return [{"text": output.outputs[0].text, "input_tokens": len(output.prompt_token_ids),
                     "output_tokens": len(output.outputs[0].token_ids)} for output in outputs]
        batcher = Microbatcher(generate_batch, size=args.batch_size, wait_seconds=args.batch_wait_ms / 1000)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            data = json.dumps({"ready": runtime_state["healthy"], "model": args.model, "frozen": True,
                               "backend": args.backend, "backend_version": importlib.metadata.version(args.backend),
                               "dtype": "bfloat16",
                               "gpu_memory_utilization": args.gpu_memory_utilization if args.backend == "vllm" else None,
                               "max_num_seqs": args.batch_size, "request_batch_size": args.batch_size,
                               "batch_wait_ms": args.batch_wait_ms,
                               "enable_prefix_caching": False, "enable_thinking": False,
                               "sampling_seed": 7, "max_model_len": 16384,
                               "batcher_sha256": hashlib.sha256(Path(__file__).with_name("microbatch.py").read_bytes()).hexdigest(),
                               "source_sha256": source_sha256}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            started = time.monotonic()
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 20_000_000:
                    raise ValueError("invalid request size")
                request = json.loads(self.rfile.read(length))
                if request.get("mode") not in (None, "ground", "marks"):
                    raise ValueError("unsupported request mode")
                content = [{"type": "text", "text": request["prompt"]}]
                if not 1 <= len(request["images"]) <= 6:
                    raise ValueError("expected one to six images")
                for index, encoded in enumerate(request["images"]):
                    im = Image.open(io.BytesIO(base64.b64decode(encoded, validate=True)))
                    if im.width * im.height > 4_000_000:
                        raise ValueError("image too large")
                    labels = request.get("image_labels", [])
                    if index < len(labels):
                        content.append({"type": "text", "text": labels[index]})
                    if args.backend == "vllm":
                        content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded}})
                    else:
                        content.append({"type": "image", "image": im.convert("RGB")})
                system = {"ground": GROUND_SYSTEM, "marks": MARK_SYSTEM}.get(request.get("mode"), SYSTEM)
                messages = [{"role": "system", "content": [{"type": "text", "text": system}]},
                            {"role": "user", "content": content}]
                max_tokens = 900
                batch_info = {"batch_size": 1}
                if args.backend == "vllm":
                    reply = batcher.submit((messages, max_tokens)).result(timeout=600)
                    answer = reply.value["text"]
                    input_tokens, output_tokens = reply.value["input_tokens"], reply.value["output_tokens"]
                    batch_info = {"batch_size": reply.batch_size, "queue_seconds": reply.queue_seconds,
                                  "batch_seconds": reply.batch_seconds}
                else:
                    inputs = processor.apply_chat_template(
                        messages, tokenize=True, add_generation_prompt=True, return_dict=True,
                        return_tensors="pt", enable_thinking=False,
                    ).to("cuda:0")
                    with torch.inference_mode():
                        output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
                    answer = processor.batch_decode(
                        output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
                    )[0]
                    input_tokens = int(inputs["input_ids"].shape[1])
                    output_tokens = int(output.shape[1] - inputs["input_ids"].shape[1])
                data = json.dumps({"text": answer, "seconds": time.monotonic() - started,
                                   "input_tokens": input_tokens, "output_tokens": output_tokens, **batch_info}).encode()
                self.send_response(200)
            except Exception as exc:
                if type(exc).__name__ == "EngineDeadError":
                    runtime_state["healthy"] = False
                data = json.dumps({"error": f"{type(exc).__name__}: {exc}",
                                   "error_type": type(exc).__name__,
                                   "category": "infrastructure" if type(exc).__name__ in {"EngineDeadError", "BatchOverloaded"} else "model"}).encode()
                self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(data)

    print(json.dumps({"ready": True, "port": args.port}), flush=True)
    server = (ThreadingHTTPServer if batcher else HTTPServer)(("127.0.0.1", args.port), Handler)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        if batcher:
            batcher.close()


if __name__ == "__main__":
    main()
