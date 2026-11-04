"""Observable execution checks; no simulator instance, reward or oracle is accepted.

This prototype assumes Panda parallel jaws and colored cubes placed on flat targets. A
sensor-derived cycle is a hypothesis supported by jaw position, relative TCP lift and RGB
overlap, not a privileged benchmark success predicate.
"""
from __future__ import annotations

from collections import deque
import re
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from hylo.core.episode_memory import MemoryEvent


class UrdfChain:
    """Forward kinematics from joint origins/axes only; meshes and simulation are unnecessary."""
    def __init__(self, path, joint_names, end_link):
        root = ET.parse(path).getroot()
        parents = {j.find("child").attrib["link"]: j for j in root.findall("joint")}
        chain = []
        while end_link in parents:
            joint = parents[end_link]
            origin = joint.find("origin")
            values = {} if origin is None else origin.attrib
            transform = np.eye(4)
            transform[:3, 3] = np.fromstring(values.get("xyz", "0 0 0"), sep=" ")
            transform[:3, :3] = Rotation.from_euler("xyz", np.fromstring(values.get("rpy", "0 0 0"), sep=" ")).as_matrix()
            axis = joint.find("axis")
            direction = np.fromstring("1 0 0" if axis is None else axis.attrib["xyz"], sep=" ")
            chain.append((joint.attrib["name"], joint.attrib["type"], transform, direction))
            end_link = joint.find("parent").attrib["link"]
        self.chain, self.names = list(reversed(chain)), tuple(joint_names)
        if not self.chain:
            raise ValueError("end link has no URDF chain")

    def __call__(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape != (len(self.names),) or not np.isfinite(q).all():
            raise ValueError("invalid joint state")
        values = dict(zip(self.names, q))
        result = np.eye(4)
        for name, kind, origin, axis in self.chain:
            motion = np.eye(4)
            if kind in {"revolute", "continuous"}:
                motion[:3, :3] = Rotation.from_rotvec(axis / np.linalg.norm(axis) * values[name]).as_matrix()
            elif kind == "prismatic":
                motion[:3, 3] = axis * values[name]
            elif kind != "fixed":
                raise ValueError(f"unsupported URDF joint: {kind}")
            result = result @ origin @ motion
        return result[:3, 3].copy()


def verb(command):
    text = command.lower().strip()
    if text.startswith(("pick", "grasp", "lift")):
        return "pick"
    if text.startswith(("place", "put")):
        return "place"
    return "other"


class PickPlaceMonitor:
    def __init__(self, fk, *, lift_distance=0.08):
        self.fk, self.lift_distance = fk, lift_distance
        self.apertures = deque(maxlen=3)
        self.contact = None
        self.lift_frame = None
        self.target = None
        self.last_template = None
        self.last_verb = "other"
        self.color = None
        self.serial = 0
        self.last_z = None
        self.release_wait = 0
        self.primitive_frames = 0

    @property
    def holding(self):
        return self.contact is not None

    def gate(self, proposal):
        new_verb = verb(proposal)
        # A press usually takes multiple executor chunks. Keep an accepted press for a
        # bounded actuator horizon, shared by both conditions, before allowing replanning.
        # This is command continuity, not evidence that the button was successfully pressed.
        if self.last_template and self.last_template.lower().startswith("press") and self.primitive_frames < 64:
            return self.last_template, "continue_press_primitive"
        if self.last_template and self.last_verb == "pick" and new_verb == "place" and self.lift_frame is None:
            return self.last_template, "wait_for_observed_grasp_and_relative_lift"
        if self.last_template and self.last_verb == "place" and self.holding and new_verb in {"pick", "other"}:
            return self.last_template, "wait_for_observed_release"
        return proposal, None

    def accept(self, template, command, target=None):
        if command == "remain static":
            return
        if verb(command) != self.last_verb or (template.lower().startswith("press") != bool(self.last_template and self.last_template.lower().startswith("press"))):
            self.primitive_frames = 0
        self.last_template, self.last_verb = template, verb(command)
        match = re.search(r"\b(red|green|blue|cyan|purple|pink|yellow) cube\b", command.lower())
        if match:
            self.color = match.group(1)
        if self.last_verb == "place" and target is not None:
            self.target = target

    def summary(self):
        return {"grasp_contact_candidate": self.holding, "relative_lift_observed": self.lift_frame is not None,
                "release_evidence_pending": self.release_wait > 0,
                "active_primitive": self.last_verb, "source": "joint_proprioception_and_public_robot_urdf"}

    def observe(self, state, gripper_command, frame, frame_id):
        self.primitive_frames += 1
        state = np.asarray(state)
        aperture = float(state[-1])
        z = float(self.fk(state[:7])[2])
        self.last_z = z
        self.apertures.append(aperture)
        events = []
        contact_evidence = (len(self.apertures) == 3 and 0.004 < aperture < 0.035
                            and max(self.apertures) - min(self.apertures) < 0.001 and gripper_command < -0.5)
        if self.contact is None and self.last_verb == "pick" and contact_evidence:
            self.contact = {"frame": frame_id, "z": z}
        if self.contact is not None and self.lift_frame is None and z - self.contact["z"] >= self.lift_distance and aperture > 0.004:
            self.lift_frame = frame_id
        if self.contact is not None and aperture >= 0.036:
            self.release_wait += 1
            if self.last_verb == "place" and self.lift_frame is not None and self.target is not None and self.color:
                from grounding import propose
                candidates, _ = propose(frame)
                radius = frame.shape[1] * 0.05
                matches = [c for c in candidates if c["color"] == self.color and
                           np.hypot(c["x"] - self.target[0], c["y"] - self.target[1]) <= radius]
                # Require release near the original contact height, not a mid-air drop.
                if matches and z <= self.contact["z"] + 0.08:
                    events.append(MemoryEvent(
                        id=f"pick-place:{self.serial}", step=frame_id, time=float(frame_id),
                        kind="motion.pick_place", subject=self.color, confidence=0.9,
                        value={"color": self.color, "contact_frame": self.contact["frame"],
                               "lift_frame": self.lift_frame, "release_frame": frame_id,
                               "release_tcp_z": z, "contact_tcp_z": self.contact["z"],
                               "target_pixels": list(self.target), "status": "sensor_supported_hypothesis"},
                        evidence=tuple(f"frame:{i}" for i in (self.contact["frame"], self.lift_frame, frame_id)),
                        source="panda-joints-and-rgb/v1",
                    ))
                    self.serial += 1
            # Allow brief gripper occlusion/settling after opening; repeated open frames do
            # not become multiple events. Missing evidence eventually expires explicitly.
            if events or self.release_wait >= 8 or self.last_verb != "place":
                self.contact, self.lift_frame, self.release_wait = None, None, 0
        elif aperture <= 0.003:
            # A fully closed empty jaw invalidates a contact hypothesis; it is not a successful grasp.
            self.contact, self.lift_frame, self.release_wait = None, None, 0
        return events
