"""Manipulation toolkit for tool-using agents (RFC-0006 part 2): stateless perception, kinematics, grasping and
motion-planning tools over one robot described by `hylo.core.robot.RobotSpec`. Programs see them as `robot`."""

from .executor import Executor
from .perception import ColourSegmenter, EnvSegmenter, MarkSegmenter, Segmenter
from .tools import Box, CameraView, Detection, Fit, Goal, Mask, Plan, Table, Tools, View, api_reference

META = {"contracts": {}, "maturity": "unverified", "plugins": {}}

__all__ = ["Box", "CameraView", "ColourSegmenter", "Detection", "EnvSegmenter", "Executor", "Fit", "Goal",
           "MarkSegmenter", "Mask", "META", "Plan", "Segmenter", "Table", "Tools", "View", "api_reference"]
