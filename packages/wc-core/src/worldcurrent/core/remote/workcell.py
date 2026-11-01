"""Workcell API (wc.workcell/v0.1): what a workcell serves and a control plane calls.

A workcell (wc-serve) runs next to a robot or simulator and executes episodes with a remote policy. A control
plane (the playground in wc-studio) starts episodes and relays operator input. Both sides use the models here,
so neither imports the other.

    GET  /healthz                                  liveness, no auth
    GET  /status                                   the workcell's state (dict; see WORKCELL_STATES)
    POST /episodes   StartEpisode  -> EpisodeStarted     409 while an episode runs or a fault is latched
    POST /operator   OperatorEvent -> OperatorAck        400 for an unknown or malformed event
    WS   /events                                   the live event stream: one JSON event per message

Auth, when the workcell has a token: `Authorization: Bearer <token>` on HTTP, `?token=<token>` on the WebSocket.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from ..contract import Contract, JsonAny

WORKCELL_API = "wc.workcell/v0.1"
PATHS = {"healthz": "/healthz", "status": "/status", "episodes": "/episodes", "operator": "/operator", "events": "/events"}
WORKCELL_STATES = ("idle", "starting", "running", "awaiting_reset", "awaiting_verdict", "fault")
OperatorEventType = Literal["estop", "stop", "clear_fault", "reset_done", "verdict"]


class PolicyEndpoint(Contract):
    url: str                                  # ws://host:port/wc or ws://host:port/openpi
    protocol: Literal["wc", "openpi"] = "wc"
    token: str | None = None
    params: dict[str, JsonAny] = Field(default_factory=dict)   # extra client params (openpi inputs, execution_steps...)


class StartEpisode(Contract):
    task_id: str | None = None                # a task of the workcell's benchmark; or give `instruction` alone
    instruction: str | None = None            # free-form task, or an override of the task's instruction
    seed: int = 0
    max_steps: int | None = None
    policy: PolicyEndpoint | None = None      # default: the workcell's configured policy


class EpisodeStarted(Contract):
    task_id: str
    policy: str


class OperatorEvent(Contract):
    type: OperatorEventType
    by: str | None = None                     # who; the control plane fills in the authenticated user
    verdict: Literal["success", "failure", "partial"] | None = None   # type == "verdict"
    note: str | None = None


class OperatorAck(Contract):
    fault: str | None
    state: str | None


__all__ = ["WORKCELL_API", "PATHS", "WORKCELL_STATES", "OperatorEventType", "PolicyEndpoint", "StartEpisode",
           "EpisodeStarted", "OperatorEvent", "OperatorAck"]
