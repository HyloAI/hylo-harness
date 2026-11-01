"""LeRobot policy + SteeredAgent + learned chunk critic (hylo.agents.critic, model in hylo.instruments.critic)."""

from __future__ import annotations

from typing import Any

from .agent import LeRobotAgent


def make_critic_steered(critic: dict[str, Any], steering: dict[str, Any] | None = None, **lerobot_kwargs: Any):
    """Samples `steering.n_samples` chunks per decision and executes the one the critic values most."""
    from hylo.agents.critic import CriticSteeredAgent

    return CriticSteeredAgent(LeRobotAgent(**lerobot_kwargs), steering, critic)
