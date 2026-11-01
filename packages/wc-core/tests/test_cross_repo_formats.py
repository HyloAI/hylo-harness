"""Formats that other repositories write and read: the active version file, the workcell API, the wc dispatcher."""

import pytest
from pydantic import ValidationError

from worldcurrent.core import cli
from worldcurrent.core.deployment import ActiveVersion, read_active, write_active
from worldcurrent.core.remote.workcell import OperatorEvent, StartEpisode


def test_active_version_round_trips_and_says_how_to_recover_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="--publish"):
        read_active(tmp_path)
    av = ActiveVersion(campaign="c", version="cand-1", agent={"type": "x", "params": {"kp": 2.0}},
                       patch={"agent.params.kp": 2.0}, lineage=("cand-1",))
    write_active(tmp_path, av)
    assert read_active(tmp_path) == av and not list(tmp_path.glob("*.tmp"))


def test_workcell_requests_are_checked_on_both_sides():
    assert StartEpisode.model_validate({"task_id": "t", "seed": 3}).seed == 3
    assert OperatorEvent.model_validate({"type": "verdict", "verdict": "partial", "by": "operator"}).verdict == "partial"
    with pytest.raises(ValidationError):
        OperatorEvent.model_validate({"type": "launch"})
    with pytest.raises(ValidationError):
        StartEpisode.model_validate({"task": "typo"})


def test_dispatcher_lists_installed_commands_and_refuses_unknown_ones(capsys):
    assert cli.main([]) == 0 and "run" in capsys.readouterr().out
    assert cli.main(["no-such-command"]) == 2


def test_two_packages_cannot_claim_the_same_command(monkeypatch):
    from importlib.metadata import EntryPoint

    eps = [EntryPoint("run", "a.cli:main", cli.GROUP), EntryPoint("run", "b.cli:main", cli.GROUP)]
    monkeypatch.setattr(cli, "entry_points", lambda group: eps)
    with pytest.raises(RuntimeError, match="more than one package"):
        cli.commands()
