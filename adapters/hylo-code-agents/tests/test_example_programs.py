"""The reference programs for mujoco/manip-v0, mujoco/bimanual-v0 and mujoco/humanoid-v0 (examples/programs) pass
the sandbox's validation as code.program sources; examples/programs/run_programs.py evaluates them."""

from pathlib import Path

import pytest

from hylo.codeagents.agents import CodeProgramAgent

PROGRAMS = Path(__file__).resolve().parents[1] / "examples" / "programs"
TASKS = ("pick_place", "stack", "put_in_bowl", "push_to_zone", "press_by_number", "sort_by_colour")
BIMANUAL = ("handover", "lift_box", "two_bins")
HUMANOID = ("g1_pick_place", "g1_two_hands", "g1_press_button")


@pytest.mark.parametrize("task", TASKS + BIMANUAL + HUMANOID)
def test_reference_program_is_a_valid_code_program(task):
    agent = CodeProgramAgent(source=(PROGRAMS / f"{task}.py").read_text())
    assert agent.entry == "solve" and agent.digest.startswith("sha256:")


def test_instruction_parsing_of_the_reference_programs():
    ns: dict = {}
    exec((PROGRAMS / "stack.py").read_text(), ns)            # plain Python: the helpers run outside the sandbox too
    labels = ["red block", "blue block", "green block", "yellow block"]
    assert ns["stack_order"]("put the red block on top of the blue block", labels) == ["blue block", "red block"]
    assert ns["stack_order"]("build a tower: the green block at the bottom, then the red block, then the blue "
                             "block on top", labels) == ["green block", "red block", "blue block"]
    assert ns["stack_order"]("put the red block on the blue block, then the green block on the red block",
                             labels) == ["blue block", "red block", "green block"]
    exec((PROGRAMS / "pick_place.py").read_text(), ns)
    plates = labels + ["plate", "pink plate"]
    assert ns["parse"]("pick up the red block and put it on the pink plate", plates) == ("red block", "pink plate")
    assert ns["parse"]("place the blue block on the plate", plates) == ("blue block", "plate")
    exec((PROGRAMS / "press_by_number.py").read_text(), ns)
    assert ns["press_order"]("push the buttons numbered 3, 5 and 1, in that order", [1, 3, 5, 7]) == [3, 5, 1]
    assert ns["press_order"]("press every button, from the highest number to the lowest", [1, 3, 5]) == [5, 3, 1]
    assert ns["press_order"]("press all the buttons in ascending order of their numbers", [4, 2]) == [2, 4]


def test_instruction_parsing_of_the_two_arm_programs():
    ns: dict = {}
    exec((PROGRAMS / "two_bins.py").read_text(), ns)
    labels = ["red block", "blue block", "white bin", "grey bin"]
    assert ns["pairs"]("at the same time, place the blue block in the grey bin and the red block in the white bin",
                       labels) == [("blue block", "grey bin"), ("red block", "white bin")]
    exec((PROGRAMS / "lift_box.py").read_text(), ns)
    assert ns["height_cm"]("use both grippers to raise the red box 12 cm and keep it level") == 12.0
    exec((PROGRAMS / "handover.py").read_text(), ns)
    assert ns["mentioned"]("move the blue bar into the green zone; you will need both arms",
                           ["blue bar", "green zone", "red zone"]) == ["green zone", "blue bar"]
