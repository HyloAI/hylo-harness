"""Boundary lint for the Hylo repositories (dev/repos.toml) and the packages inside them.

Runs from a development workspace: the repositories side by side in one directory (dev/bootstrap.sh). It checks
the repositories that are present, so it also runs with the harness alone.

1. Package layering: every `hylo.*` import (and "pkg.mod:attr" / "-m pkg.mod" string reference) in a
   package's src/ must target a package allowed for it in ALLOWED (docs/dependency-graph.md).
2. Repository rule: a package may import or declare only packages of its own repository and of the
   repositories it depends on (repos.toml). Leaves never import each other; the harness imports no leaf.
3. Robot side: packages of a `robot_side` repository (serve) declare no Hylo dependency but hylo-core,
   so a robot computer installs nothing else by default.
4. No code or config file reaches into another repository by path (a "../<repo>/" path). Exempt: deployment
   configs (they only compose installed repositories) and install scripts, which find a sibling checkout
   through an environment variable with a side-by-side default (bench's envs/libero/setup.sh: HYLO_HARNESS).
"""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path

import pytest

HARNESS = Path(__file__).resolve().parents[1]
WORKSPACE = HARNESS.parent
REPOS = tomllib.loads((HARNESS / "dev" / "repos.toml").read_text())["repos"]
CODE_REPOS = [r for r, spec in REPOS.items() if spec.get("code", True)]

# module prefix -> owning distribution
OWNER = {
    "hylo.core": "hylo-core",
    "hylo.testing": "hylo-core",
    "hylo.eval": "hylo-eval",
    "hylo.bench": "hylo-bench",
    "hylo.agents": "hylo-agents",
    "hylo.memory": "hylo-memory",
    "hylo.evolve": "hylo-evolve",
    "hylo.envs.mujoco": "hylo-env-mujoco",
    "hylo.reasoners.llm": "hylo-reasoner-llm",
    "hylo.proposers.llm": "hylo-proposer-llm",
    "hylo.envs.libero": "hylo-env-libero",
    "hylo.policies.lerobot": "hylo-policy-lerobot",
    "hylo.policies.openpi": "hylo-policy-openpi",
    "hylo.cli": "hylo-cli",
    "hylo.learn": "hylo-learn",
    "hylo.playground": "hylo-playground",
    "hylo.catalog": "hylo-catalog",
    "hylo.serve": "hylo-serve",
    "hylo.envs.lerobot_robot": "hylo-env-lerobot-robot",
    "hylo.datasources.video": "hylo-data-video",
    "hylo.policies.reference": "hylo-policies",
    "hylo_examples": "hylo-plugin-examples",
}

# Package layering (within and across repositories). Every package is allowed itself.
ALLOWED = {
    # harness
    "hylo-core": set(),
    "hylo-eval": {"hylo-core"},
    "hylo-agents": {"hylo-core"},
    "hylo-memory": {"hylo-core"},
    "hylo-policies": {"hylo-core", "hylo-agents"},
    "hylo-learn": {"hylo-core", "hylo-eval", "hylo-agents"},
    "hylo-evolve": {"hylo-core", "hylo-eval", "hylo-learn"},
    "hylo-catalog": {"hylo-core", "hylo-eval"},
    "hylo-cli": {"hylo-core", "hylo-eval", "hylo-agents", "hylo-memory", "hylo-evolve", "hylo-learn"},
    "hylo-reasoner-llm": {"hylo-core"},
    "hylo-proposer-llm": {"hylo-core", "hylo-agents", "hylo-evolve", "hylo-reasoner-llm"},
    "hylo-policy-lerobot": {"hylo-core", "hylo-agents", "hylo-memory"},
    "hylo-policy-openpi": {"hylo-core"},
    "hylo-data-video": {"hylo-core"},
    # bench
    "hylo-bench": {"hylo-core", "hylo-eval"},
    "hylo-env-libero": {"hylo-core"},
    "hylo-env-mujoco": {"hylo-core"},
    # serve
    "hylo-serve": {"hylo-core"},
    "hylo-env-lerobot-robot": {"hylo-core"},
    # studio
    "hylo-playground": {"hylo-core", "hylo-eval", "hylo-catalog"},
    # plugins
    "hylo-plugin-examples": {"hylo-core", "hylo-eval", "hylo-learn"},
}

HEAVY = {"torch", "jax", "mujoco", "langchain_core", "langgraph", "dspy", "robosuite"}
HEAVY_ALLOWED = {"hylo-env-mujoco": {"mujoco"}, "hylo-policy-lerobot": {"torch"}}

NOT_MODULES = {"hylo.commands", "hylo.meta"}      # entry-point group names, not module references
REF = re.compile(r"^hylo(\.[a-z_0-9]+)+(:[A-Za-z_][A-Za-z_0-9.]*)?$")


def _package_dirs() -> dict[str, tuple[str, Path]]:
    """distribution name -> (repository, package directory)"""
    out = {}
    for repo in CODE_REPOS:
        for pyproject in sorted((WORKSPACE / repo).glob("*/*/pyproject.toml")):
            name = tomllib.loads(pyproject.read_text())["project"]["name"]
            out[name] = (repo, pyproject.parent)
    return out


PACKAGES = _package_dirs()


def _repo_of(dist: str) -> str:
    return PACKAGES[dist][0]


def _reachable(repo: str) -> set[str]:
    return {repo, *REPOS[repo].get("depends", [])}


def _owner(module: str) -> str | None:
    best = max((p for p in OWNER if module == p or module.startswith(p + ".")), key=len, default=None)
    return OWNER[best] if best else None


def _imports(path: Path) -> list[str]:
    """Imported modules, plus string references to modules ("pkg.mod:attr" factories, "-m pkg.mod" commands):
    a string reference is a dependency the import graph would otherwise hide."""
    tree = ast.parse(path.read_text(), filename=str(path))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            out.append(node.module)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and REF.match(node.value) \
                and node.value not in NOT_MODULES:
            out.append(node.value.partition(":")[0])
    return out


def _declared(dist: str) -> set[str]:
    project = tomllib.loads((PACKAGES[dist][1] / "pyproject.toml").read_text())["project"]
    names = {re.split(r"[\s<>=!~;\[]", d, maxsplit=1)[0] for d in project.get("dependencies", [])}
    return {n for n in names if n in PACKAGES}


PRESENT = sorted(PACKAGES)


def test_every_package_is_classified():
    assert not set(PACKAGES) - set(ALLOWED), "add the new package to ALLOWED"
    assert {"hylo-core", "hylo-eval"} <= set(PACKAGES)


def test_layering_never_points_from_the_base_to_a_leaf():
    for dist in PRESENT:
        reach = _reachable(_repo_of(dist))
        crossing = {d for d in ALLOWED[dist] if d in PACKAGES and _repo_of(d) not in reach}
        assert not crossing, f"{dist}: ALLOWED crosses into {sorted(crossing)}"


@pytest.mark.parametrize("dist", PRESENT)
def test_package_imports_respect_layering(dist: str):
    repo, root = PACKAGES[dist]
    violations = []
    for py in (root / "src").rglob("*.py"):
        for mod in _imports(py):
            top = mod.split(".")[0]
            if mod.startswith("hylo.") or top == "hylo_examples":
                owner = _owner(mod)
                if owner is None or (owner != dist and owner not in ALLOWED[dist]):
                    violations.append(f"{py.relative_to(WORKSPACE)} imports {mod} (owner {owner})")
            elif top in HEAVY and top not in HEAVY_ALLOWED.get(dist, set()):
                violations.append(f"{py.relative_to(WORKSPACE)} imports heavy dependency {mod}")
    assert not violations, "\n".join(violations)


@pytest.mark.parametrize("dist", PRESENT)
def test_declared_dependencies_stay_within_reachable_repositories(dist: str):
    repo = _repo_of(dist)
    bad = {d for d in _declared(dist) if _repo_of(d) not in _reachable(repo)}
    assert not bad, f"{dist} ({repo}) declares {sorted(bad)} from {sorted({_repo_of(d) for d in bad})}"
    if REPOS[repo].get("robot_side"):
        extra = _declared(dist) - {"hylo-core"}
        assert not extra, f"{dist} is robot-side: declare {sorted(extra)} as an optional extra, not a dependency"


CROSS = re.compile(r"\.\./(" + "|".join(map(re.escape, REPOS)) + r")/")
CODE_SUFFIXES = {".py", ".yaml", ".yml", ".toml", ".ts", ".tsx", ".json"}


@pytest.mark.parametrize("repo", [r for r in CODE_REPOS if (WORKSPACE / r).is_dir()])
def test_no_code_reaches_into_another_repository_by_path(repo: str):
    hits = []
    for f in (WORKSPACE / repo).rglob("*"):
        parts = set(f.relative_to(WORKSPACE / repo).parts)
        if f.suffix not in CODE_SUFFIXES or parts & {".venv", "node_modules", "runs", "static", "envs"} or not f.is_file():
            continue
        for i, line in enumerate(f.read_text(errors="ignore").splitlines(), 1):
            if CROSS.search(line):
                hits.append(f"{f.relative_to(WORKSPACE)}:{i}: {line.strip()[:100]}")
    assert not hits, "\n".join(hits)


def test_the_workspace_file_matches_the_repositories():
    import importlib.util

    spec = importlib.util.spec_from_file_location("workspace", HARNESS / "dev" / "workspace.py")
    ws = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ws)
    assert (WORKSPACE / "pyproject.toml").read_text() == ws.render(WORKSPACE), "run python harness/dev/workspace.py"
