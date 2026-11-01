#!/usr/bin/env bash
# Development workspace: the worldcurrent repositories side by side in one directory, installed into one venv.
#
#   harness/dev/bootstrap.sh [DIR]      DIR: where the repositories sit (default: the directory holding this harness)
#
# A repository that is missing is cloned when WC_GIT_BASE is set (clone URL: $WC_GIT_BASE/worldcurrent-<name>.git),
# skipped otherwise. Then DIR/pyproject.toml is (re)written and `uv sync` installs every package editable.
set -euo pipefail
HARNESS=$(cd "$(dirname "$0")/.." && pwd)
DIR=${1:-$(dirname "$HARNESS")}
REPOS=$(python3 -c 'import sys, tomllib; r = tomllib.load(open(sys.argv[1], "rb"))["repos"]; print(" ".join(k for k in r if k != "deployments-template"))' "$HARNESS/dev/repos.toml")
for repo in $REPOS; do
  if [ -d "$DIR/$repo" ]; then
    continue
  elif [ -n "${WC_GIT_BASE:-}" ]; then
    git clone "$WC_GIT_BASE/worldcurrent-$repo.git" "$DIR/$repo"
  else
    echo "not checked out: $repo (set WC_GIT_BASE to clone it)"
  fi
done
python3 "$HARNESS/dev/workspace.py" "$DIR"
cd "$DIR" && uv sync
echo "workspace ready in $DIR: uv run pytest   (boundaries: uv run pytest harness/dev)"
