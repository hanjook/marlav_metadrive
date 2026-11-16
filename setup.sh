#!/usr/bin/env bash
# One-time setup. The venv lives outside this folder because Ray 2.2 cannot start workers when the
# Python path contains a space ("UC Berkeley").
set -e
cd "$(dirname "$0")"
VENV="$HOME/.venvs/metadrive-static-goal"
[ -d upstream ] || git clone https://github.com/metadriverse/metadrive.git upstream
git -C upstream checkout fe54b03327b14bcfc17a9a04b60c8abbee4d7ab1
uv venv "$VENV" --python 3.10
uv pip install --python "$VENV/bin/python" -e ./upstream -r requirements.txt
# python.org Python has no CA bundle configured; use certifi's for MetaDrive's asset download
SSL_CERT_FILE="$("$VENV/bin/python" -c 'import certifi; print(certifi.where())')" "$VENV/bin/python" -m metadrive.pull_asset
echo "Done. Use: $VENV/bin/python static_goal/..."
