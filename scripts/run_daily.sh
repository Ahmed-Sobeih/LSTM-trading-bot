#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ -f /workspace/lstm-environment/activate.sh ]]; then
    source /workspace/lstm-environment/activate.sh
fi
exec python daily.py "$@"
