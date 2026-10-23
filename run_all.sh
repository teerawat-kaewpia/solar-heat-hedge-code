#!/usr/bin/env bash
# Convenience wrapper for macOS/Linux/Git Bash. The Python runner is canonical.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$SCRIPT_DIR/run_all.py" "$@"
