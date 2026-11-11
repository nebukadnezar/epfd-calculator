#!/bin/sh
# Run a script with the Python interpreter named in config.json:
#   ./run.sh run_example.py
cd "$(dirname "$0")" || exit 1
PY=$(sed -n 's/.*"python"[^"]*"\([^"]*\)".*/\1/p' "${EPFD_CONFIG:-config.json}")
case "$PY" in "~"*) PY="$HOME${PY#\~}" ;; esac
exec "${PY:-python3}" "$@"
