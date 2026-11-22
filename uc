#!/bin/sh
# Uncrashed track editor CLI from this checkout: ./uc <command> ...   (./uc help)
repo=$(readlink -f -- "$0")
repo=${repo%/*}
if [ -x "$repo/.venv/bin/python" ]; then python="$repo/.venv/bin/python"; else python=python3; fi
PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}" exec "$python" -m uncrashed_mcp.cli "$@"
