#!/bin/bash
# Reproducible entry point included in clean installation/test archives.
root="$(realpath "$(dirname "$0")/..")"
exec "$root/.pixi/envs/default/bin/python" "$root/scripts/e2e_cycle.py" "$@"
