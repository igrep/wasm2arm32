#!/bin/bash

set -eux

prg_full_local="$1"
shift

args="$@"

prg="$(basename "$prg_full_local")"
prg_full_remote='~/tmp/wasm2arm-test'

cat "$prg_full_local" | ssh "$SSH_RASPI_USER_HOST" "cat > $prg_full_remote && chmod +x $prg_full_remote && $prg_full_remote $args"
# ssh "$SSH_RASPI_USER_HOST" -- "$prg_full_remote"
