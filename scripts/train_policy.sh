#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
platform="${1:?Usage: train_policy.sh robotwin|franka raw|latent [GPU IDs] [key=value ...]}"
mode="${2:?Missing policy mode}"
case "$mode" in
  raw|latent) ;;
  *) echo "Policy mode must be raw or latent" >&2; exit 2 ;;
esac
gpus="${3:-0}"
shift "$(( $# >= 3 ? 3 : $# ))"
IFS=',' read -r -a devices <<< "$gpus"
distributed=()
if (( ${#devices[@]} > 1 )); then distributed=(--multi_gpu); fi
CUDA_VISIBLE_DEVICES="$gpus" python -m accelerate.commands.launch \
  "${distributed[@]}" --num_processes "${#devices[@]}" \
  --mixed_precision "${MIXED_PRECISION:-no}" --main_process_port "${MAIN_PROCESS_PORT:-29500}" \
  train.py --config "configs/policy/${platform}.yaml" --mode "$mode" "$@"
