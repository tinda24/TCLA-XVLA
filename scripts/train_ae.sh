#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
platform="${1:?Usage: train_ae.sh robotwin|franka [GPU IDs] [key=value ...]}"
gpus="${2:-0}"
shift "$(( $# >= 2 ? 2 : $# ))"
IFS=',' read -r -a devices <<< "$gpus"
distributed=()
if (( ${#devices[@]} > 1 )); then distributed=(--multi_gpu); fi
CUDA_VISIBLE_DEVICES="$gpus" python -m accelerate.commands.launch \
  "${distributed[@]}" --num_processes "${#devices[@]}" \
  --mixed_precision no --main_process_port "${MAIN_PROCESS_PORT:-29500}" \
  train_ae.py --config_yaml "configs/ae/${platform}.yaml" "$@"
