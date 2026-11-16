#!/usr/bin/env bash
# RoboTwin raw + latent policy training on 8 GPUs.
# Same hyperparameters as configs/policy/base.yaml, except:
#   10000 iterations, checkpoint every 1000 steps.
#   Names look like ckpt-10000-bs8x8-bins-24 (raw uses bins-none).
# Raw chunk_size and the frozen ActionAE chunk are both required to be 32.
# Latent policy tokens are chunk_size / downsample_rate (8 when downsample_rate is 4).
#
# Usage:
#   bash scripts/train_robotwin_1k10k_8gpu.sh          # raw, then latent
#   bash scripts/train_robotwin_1k10k_8gpu.sh raw
#   bash scripts/train_robotwin_1k10k_8gpu.sh latent
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

MODE="${1:-both}"
case "${MODE}" in
  raw|latent|both) ;;
  *) echo "Usage: $0 [raw|latent|both]" >&2; exit 2 ;;
esac

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
ITERS="${ITERS:-10000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
CHUNK_SIZE="${CHUNK_SIZE:-32}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/results/robotwin_policy_1k10k}"
CONDA_ENV="${CONDA_ENV:-XVLA}"

if [[ -f /mnt/pre02/data/hqd/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /mnt/pre02/data/hqd/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV}"
fi

IFS=',' read -r -a devices <<< "${GPUS}"
if (( ${#devices[@]} != 8 )); then
  echo "Expected 8 GPUs, got ${#devices[@]} (${GPUS})" >&2
  exit 2
fi
if (( ITERS % SAVE_INTERVAL != 0 || SAVE_INTERVAL <= 0 )); then
  echo "ITERS (${ITERS}) must be a positive multiple of SAVE_INTERVAL (${SAVE_INTERVAL})" >&2
  exit 2
fi

python - <<PY
import sys
from pathlib import Path
import yaml

sys.path.insert(0, "${ROOT}")
from configuration import policy_config

chunk = int("${CHUNK_SIZE}")
cfg = policy_config("configs/policy/robotwin.yaml", "raw")
raw_chunk = int(cfg.raw.chunk_size)
if raw_chunk != chunk:
    raise SystemExit(f"policy raw.chunk_size is {raw_chunk}, expected {chunk}")

ae_dir = Path(str(cfg.paths.action_ae.robotwin))
ae_cfg_path = ae_dir / "config.yaml"
framework = yaml.safe_load(ae_cfg_path.read_text())["framework"]
ae_chunk = int(framework["chunk_size"])
downsample = int(framework["downsample_rate"])
if ae_chunk != chunk:
    raise SystemExit(f"ActionAE chunk_size is {ae_chunk}, expected {chunk} ({ae_cfg_path})")
if downsample <= 0 or ae_chunk % downsample != 0:
    raise SystemExit(f"chunk_size {ae_chunk} is not divisible by downsample_rate {downsample}")
print(
    f"chunk aligned: raw={raw_chunk} ae={ae_chunk} "
    f"latent_tokens={ae_chunk // downsample} latent_dim={framework['latent_action_dim']}"
)
PY

run_one() {
  local mode="$1"
  local port="$2"
  echo "=== ${mode} GPUs=${GPUS} iters=${ITERS} save_every=${SAVE_INTERVAL} chunk=${CHUNK_SIZE} ==="
  echo "    output=${OUTPUT_ROOT}/${mode}"
  mkdir -p "${OUTPUT_ROOT}/${mode}"
  CUDA_VISIBLE_DEVICES="${GPUS}" \
  MAIN_PROCESS_PORT="${port}" \
  MIXED_PRECISION="${MIXED_PRECISION:-no}" \
  bash scripts/train_policy.sh robotwin "${mode}" "${GPUS}" \
    "training.iters=${ITERS}" \
    "training.save_interval=${SAVE_INTERVAL}" \
    "raw.chunk_size=${CHUNK_SIZE}" \
    "output_dir=${OUTPUT_ROOT}/${mode}"
}

case "${MODE}" in
  raw) run_one raw "${RAW_PORT:-29511}" ;;
  latent) run_one latent "${LATENT_PORT:-29512}" ;;
  both)
    run_one raw "${RAW_PORT:-29511}"
    run_one latent "${LATENT_PORT:-29512}"
    ;;
esac
