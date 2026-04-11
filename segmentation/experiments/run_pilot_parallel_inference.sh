#!/usr/bin/env bash
set -euo pipefail

# Parallel inference benchmark for pilot model at 512 and 1024.
# Uses two CPU cores (core 0 and core 1) when taskset is available.
# Writes separate logs and JSON outputs per resolution.
#
# Usage:
#   bash segmentation/experiments/run_pilot_parallel_inference.sh
#   MODEL_PATH=/abs/path/model.keras RUNS=3 WARMUP=1 bash segmentation/experiments/run_pilot_parallel_inference.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/segmentation/results/pilot/pilot_model.keras}"
RUNS="${RUNS:-1}"
WARMUP="${WARMUP:-1}"

OUT_BASE="${ROOT_DIR}/segmentation/results/pilot/inference_parallel"
OUT_512="${OUT_BASE}/512"
OUT_1024="${OUT_BASE}/1024"
LOG_512="${OUT_512}/run.log"
LOG_1024="${OUT_1024}/run.log"

mkdir -p "${OUT_512}" "${OUT_1024}"

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "[ERROR] Model file not found: ${MODEL_PATH}"
  echo "Set MODEL_PATH to your .keras/.h5 model."
  exit 1
fi

echo "============================================================"
echo "Parallel Inference Benchmark (Pilot Model)"
echo "Model   : ${MODEL_PATH}"
echo "Runs    : ${RUNS}"
echo "Warmup  : ${WARMUP}"
echo "Output  : ${OUT_BASE}"
echo "============================================================"

run_job() {
  local core="$1"
  local resolution="$2"
  local outdir="$3"
  local logfile="$4"

  local cmd=(
    python "${SCRIPT_DIR}/rpi_benchmark.py"
    --model "${MODEL_PATH}"
    --resolution "${resolution}"
    --runs "${RUNS}"
    --warmup "${WARMUP}"
    --output "${outdir}"
  )

  if command -v taskset >/dev/null 2>&1; then
    nohup taskset -c "${core}" "${cmd[@]}" > "${logfile}" 2>&1 &
  else
    nohup "${cmd[@]}" > "${logfile}" 2>&1 &
  fi

  echo $!
}

PID_512="$(run_job 0 512 "${OUT_512}" "${LOG_512}")"
PID_1024="$(run_job 1 1024 "${OUT_1024}" "${LOG_1024}")"

echo "[*] Started 512 job  (PID=${PID_512})"
echo "[*] Started 1024 job (PID=${PID_1024})"

echo "[*] Waiting for both jobs to finish..."
wait "${PID_512}"
wait "${PID_1024}"

JSON_512="${OUT_512}/$(basename "${MODEL_PATH%.*}")_benchmark.json"
JSON_1024="${OUT_1024}/$(basename "${MODEL_PATH%.*}")_benchmark.json"

echo ""
echo "============================================================"
echo "Done"
echo "512 JSON  : ${JSON_512}"
echo "1024 JSON : ${JSON_1024}"
echo "512 LOG   : ${LOG_512}"
echo "1024 LOG  : ${LOG_1024}"
echo "============================================================"
