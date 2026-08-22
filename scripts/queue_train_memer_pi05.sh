#!/usr/bin/env bash

set -euo pipefail

# Queue a MemER pi0.5 run behind an existing GPU job. The launcher waits for
# WAIT_PIDS and then for all selected GPUs to become idle. It falls back to the
# next batch size only when the failed attempt looks like an OOM.

OPENPI_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_SCRIPT="${OPENPI_DIR}/scripts/train_memer_pi05.sh"

GPU_IDS="${GPU_IDS:-4,5,6,7}"
WAIT_PIDS="${WAIT_PIDS:-}"
POLL_SECONDS="${POLL_SECONDS:-30}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${OPENPI_DIR}/logs/memer_pi05}"
CONFIG_NAME="${CONFIG_NAME:-memer_wa01}"

mkdir -p "${LOG_DIR}"
printf '%s\n' "$$" > "${LOG_DIR}/queue_${RUN_TAG}.pid"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

is_positive_integer() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

if ! is_positive_integer "${POLL_SECONDS}"; then
  log "POLL_SECONDS must be a positive integer, got ${POLL_SECONDS}."
  exit 2
fi
if [[ ! -x "${TRAIN_SCRIPT}" ]]; then
  log "Training launcher is missing or not executable: ${TRAIN_SCRIPT}"
  exit 1
fi

wait_for_initial_processes() {
  local pid
  local -a alive

  if [[ -z "${WAIT_PIDS}" ]]; then
    return
  fi

  log "Waiting for current training PIDs: ${WAIT_PIDS}"
  while true; do
    alive=()
    for pid in ${WAIT_PIDS//,/ }; do
      if [[ -d "/proc/${pid}" ]]; then
        alive+=("${pid}")
      fi
    done
    if (( ${#alive[@]} == 0 )); then
      log "All recorded training PIDs have exited."
      return
    fi
    log "Still running: ${alive[*]}"
    sleep "${POLL_SECONDS}"
  done
}

query_gpu_pids() {
  timeout 20s nvidia-smi -i "${GPU_IDS}" \
    --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | awk '$1 ~ /^[0-9]+$/ {print $1}' \
    | sort -u \
    | paste -sd, -
}

wait_for_idle_gpus() {
  local gpu_pids
  local idle_checks=0

  log "Waiting for GPUs ${GPU_IDS} to have no compute processes."
  while (( idle_checks < 2 )); do
    if gpu_pids="$(query_gpu_pids)"; then
      if [[ -z "${gpu_pids}" ]]; then
        idle_checks=$((idle_checks + 1))
        log "GPUs are idle (${idle_checks}/2 confirmations)."
      else
        idle_checks=0
        log "GPU compute processes still present: ${gpu_pids}"
      fi
    else
      idle_checks=0
      log "nvidia-smi query failed; keeping the job queued."
    fi
    if (( idle_checks < 2 )); then
      sleep "${POLL_SECONDS}"
    fi
  done
}

is_memory_failure() {
  local status="$1"
  local attempt_log="$2"

  if [[ "${status}" == "137" ]]; then
    return 0
  fi
  rg -qi \
    'out of memory|resource_exhausted|cuda_error_out_of_memory|failed to allocate|allocation failed|oom' \
    "${attempt_log}"
}

run_attempt() {
  local batch_size="$1"
  local train_steps="$2"
  local exp_name="memer_wa01_bs${batch_size}_${train_steps}steps_${RUN_TAG}"
  local attempt_log="${LOG_DIR}/${exp_name}.log"
  local status

  log "Starting ${exp_name} on GPUs ${GPU_IDS}."
  log "Training log: ${attempt_log}"

  set +e
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
  BATCH_SIZE="${batch_size}" \
  NUM_TRAIN_STEPS="${train_steps}" \
  EXP_NAME="${exp_name}" \
  COMPUTE_NORM_STATS=false \
  RESUME=false \
  OVERWRITE=false \
  PYTHONUNBUFFERED=1 \
    "${TRAIN_SCRIPT}" "${CONFIG_NAME}" 2>&1 | tee -a "${attempt_log}"
  status="${PIPESTATUS[0]}"
  set -e

  if [[ "${status}" == "0" ]]; then
    log "Training completed successfully: ${exp_name}"
    return 0
  fi

  log "Training exited with status ${status}: ${exp_name}"
  if is_memory_failure "${status}" "${attempt_log}"; then
    log "Detected an OOM/memory allocation failure; trying the next batch size."
    return 10
  fi

  log "Failure was not identified as OOM. Stopping instead of hiding the error."
  return 20
}

wait_for_initial_processes
wait_for_idle_gpus

if run_attempt 96 30000; then
  exit 0
else
  status=$?
  if [[ "${status}" != "10" ]]; then
    exit "${status}"
  fi
fi

if run_attempt 80 35000; then
  exit 0
else
  status=$?
  if [[ "${status}" != "10" ]]; then
    exit "${status}"
  fi
fi

if run_attempt 64 45000; then
  exit 0
else
  status=$?
  if [[ "${status}" == "10" ]]; then
    log "All three batch-size attempts ran out of memory."
  fi
  exit "${status}"
fi
