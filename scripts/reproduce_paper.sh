#!/usr/bin/env bash
# Reproduce the Page-EntroKV paper results end-to-end.
#
# Stages:
#   1. install  -- create the conda environment (environment.yml)
#   2. test     -- run the unit + property test suite
#   3. niah     -- Needle-In-A-Haystack benchmark
#   4. longbench-- LongBench-style benchmark
#   5. ruler    -- RULER-style synthetic benchmark
#   6. all      -- everything, in order
#
# Usage:
#   bash scripts/reproduce_paper.sh [stage] [--data DIR]
#
# Environment variables:
#   ENTROKV_DATA_DIR   directory with {niah,longbench,ruler}.jsonl corpora
#   ENTROKV_BACKEND    "synthetic" (default) or "hf"
#   ENTROKV_MODEL      HF model id when ENTROKV_BACKEND=hf

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="${1:-all}"
DATA_DIR="${ENTROKV_DATA_DIR:-$ROOT_DIR/data}"
BACKEND="${ENTROKV_BACKEND:-synthetic}"
MODEL="${ENTROKV_MODEL:-}"
OUT_DIR="$ROOT_DIR/results"
mkdir -p "$OUT_DIR"

log() { printf '\033[1;34m[entrokv]\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m[entrokv] %s\033[0m\n' "$*" >&2; exit 1; }

run_python() {
  if command -v conda >/dev/null 2>&1 && conda env list 2>/dev/null | grep -q "entrokv"; then
    conda run -n entrokv python "$@"
  else
    python "$@"
  fi
}

require_data() {
  if [[ ! -f "$1" ]]; then
    fail "missing corpus: $1 (set ENTROKV_DATA_DIR or populate data/)"
  fi
}

install_env() {
  log "installing conda environment 'entrokv'"
  conda env create -f "$ROOT_DIR/environment.yml" -n entrokv
  log "installing package (editable)"
  conda run -n entrokv pip install -e "$ROOT_DIR"
}

run_tests() {
  log "running test suite"
  run_python -m pytest "$ROOT_DIR/tests" -q
}

run_niah() {
  local data="$DATA_DIR/niah.jsonl"
  require_data "$data"
  log "running NIAH benchmark ($BACKEND)"
  run_python "$ROOT_DIR/scripts/run_niah.py" \
    --data "$data" --backend "$BACKEND" ${MODEL:+--model "$MODEL"} \
    --output "$OUT_DIR/niah_report.json"
}

run_longbench() {
  local data="$DATA_DIR/longbench.jsonl"
  require_data "$data"
  log "running LongBench benchmark ($BACKEND)"
  run_python "$ROOT_DIR/scripts/run_longbench.py" \
    --data "$data" --backend "$BACKEND" ${MODEL:+--model "$MODEL"} \
    --output "$OUT_DIR/longbench_report.json"
}

run_ruler() {
  local data="$DATA_DIR/ruler.jsonl"
  require_data "$data"
  log "running RULER benchmark ($BACKEND)"
  run_python "$ROOT_DIR/scripts/run_ruler.py" \
    --data "$data" --backend "$BACKEND" ${MODEL:+--model "$MODEL"} \
    --output "$OUT_DIR/ruler_report.json"
}

case "$STAGE" in
  install)   install_env ;;
  test)      run_tests ;;
  niah)      run_niah ;;
  longbench) run_longbench ;;
  ruler)     run_ruler ;;
  all)
    run_tests
    run_niah
    run_longbench
    run_ruler
    log "all stages complete; reports in $OUT_DIR"
    ;;
  *)
    fail "unknown stage '$STAGE' (expected: install|test|niah|longbench|ruler|all)"
    ;;
esac
