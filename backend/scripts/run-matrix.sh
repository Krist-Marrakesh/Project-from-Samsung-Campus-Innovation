#!/usr/bin/env bash
# Slice 1 — research benchmark harness driver.
#
# Posts to /bench/matrix-from-presets, saves the JSON, and pivots it into a
# flat CSV (one row per scenario × stage). The CSV is the artefact that
# becomes the trade-off graphs in the research report.
#
# Usage:
#   scripts/run-matrix.sh [PRESET] [WARMUP] [RUNS] [OUTPUT_DIR]
#
# Defaults:
#   PRESET     = research          (curated subset; see BenchmarkPresets.java)
#                Pass 'full' for the Cartesian product (~30 minutes laptop run).
#   WARMUP     = 3                 (enough for HotSpot tiered compilation)
#   RUNS       = 10                (statistically meaningful percentiles)
#   OUTPUT_DIR = data/bench/<ts>
#
# Prereqs:
#   * Backend running on :8080 with FRACTALOV_BENCH_ENABLED=true.
#   * jq on PATH for CSV pivot.
#
# Exit codes:
#   0 success, 1 prereq missing, 2 backend rejected the request

set -euo pipefail

PRESET="${1:-research}"
WARMUP="${2:-3}"
RUNS="${3:-10}"
TS="$(date +%Y%m%d-%H%M%S)"
OUTPUT_DIR="${4:-data/bench/${TS}}"
BACKEND_URL="${FRACTALOV_BACKEND_URL:-http://localhost:8080}"

command -v jq >/dev/null 2>&1 || {
    echo "jq is required for CSV pivot — install via 'brew install jq' or your package manager" >&2
    exit 1
}
command -v curl >/dev/null 2>&1 || {
    echo "curl is required" >&2
    exit 1
}

mkdir -p "${OUTPUT_DIR}"
JSON_PATH="${OUTPUT_DIR}/matrix.json"
CSV_PATH="${OUTPUT_DIR}/matrix.csv"
META_PATH="${OUTPUT_DIR}/run-meta.json"

# Capture environment metadata so the CSV is interpretable months from now.
cat > "${META_PATH}" <<EOF
{
  "timestamp": "${TS}",
  "preset": "${PRESET}",
  "warmup": ${WARMUP},
  "runs": ${RUNS},
  "backend_url": "${BACKEND_URL}",
  "host_os": "$(uname -s)",
  "host_arch": "$(uname -m)",
  "host_cores": "$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo unknown)"
}
EOF

echo "[matrix] preset=${PRESET} warmup=${WARMUP} runs=${RUNS} → ${OUTPUT_DIR}"

REQUEST_BODY=$(cat <<EOF
{
  "preset": "${PRESET}",
  "warmup": ${WARMUP},
  "runs": ${RUNS},
  "includeRawSamples": false
}
EOF
)

HTTP_STATUS=$(curl -sS -o "${JSON_PATH}" -w "%{http_code}" \
    -X POST "${BACKEND_URL}/bench/matrix-from-presets" \
    -H 'Content-Type: application/json' \
    -d "${REQUEST_BODY}")

if [ "${HTTP_STATUS}" != "200" ]; then
    echo "[matrix] backend returned HTTP ${HTTP_STATUS}" >&2
    cat "${JSON_PATH}" >&2 || true
    exit 2
fi

# CSV pivot — one row per (scenario, stage). Tags are flattened into columns
# so a research script can group by family / resolution / etc. directly.
jq -r '
  ["matrixId","scenario","family","widthPx","heightPx","maxIter","ssaa","colorMode","palette",
   "stage","runs","minMs","maxMs","meanMs","p50Ms","p90Ms","p99Ms"] as $hdr
  | $hdr,
    (.matrix.results[] as $r
      | ["render","colorize","encode","total"]
      | map({stage: ., stats: ($r.breakdown[.])})
      | .[] | [
          $r.scenario | ($r.scenario),
          $r.scenario,
          $r.tags.family,
          $r.tags.widthPx,
          $r.tags.heightPx,
          $r.tags.maxIter,
          $r.tags.ssaa,
          $r.tags.colorMode,
          $r.tags.palette,
          .stage,
          .stats.runs,
          .stats.minMs,
          .stats.maxMs,
          .stats.meanMs,
          .stats.p50Ms,
          .stats.p90Ms,
          .stats.p99Ms
        ]
    )
  | @csv
' < "${JSON_PATH}" > "${CSV_PATH}"

# Repair the header injection in the jq filter above — the matrixId comes from
# the top level, not per-row. Simpler to fix here than complicate the filter.
TMP_CSV="$(mktemp)"
MATRIX_ID="$(jq -r '.matrix.matrixId' < "${JSON_PATH}")"
{
    head -n1 "${CSV_PATH}"
    tail -n +2 "${CSV_PATH}" | sed "s/^\"[^\"]*\"/\"${MATRIX_ID}\"/"
} > "${TMP_CSV}"
mv "${TMP_CSV}" "${CSV_PATH}"

ROWS=$(($(wc -l < "${CSV_PATH}") - 1))
echo "[matrix] wrote ${ROWS} rows to ${CSV_PATH}"
echo "[matrix] matrixId=${MATRIX_ID}"
echo "[matrix] preview:"
head -n6 "${CSV_PATH}"
