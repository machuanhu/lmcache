#!/bin/bash

set -euo pipefail

BASE_DIR="/root/lmcache/benchmarks/multi-round-qa"
TEST_SCRIPT="${BASE_DIR}/test.sh"
LOG_DIR="${BASE_DIR}/logs"
mkdir -p "${LOG_DIR}"

timestamp="$(date +%Y%m%d_%H%M%S)"
declare -a QPS_VALUES=(8 16 32 48 64 80)
declare -a BATCH_CONVERSATIONS_VALUES=(1 5)

echo "Starting QPS sweep with different batch-conversations values"

for batch_conv in "${BATCH_CONVERSATIONS_VALUES[@]}"; do
    batch_log_dir="${LOG_DIR}/batch_${batch_conv}"
    mkdir -p "${batch_log_dir}"
    SUMMARY_LOG="${batch_log_dir}/qps_summary_${timestamp}.log"
    
    echo "=========================================="
    echo "Testing with batch-conversations=${batch_conv}"
    echo "Summaries will be written to ${SUMMARY_LOG}"
    echo "=========================================="
    printf "" > "${SUMMARY_LOG}"
    
    for qps in "${QPS_VALUES[@]}"; do
        echo "Running test for QPS=${qps}, batch-conversations=${batch_conv}..."

        run_output="$(bash "${TEST_SCRIPT}" "${qps}" "${batch_conv}")"
        printf "%s\n" "${run_output}"

        summary=$(printf "%s\n" "${run_output}" | awk '
            /==================== Performance summary ======================/ {start=NR}
            {lines[NR]=$0}
            END {
                if (start > 0) {
                    for (i = start; i <= NR; i++) {
                        print lines[i]
                    }
                }
            }')

        {
            echo "================ QPS ${qps} ================"
            if [[ -n "${summary}" ]]; then
                echo "${summary}"
            else
                echo "No performance summary found for QPS ${qps}"
            fi
            echo
        } >> "${SUMMARY_LOG}"
    done
    
    echo "QPS sweep complete for batch-conversations=${batch_conv}. Summaries saved to ${SUMMARY_LOG}"
    echo
done

echo "All QPS sweeps complete!"


