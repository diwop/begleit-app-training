# shellcheck shell=bash
# Sourced. The shared tail of train.sh and eval.sh: publish the log, then decide whether
# to keep the machine alive. All of it is a no-op on a laptop.

finish() {
    local exit_code="$1" log_file="$2" phase="$3"
    local timestamp
    timestamp="$(date +"%Y%m%d_%H%M%S")"

    if [ -n "${S3_BUCKET:-}" ]; then
        echo "S3_BUCKET is set to '${S3_BUCKET}'. Copying logs..."
        if "$PY_EVAL" -u -c "import boto3,sys; boto3.client('s3').upload_file(sys.argv[1], sys.argv[2], sys.argv[3])" \
                "$log_file" "$S3_BUCKET" "logs/${timestamp}_${phase}.log"; then
            echo "Logs copied to S3 as logs/${timestamp}_${phase}.log."
        else
            echo "WARNING: Could not copy logs to S3."
            [ "$IS_LOCAL" = "0" ] && sleep 60
        fi
    fi

    if [ "$exit_code" -eq 0 ]; then
        echo "✅ ${phase} completed successfully!"
    else
        echo "[FATAL] ${phase} failed with exit code ${exit_code}."
        # Give a human time to attach and read the container logs before the pod dies.
        [ "$IS_LOCAL" = "0" ] && sleep 60
    fi

    if [ "$IS_LOCAL" = "1" ]; then
        return "$exit_code"
    fi

    if [ "${KEEP_ALIVE:-false}" = "true" ]; then
        echo "KEEP_ALIVE flag is active. Bypassing RunPod shutdown API."
    elif [ "$IS_RUNPOD" = "1" ]; then
        echo "RunPod environment detected. Shutting down pod to save costs..."
        curl -s --request POST "https://api.runpod.io/graphql" \
            --header "Authorization: Bearer $RUNPOD_API_KEY" \
            --header "Content-Type: application/json" \
            --data "{\"query\": \"mutation { podStop(input: {podId: \\\"$RUNPOD_POD_ID\\\"}) { id } }\"}"
    else
        # A plain Linux+CUDA host: nothing to shut down, and sleeping forever would just
        # wedge someone's terminal.
        echo "Not RunPod; leaving the host alone."
    fi
    return "$exit_code"
}
