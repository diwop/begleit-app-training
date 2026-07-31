# shellcheck shell=bash
# Sourced. Publishes the live progress of a running job to S3, so a training run can be
# watched from a laptop without SSH into the pod.
#
# src-train/train.py syncs to S3 exactly once, after the whole pipeline finishes. That is
# the authoritative copy and it stays -- but until it happens the bucket holds nothing at
# all, so there is no way to see how a run is going, and a pod that dies at hour three
# takes its TensorBoard history with it.
#
# Deliberately NOT the whole output directory: that contains `checkpoint-*/`, which holds
# optimizer state measured in gigabytes. Re-uploading it every minute would spend the
# uplink on data nobody is watching. Only the TensorBoard events and the run log go up.
#
# The destination is a fixed `live/<config>/` prefix rather than the run-id prefix
# train.py uses, so the address to point TensorBoard at is the same for every run:
#
#     aws s3 sync s3://$S3_BUCKET/live/train-gemma4/runs ./runs && tensorboard --logdir ./runs
#
# Each run overwrites it; the durable per-run copy is what train.py writes at the end.

# s3_sync_start <output_dir> <config_name> [logfile] -- sets S3_SYNC_PID.
s3_sync_start() {
    local dir="$1" config_name="$2" log="${3:-}"
    S3_SYNC_PID=""

    [ -n "${S3_BUCKET:-}" ] || return 0
    command -v aws >/dev/null || {
        echo "ℹ️  no aws CLI; live progress will not be published to S3."
        return 0
    }

    local interval="${S3_SYNC_INTERVAL:-60}"
    local dest="s3://${S3_BUCKET}/live/${config_name}"
    echo "📡 Publishing live progress every ${interval}s to ${dest}"
    (
        while :; do
            sleep "$interval"
            # --delete is deliberately absent: a half-written event file must never cause
            # the previous good one to be removed from the bucket.
            [ -d "$dir/runs" ] && aws s3 sync "$dir/runs" "$dest/runs" \
                --only-show-errors 2>/dev/null
            [ -n "$log" ] && [ -f "$log" ] && aws s3 cp "$log" "$dest/$(basename "$log")" \
                --only-show-errors 2>/dev/null
        done
    ) &
    S3_SYNC_PID=$!
}

s3_sync_stop() {
    [ -n "${S3_SYNC_PID:-}" ] && kill "$S3_SYNC_PID" 2>/dev/null
    S3_SYNC_PID=""
    return 0
}
