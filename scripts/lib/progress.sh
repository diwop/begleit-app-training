# shellcheck shell=bash
# Sourced. Reports model-download progress, which is otherwise completely invisible.
#
# huggingface_hub draws tqdm bars with carriage returns, and none of that survives
# redirection into a log file -- so a 51 GB download looks identical to a hung process for
# as long as it takes. Bytes on disk are the only honest signal: the hub client writes each
# blob as `<name>.incomplete` and renames it when the transfer finishes.
#
# The reporter runs in the background, prints only while the cache is actually growing, and
# stays silent afterwards, so it costs one line every DOWNLOAD_PROGRESS_INTERVAL seconds
# during the download and nothing at all the rest of the time.

# progress_start [logfile] -- sets PROGRESS_PID.
progress_start() {
    local log="${1:-}"
    local dir="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"
    local interval="${DOWNLOAD_PROGRESS_INTERVAL:-30}"

    mkdir -p "$dir" 2>/dev/null || return 0
    (
        local prev=0 now delta inflight line
        while :; do
            sleep "$interval"
            # -sk, not -sb: BSD du on macOS has no -b, and this file runs on both platforms.
            now="$(du -sk "$dir" 2>/dev/null | tail -1 | cut -f1)"
            case "$now" in ''|*[!0-9]*) continue ;; esac
            delta=$((now - prev))
            prev="$now"
            [ "$delta" -gt 0 ] || continue
            inflight="$(find "$dir" -name '*.incomplete' 2>/dev/null | wc -l | tr -d ' ')"
            line="$(awk -v k="$now" -v d="$delta" -v s="$interval" -v n="$inflight" \
                'BEGIN{printf "  ⬇ model cache %.1f GiB (+%.0f MB/s, %d file(s) in flight)", \
                       k/1048576, d/1024/s, n}')"
            if [ -n "$log" ]; then
                printf '%s\n' "$line" | tee -a "$log"
            else
                printf '%s\n' "$line"
            fi
        done
    ) &
    PROGRESS_PID=$!
}

progress_stop() {
    [ -n "${PROGRESS_PID:-}" ] && kill "$PROGRESS_PID" 2>/dev/null
    PROGRESS_PID=""
    return 0
}
