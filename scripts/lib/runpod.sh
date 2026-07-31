# shellcheck shell=bash
# Sourced, never executed. Everything that talks to RunPod lives here; the flow lives in
# scripts/start_runpod.sh.
#
# Two APIs, deliberately:
#   REST     rest.runpod.io/v1   pod lifecycle. Not runpodctl: `pod create --docker-args`
#                                returns success and silently drops the field, and the
#                                start command is the one thing we cannot get wrong
#                                (docs/launcher-hardening.md, points 1 and 5).
#   GraphQL  api.runpod.io       the GPU catalogue with price and stock status. The REST
#                                API has no equivalent endpoint.
#
# Written for bash 3.2, the /bin/bash macOS ships: no mapfile, no associative arrays.

RUNPOD_REST="https://rest.runpod.io/v1"
RUNPOD_GRAPHQL="https://api.runpod.io/graphql"

rp_preflight() {
    if [ -z "${RUNPOD_API_KEY:-}" ]; then
        echo "❌ RUNPOD_API_KEY is unset. It lives in .envrc -- run 'direnv allow'." >&2
        return 1
    fi
    local tool
    for tool in curl jq ssh git; do
        command -v "$tool" >/dev/null || { echo "❌ $tool not found on PATH" >&2; return 1; }
    done
}

# Body on stdout, non-zero return on anything but 2xx. Callers handle the return value
# themselves: a failing create or start is a decision point, not an abort (point 11).
#
# The status code goes through a file, not a variable. Nearly every call site captures the
# body with `$(...)`, which runs rp_api in a subshell, and a subshell cannot hand a
# variable back -- so a plain RP_CODE would report the code of some *earlier* call in
# exactly the error messages that need to be right.
RP_CODE_FILE="${TMPDIR:-/tmp}/runpod-http-code.$$"
rp_code() { cat "$RP_CODE_FILE" 2>/dev/null || echo '?'; }

# `endpoint`, not `path`: this file gets sourced, and in zsh `path` is the array tied to
# $PATH -- assigning to it wipes every command out of the shell.
rp_api() {
    local method="$1" endpoint="$2" body="${3:-}" out code
    out="$(mktemp "${TMPDIR:-/tmp}/runpod.XXXXXX")"
    local args
    args=(-sS -o "$out" -w '%{http_code}' -X "$method" "$RUNPOD_REST$endpoint"
          -H "Authorization: Bearer $RUNPOD_API_KEY")
    [ -n "$body" ] && args=("${args[@]}" -H 'Content-Type: application/json' -d "$body")
    code="$(curl "${args[@]}")"
    printf '%s' "$code" > "$RP_CODE_FILE"
    cat "$out"
    rm -f "$out"
    case "$code" in 2*) return 0 ;; *) return 1 ;; esac
}

rp_graphql() {
    curl -sS -X POST "$RUNPOD_GRAPHQL?api_key=$RUNPOD_API_KEY" \
        -H 'Content-Type: application/json' -d "$1"
}

# Secure cloud only, and never defaulted: Community Cloud runs on third-party hosts outside
# RunPod's audited GDPR boundary, and this pipeline puts Lebenshilfe source texts and live
# S3 credentials on the pod (point 10).
#
# What matters is VRAM in total, not per card: 1x RTX PRO 6000 (96 GB) and 2x L40S (2x48)
# are the same offer for this pipeline, and src-train/train.py runs Gemma at any GPU count.
# So this is called once per candidate count and the caller merges the results.
#
# Prints TSV `price<TAB>count<TAB>vram per card<TAB>total vram<TAB>stock<TAB>gpu type id`.
# The price is the total for gpu_count cards, which is what the catalogue quotes.
rp_gpu_candidates() {
    local gpu_count="$1" min_total_vram="$2" max_price="$3" allow_json="$4" query resp
    query="$(jq -nc --argjson c "$gpu_count" '{
        query: "query($i: GpuLowestPriceInput) { gpuTypes { id memoryInGb secureCloud lowestPrice(input: $i) { uninterruptablePrice stockStatus } } }",
        variables: { i: { gpuCount: $c, secureCloud: true } }
    }')"
    resp="$(rp_graphql "$query")"
    if [ "$(jq -r 'has("errors")' <<<"$resp")" = "true" ]; then
        echo "❌ GPU catalogue query failed:" >&2
        jq -r '.errors' <<<"$resp" >&2
        return 1
    fi
    # A null stockStatus means the type is listed but nothing is rentable right now.
    jq -r --argjson n "$gpu_count" --argjson v "$min_total_vram" --argjson p "$max_price" \
          --argjson allow "$allow_json" '
        [ .data.gpuTypes[]
          | select(.secureCloud
                   and (.id as $gid | $allow | index($gid))
                   and (.memoryInGb * $n) >= $v
                   and .lowestPrice.uninterruptablePrice != null
                   and .lowestPrice.uninterruptablePrice <= $p
                   and .lowestPrice.stockStatus != null) ]
        | sort_by(.lowestPrice.uninterruptablePrice)[]
        | "\(.lowestPrice.uninterruptablePrice)\t\($n)\t\(.memoryInGb)\t\(.memoryInGb * $n)\t\(.lowestPrice.stockStatus)\t\(.id)"
    ' <<<"$resp"
}

# The public TCP port for a private port, from GraphQL `runtime.ports`.
#
# REST's `portMappings` is a flat {privatePort: publicPort} map, which cannot represent
# 22/tcp and 22/udp at the same time -- and the templates expose both. The UDP entry wins,
# so REST reported 15304 while sshd was listening on 15303, and the launcher spent five
# minutes probing a port nothing was bound to before giving up. runtime.ports carries the
# protocol, so it is the only source that can answer this correctly.
rp_tcp_port() {
    local pod_id="$1" private="${2:-22}" query resp
    query="$(jq -nc --arg id "$pod_id" \
        '{query: ("query { pod(input:{podId:\"" + $id + "\"}) { runtime { ports { privatePort publicPort type } } } }")}')"
    resp="$(rp_graphql "$query")"
    jq -r --argjson p "$private" '
        (.data.pod.runtime.ports // [])
        | map(select(.privatePort == $p and ((.type // "") | ascii_downcase) == "tcp"))
        | (.[0].publicPort // empty)' <<<"$resp" 2>/dev/null
}

# Poll until the pod is up AND reachable. desiredStatus alone is not enough: it flips to
# RUNNING before the machine has published an IP and a port mapping for 22.
# Sets RP_SSH_HOST and RP_SSH_PORT.
rp_wait_running() {
    local pod_id="$1" timeout="${2:-900}" waited=0 pod status ip port
    while [ "$waited" -lt "$timeout" ]; do
        pod="$(rp_api GET "/pods/$pod_id")" || true
        status="$(jq -r '.desiredStatus // "?"' <<<"$pod" 2>/dev/null)"
        ip="$(jq -r '.publicIp // ""' <<<"$pod" 2>/dev/null)"
        # GraphQL first, REST only as a fallback -- see rp_tcp_port.
        port="$(rp_tcp_port "$pod_id")"
        [ -n "$port" ] || port="$(jq -r '(.portMappings // {})."22" // ""' <<<"$pod" 2>/dev/null)"
        if [ "$status" = "RUNNING" ] && [ -n "$ip" ] && [ -n "$port" ]; then
            printf '\r  pod %s is up: %s:%s%-20s\n' "$pod_id" "$ip" "$port" ""
            RP_SSH_HOST="$ip"
            RP_SSH_PORT="$port"
            return 0
        fi
        printf '\r  waiting for %s: status=%s ip=%s ssh=%s (%ss)   ' \
            "$pod_id" "$status" "${ip:-–}" "${port:-–}" "$waited"
        sleep 5
        waited=$((waited + 5))
    done
    echo
    echo "❌ pod $pod_id was not reachable within ${timeout}s" >&2
    return 1
}

rp_wait_stopped() {
    local pod_id="$1" timeout="${2:-300}" waited=0 status
    while [ "$waited" -lt "$timeout" ]; do
        status="$(rp_api GET "/pods/$pod_id" | jq -r '.desiredStatus // "?"')"
        [ "$status" != "RUNNING" ] && { printf '\r  pod %s stopped%-20s\n' "$pod_id" ""; return 0; }
        printf '\r  waiting for %s to stop: status=%s (%ss)   ' "$pod_id" "$status" "$waited"
        sleep 5
        waited=$((waited + 5))
    done
    echo
    echo "❌ pod $pod_id did not stop within ${timeout}s" >&2
    return 1
}

# Our own sshd, not RunPod's proxy: the proxy discards remote commands, so no automation
# can tail a log or run a diagnostic through it (point 3). Pod IPs get recycled across
# machines, so a known_hosts entry would only ever produce a false alarm.
# RP_SSH_KEY pins the identity to the key we actually installed on the pod. Without it ssh
# offers every key in ~/.ssh and everything the agent holds, in its own order, and sshd
# closes the connection after MaxAuthTries (6) -- so a machine with a handful of keys fails
# to log in to a pod whose authorized_keys is perfectly correct, while a machine with no
# agent connects first try.
rp_ssh() {
    local host="$1" port="$2"
    shift 2
    local id
    id=()
    [ -n "${RP_SSH_KEY:-}" ] && [ -f "$RP_SSH_KEY" ] && id=(-i "$RP_SSH_KEY" -o IdentitiesOnly=yes)
    # BatchMode: never stop on an interactive prompt. Everything here authenticates with a
    # key, and a hidden passphrase prompt is indistinguishable from a hung connection.
    ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR \
        -o ConnectTimeout=10 -o ServerAliveInterval=30 -o BatchMode=yes \
        "${id[@]}" -p "$port" "root@$host" "$@"
}

# Templates carry the image, the disk and volume sizes, the ports, and the secrets. Pick one
# by id, or by matching the mode against the account's `begleit-*` templates. Prints the
# whole template object; prompts only when the name alone cannot decide.
rp_find_template() {
    local wanted_id="$1" mode="$2" all matches count name id
    all="$(rp_api GET /templates)" || {
        echo "❌ could not list templates: $all" >&2; return 1; }

    if [ -n "$wanted_id" ]; then
        matches="$(jq -c --arg id "$wanted_id" '[ .[] | select(.id == $id) ]' <<<"$all")"
        [ "$(jq -r 'length' <<<"$matches")" -eq 1 ] || {
            echo "❌ no template with id '$wanted_id'" >&2; return 1; }
        jq -c '.[0]' <<<"$matches"
        return 0
    fi

    # The mode has to be a delimited word. A plain substring match puts every template in
    # every bucket: "train" occurs inside "begleit-app-training", and "eval" inside
    # "evaluate-separately", so both modes matched all four templates.
    # Serverless templates share the endpoint but cannot run a pod.
    matches="$(jq -c --arg m "$mode" '[ .[]
        | select((.isServerless // false) | not)
        | select(.name | ascii_downcase | test("begleit"))
        | select(.name | ascii_downcase | test("(^|[_-])" + $m + "([_-]|$)")) ]' <<<"$all")"
    count="$(jq -r 'length' <<<"$matches")"

    if [ "$count" -eq 0 ]; then
        echo "❌ no '$mode' template whose name contains 'begleit'. Set TEMPLATE_ID=..." >&2
        jq -r '.[] | "     \(.id)  \(.name)"' <<<"$all" >&2
        return 1
    fi
    if [ "$count" -eq 1 ]; then
        jq -c '.[0]' <<<"$matches"
        return 0
    fi

    echo "==> $count '$mode' templates:" >&2
    jq -r 'to_entries[] | "  [\(.key + 1)] \(.value.name)  \(.value.imageName)"' <<<"$matches" >&2
    local choice=""
    read -r -p "  use which one? [number] " choice </dev/tty || true
    case "$choice" in
        ""|*[!0-9]*|0) echo "aborted." >&2; return 1 ;;
    esac
    jq -ce --argjson i "$((choice - 1))" '.[$i] // empty' <<<"$matches" \
        || { echo "❌ no option $choice" >&2; return 1; }
}

# The live API answers with `imageName` and leaves `image` null, while the published
# OpenAPI schema documents only `image`. Reading the documented field alone made every
# verification fail against a pod that was in fact configured correctly, so read both.
rp_pod_image() {
    jq -r '.imageName // .image // ""' <<<"$1" 2>/dev/null
}

# Never trust a write, verify with an independent read (point 7). Both known ways of losing
# the start command -- runpodctl dropping --docker-args, and PATCH on a running pod --
# report success first, and a pod that falls back to the image default is the exact failure
# that cost two hours to diagnose (point 1).
rp_verify_pod_config() {
    local pod_id="$1" want_image="$2" want_cmd="$3" want_entry="$4" attempt pod=""
    for attempt in 1 2 3; do
        pod="$(rp_api GET "/pods/$pod_id")" || true
        if [ "$(rp_pod_image "$pod")" = "$want_image" ] \
            && [ "$(jq -c '.dockerStartCmd // []' <<<"$pod" 2>/dev/null)" = "$want_cmd" ] \
            && [ "$(jq -c '.dockerEntrypoint // []' <<<"$pod" 2>/dev/null)" = "$want_entry" ]; then
            return 0
        fi
        sleep 5
    done
    echo "❌ pod $pod_id reports success but is not configured as asked:" >&2
    echo "     image        : $(rp_pod_image "$pod")" >&2
    echo "     entrypoint   : $(jq -c '.dockerEntrypoint // []' <<<"$pod")" >&2
    echo "     start command: $(jq -c '.dockerStartCmd // []' <<<"$pod" | cut -c1-70)" >&2
    echo "   It would run the image's default command. Stop it: runpodctl stop pod $pod_id" >&2
    return 1
}

# Point 25: an accidental pod should be visible in seconds, not minutes.
rp_cost_summary() {
    rp_api GET /pods | jq -r '
        [ .[] | select(.desiredStatus == "RUNNING") ]
        | "💰 running now: \(length) pod(s), $\((map(.costPerHr // 0) | add // 0) * 100 | round / 100)/hr"
    ' 2>/dev/null || true
}
