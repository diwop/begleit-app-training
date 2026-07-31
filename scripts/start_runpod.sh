#!/bin/bash
# Put a RunPod pod to work on the train or eval pipeline, then tail its log.
#
#   bash scripts/start_runpod.sh train
#   bash scripts/start_runpod.sh eval
#
# Reuses an existing GPU pod when there is one; otherwise offers the cheapest secure-cloud
# GPU that clears the VRAM and price limits and creates it only after a yes. The pod's
# start command does the actual work -- it fetches scripts/launch.sh from the branch and
# runs it, so this script never needs a second channel to kick the run off.
#
# Every rule enforced below comes from a failure recorded in docs/launcher-hardening.md;
# the point numbers in the comments refer to it.
set -e

cd "$(dirname "$0")/.."
# shellcheck source=lib/runpod.sh
source scripts/lib/runpod.sh

usage() {
    echo "usage: bash scripts/start_runpod.sh train|eval [attach] [--keep-alive]" >&2
    echo "  attach        tail a run that is already going, change nothing" >&2
    echo "  --keep-alive  leave the pod running after the run, for debugging" >&2
}

MODE="${1:-}"
case "$MODE" in
    train)                MODE=train; IMAGE_KEY=train_image ;;
    eval|evaluation|test) MODE=eval;  IMAGE_KEY=eval_image ;;
    *) usage; exit 1 ;;
esac
shift

# `attach` only tails an existing run. Without it, the sole way back to a running job was to
# run this script again, which restarts the container and kills the very run you wanted to
# look at.
ATTACH=0
KEEP_ALIVE_FLAG=0
for arg in "$@"; do
    case "$arg" in
        attach)                        ATTACH=1 ;;
        --keep-alive|--keep-running)   KEEP_ALIVE_FLAG=1 ;;
        *) echo "❌ unknown argument: $arg" >&2; usage; exit 1 ;;
    esac
done

# Always sent explicitly, never left to the pod's history. Reuse merges our env over the
# pod's existing env, so omitting this would let a pod created back when the template said
# KEEP_ALIVE=true keep running forever -- the default has to be enforced, not just declared.
# An exported KEEP_ALIVE=true still counts as an explicit opt-in.
if [ "$KEEP_ALIVE_FLAG" = "1" ] || [ "${KEEP_ALIVE:-}" = "true" ]; then
    KEEP_ALIVE=true
else
    KEEP_ALIVE=false
fi

# VRAM in total, not per card, and not a fixed number of cards. Runs that worked: 1x RTX
# PRO 6000 (96 GB), 2x L40S (2x48), and 1x H100 SXM (80) as the pricier option. Gemma runs
# at any GPU count -- src-train/train.py only requires 8 for Mistral. MAX_USD_PER_HR is the
# price for the whole pod, which is what RunPod's catalogue quotes.
GPU_COUNTS="${GPU_COUNTS:-1 2}"
MIN_TOTAL_VRAM_GB="${MIN_TOTAL_VRAM_GB:-80}"
# Two ceilings, not one. A single card avoids cross-GPU communication entirely -- no NCCL,
# no P2P workarounds, no tensor parallelism -- so it wins over a cheaper 2-card pod as long
# as it stays under PREFER_USD_PER_HR. MAX_USD_PER_HR is the last resort above that.
PREFER_USD_PER_HR="${PREFER_USD_PER_HR:-2}"
MAX_USD_PER_HR="${MAX_USD_PER_HR:-3}"

# An explicit allow-list, because "big enough and cheap enough" picks the wrong card: the
# cheapest offers that clear 80 GB are 2x A40 at $0.88 and 2x RTX A6000 at $1.06, and both
# are Ampere. FP8 needs compute capability 8.9 or newer -- Ada, Hopper, Blackwell -- and
# the deployment target is the FP8 base (TODO 2), so an Ampere pod cannot test what we ship.
#
# Only cards that can reach 80 GB in one or two of them are listed; smaller Ada/Blackwell
# parts (L4, RTX 4090, RTX 5090, RTX PRO 4000/4500) would need four or more. RunPod's ids
# are exact strings, so these are the API ids, not marketing names.
FP8_GPUS=(
    # Ada Lovelace, sm_89 -- 48 GB each, so two cards
    "NVIDIA L40S"
    "NVIDIA L40"
    "NVIDIA RTX 6000 Ada Generation"
    # Blackwell, sm_120
    "NVIDIA RTX PRO 5000 Blackwell"                      # 48 GB, two cards
    "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"  # 96 GB, single card
    "NVIDIA RTX PRO 6000 Blackwell Server Edition"       # 96 GB, single card
    "NVIDIA B200"                                        # 180 GB
    "NVIDIA B300 SXM6 AC"                                # 288 GB
    # Hopper, sm_90
    "NVIDIA H100 PCIe"
    "NVIDIA H100 80GB HBM3"                              # the SXM part
    "NVIDIA H100 NVL"
    "NVIDIA H200"
    "NVIDIA H200 NVL"
)
GPU_ALLOWLIST_JSON="$(printf '%s\n' "${FP8_GPUS[@]}" | jq -R . | jq -sc .)"
POD_NAME="${POD_NAME:-begleit-$MODE}"
REPO_URL="${REPO_URL:-https://github.com/diwop/begleit-app-training.git}"
BRANCH="${BRANCH:-$(git rev-parse --abbrev-ref HEAD)}"
# Backstop only, and deliberately close to the real runtimes: training takes ~15 min and an
# eval ~25 min (~45 with torch.compile), so 2h is generous while capping a stuck pod at ~$4
# instead of ~$24.
MAX_POD_HOURS="${MAX_POD_HOURS:-2}"

rp_preflight
trap 'rm -f "$RP_CODE_FILE"' EXIT

# ------------------------------------------------------------------------------ template
# Disk, volume, mount path, ports and -- above all -- the secrets come from the RunPod
# template, not from this script. HF_TOKEN and the AWS keys are stored there as
# `{{ RUNPOD_SECRET_* }}` references, which RunPod resolves for the template. We never copy
# those strings into the pod's own env: the documentation only describes them resolving at
# template level, so a copy could travel as literal text and fail silently.
TEMPLATE="$(rp_find_template "${TEMPLATE_ID:-}" "$MODE")" || exit 1
TEMPLATE_ID="$(jq -r '.id' <<<"$TEMPLATE")"
TEMPLATE_NAME="$(jq -r '.name' <<<"$TEMPLATE")"
TEMPLATE_IMAGE="$(jq -r '.imageName // ""' <<<"$TEMPLATE")"
TEMPLATE_ENV_KEYS="$(jq -c '(.env // {}) | keys' <<<"$TEMPLATE")"
VOLUME_MOUNT="$(jq -r '.volumeMountPath // "/app"' <<<"$TEMPLATE")"

# On the volume, not on the container disk: /var/log dies with the pod, and the pod stops
# itself the moment the run ends -- which deleted the log exactly when it was worth reading.
POD_LOG="$VOLUME_MOUNT/logs/pod-$MODE.log"

# The image is the one place the repo overrules the template. README.md is the declared
# source of truth and scripts/lib/platform.sh already refuses to run when the pinned vLLM
# release drifts from it; the eval template still names the SGLang image this project
# abandoned.
IMAGE="$(sed -n "s/^$IMAGE_KEY: *//p" README.md | head -1)"
[ -n "$IMAGE" ] || { echo "❌ no '$IMAGE_KEY' in the README.md front matter" >&2; exit 1; }

# The pod clones from GitHub, so an unpushed branch or commit simply is not there. This is
# the cheapest possible place to notice that -- the alternative is a $3/hr pod running
# yesterday's code.
# Test the output, not the exit code: a pipeline reports the exit code of `cut`, so
# `if ! REMOTE_SHA="$(git ls-remote ... | cut -f1)"` never fires and the guard is decorative.
REMOTE_SHA="$(git ls-remote origin "refs/heads/$BRANCH" 2>/dev/null | head -1 | cut -f1)"
if [ "$ATTACH" = "0" ] && [ -z "$REMOTE_SHA" ]; then
    echo "❌ no branch '$BRANCH' on origin. The pod clones from GitHub; push it first." >&2
    exit 1
fi
if [ "$ATTACH" = "0" ]; then
    if [ "$REMOTE_SHA" != "$(git rev-parse HEAD)" ]; then
        echo "⚠️  origin/$BRANCH is at ${REMOTE_SHA:0:7}, your HEAD at $(git rev-parse --short HEAD)."
    fi
    [ -n "$(git status --porcelain)" ] && echo "⚠️  uncommitted changes: the pod runs origin/$BRANCH, not your working tree."
fi

PUB_KEY_FILE="${SSH_PUBKEY:-$HOME/.ssh/id_ed25519.pub}"
[ -f "$PUB_KEY_FILE" ] || PUB_KEY_FILE="$(ls "$HOME"/.ssh/*.pub 2>/dev/null | head -1)"
if [ -z "$PUB_KEY_FILE" ] || [ ! -f "$PUB_KEY_FILE" ]; then
    echo "❌ no SSH public key found. Generate one: ssh-keygen -t ed25519" >&2
    exit 1
fi
PUBLIC_KEY="$(cat "$PUB_KEY_FILE")"
# Log in with the matching private key and nothing else -- see rp_ssh.
RP_SSH_KEY="${PUB_KEY_FILE%.pub}"

# Not a pre-flight abort, on purpose: gating is a property of the repo, not of the name --
# src-train/train.py makes the same call, and gemma-4-26b-a4b-it in fact downloaded without
# a token on 2026-07-31. Mistral still needs one.
if [ "$ATTACH" = "0" ] && [ -z "${HF_TOKEN:-}" ]; then
    echo "⚠️  HF_TOKEN is unset; a gated base model would fail to download."
fi

# ---------------------------------------------------------------------------- start cmd
# Never inherit the image default (point 1). The three images we use behave three different
# ways, and one of them -- vLLM's -- silently serves Qwen3-0.6B on 90% of the card.
read -r -d '' START_SCRIPT <<'POD_START' || true
set -x
export HOME="${HOME:-/root}"
mkdir -p /runner /var/log "$(dirname "@POD_LOG@")"

# The images disagree about what they ship: Axolotl has git and sshd, the vLLM image has
# neither, nor curl. Probe, install only the gap (point 2).
MISSING=""
for tool in curl git; do command -v "$tool" >/dev/null 2>&1 || MISSING="$MISSING $tool"; done
[ -x /usr/sbin/sshd ] || MISSING="$MISSING openssh-server"
if [ -n "$MISSING" ]; then
    apt-get update -qq && apt-get install -y -qq $MISSING
fi

# RunPod's proxy SSH discards remote commands, so reading logs needs a real sshd (point 3).
mkdir -p /run/sshd "$HOME/.ssh"
printf '%s\n' "$PUBLIC_KEY" > "$HOME/.ssh/authorized_keys"
chmod 700 "$HOME/.ssh"
chmod 600 "$HOME/.ssh/authorized_keys"
pgrep -x sshd >/dev/null 2>&1 || /usr/sbin/sshd

# Cost guard (point 24). RunPod has no stop-after, and scripts/lib/finish.sh only stops the
# pod if the pipeline actually reaches its end -- a start command that dies before that
# leaves an idle GPU billing until somebody happens to look.
setsid bash -c 'sleep @MAX_POD_SECONDS@
echo "[watchdog] @MAX_POD_HOURS@h reached, stopping the pod."
curl -s --request POST "https://api.runpod.io/graphql" \
    --header "Authorization: Bearer $RUNPOD_API_KEY" \
    --header "Content-Type: application/json" \
    --data "{\"query\": \"mutation { podStop(input: {podId: \\\"$RUNPOD_POD_ID\\\"}) { id } }\"}"' \
    </dev/null >>"@POD_LOG@" 2>&1 &

curl -fsSL "@LAUNCH_URL@" -o /runner/launch.sh

# Append, with a marker: the log now survives a stop, so keep the earlier runs too.
echo "===== @MODE@ run started $(date -u +%Y-%m-%dT%H:%M:%SZ) =====" >> "@POD_LOG@"

# setsid: the run must outlive both this start command and any SSH session (point 13).
setsid bash /runner/launch.sh </dev/null >>"@POD_LOG@" 2>&1 &

# Never exit. A finished pipeline has to leave the container reachable (point 4).
sleep infinity
POD_START

LAUNCH_URL="${REPO_URL%.git}"
LAUNCH_URL="${LAUNCH_URL/github.com/raw.githubusercontent.com}/$BRANCH/scripts/launch.sh"
START_SCRIPT="${START_SCRIPT//@LAUNCH_URL@/$LAUNCH_URL}"
START_SCRIPT="${START_SCRIPT//@POD_LOG@/$POD_LOG}"
START_SCRIPT="${START_SCRIPT//@MODE@/$MODE}"
START_SCRIPT="${START_SCRIPT//@MAX_POD_HOURS@/$MAX_POD_HOURS}"
START_SCRIPT="${START_SCRIPT//@MAX_POD_SECONDS@/$((MAX_POD_HOURS * 3600))}"
# The entrypoint has to be overridden as well, not just the command. dockerStartCmd becomes
# the container's CMD, and Docker APPENDS the CMD to the image's ENTRYPOINT -- so on the
# vLLM image, whose entrypoint is `vllm serve`, the pod ran
#     vllm serve bash -c "set -x; export HOME=..."
# and vLLM parsed this whole script as the value of --compilation-config, crash-looping
# forever while sshd never started. The Axolotl image has no entrypoint, which is why
# training worked and hid this completely.
START_ENTRYPOINT_JSON='["bash","-c"]'
START_CMD_JSON="$(jq -nc --arg s "$START_SCRIPT" '[$s]')"

# SSH sessions do not inherit the container env (point 12), so everything the pipeline
# reads has to be passed here. RUNPOD_API_KEY included: scripts/lib/finish.sh needs it to
# stop the pod when the run is done.
build_env_json() {
    local json='{}' key value
    for key in "$@"; do
        value="${!key:-}"
        [ -n "$value" ] || continue
        json="$(jq -c --arg k "$key" --arg v "$value" '. + {($k): $v}' <<<"$json")"
    done
    printf '%s' "$json"
}
# The last four are debugging knobs, unset in a normal run and simply not sent then.
# CUDA_LAUNCH_BLOCKING earns its place: CUDA reports errors asynchronously, so a fault in
# one kernel surfaces at the next synchronising call and the traceback points somewhere
# innocent. The 2026-07-31 crash blamed a 3-element fp32 matmul inside Gemma 4's rotary
# embedding, which is a symptom, not a cause. Setting it to 1 makes every launch
# synchronous -- much slower, and the only way to see where the fault actually is.
# VALIDATION_METRICS_* tune src-train/validation_metrics.py without editing a config.
ENV_JSON="$(build_env_json MODE BRANCH REPO_URL PUBLIC_KEY RUNPOD_API_KEY HF_TOKEN \
    S3_BUCKET AWS_DEFAULT_REGION AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY \
    KEEP_ALIVE TRAIN_CONFIG TP_SIZE SMOKE_BASE SMOKE_ADAPTER_S3 SMOKE_EAGER \
    CUDA_LAUNCH_BLOCKING TORCH_USE_CUDA_DSA VALIDATION_METRICS_OFF \
    VALIDATION_METRICS_SAMPLES VALIDATION_METRICS_MAX_TOKENS \
    ATTN_IMPLEMENTATION GEMMA4_HYBRID_ATTN EVAL_STRATEGY ROPE_DEBUG)"

echo
if [ "$ATTACH" = "1" ]; then
    echo "==> attaching to the $MODE log; nothing on the pod is changed"
    rp_cost_summary
else
echo "==> $MODE on RunPod"
echo "  template: $TEMPLATE_NAME ($TEMPLATE_ID)"
echo "            disk/volume/ports and the secrets ($(jq -r 'join(", ")' <<<"$TEMPLATE_ENV_KEYS")) come from it"
if [ -n "$TEMPLATE_IMAGE" ] && [ "$TEMPLATE_IMAGE" != "$IMAGE" ]; then
    echo "  ⚠️  the template names a different image; README's $IMAGE_KEY wins:"
    echo "        template: $TEMPLATE_IMAGE"
fi
echo "  image  : $IMAGE"
echo "  branch : $BRANCH (${REMOTE_SHA:0:7})"
echo "  budget : FP8-capable, >=${MIN_TOTAL_VRAM_GB} GB total, secure cloud"
echo "           1 GPU under \$${PREFER_USD_PER_HR}/hr, then multi-GPU under \$${PREFER_USD_PER_HR}/hr, then up to \$${MAX_USD_PER_HR}/hr"
if [ "$KEEP_ALIVE" = "true" ]; then
    echo "  after  : --keep-alive -- the pod stays up and billing when the run ends"
else
    echo "  after  : the pod stops itself when the run ends (--keep-alive keeps it warm)"
fi
echo "  guard  : the pod stops itself after ${MAX_POD_HOURS}h no matter what (MAX_POD_HOURS)"
rp_cost_summary
fi

is_fp8_gpu() {
    local gpu
    for gpu in "${FP8_GPUS[@]}"; do [ "$gpu" = "$1" ] && return 0; done
    return 1
}

confirm() {
    local answer=""
    read -r -p "$1 [y/N] " answer </dev/tty || true
    case "$answer" in [yY]*) return 0 ;; *) return 1 ;; esac
}

# ------------------------------------------------------------------------------- create
create_pod() {
    local rows found row line n i tier last_tier choice price count gpu_id body response created_id
    while :; do
        echo
        echo "==> FP8-capable secure-cloud pods with >=${MIN_TOTAL_VRAM_GB} GB total, best first:"
        # One query per card count, merged and re-sorted: 2x48 GB can undercut 1x96 GB, so
        # the counts have to compete on price rather than be decided up front.
        found=()
        for n in $GPU_COUNTS; do
            while IFS= read -r line; do found=("${found[@]}" "$line"); done \
                < <(rp_gpu_candidates "$n" "$MIN_TOTAL_VRAM_GB" "$MAX_USD_PER_HR" "$GPU_ALLOWLIST_JSON")
        done

        if [ "${#found[@]}" -eq 0 ]; then
            echo "  none available right now."
            confirm "  search again?" || { echo "aborted."; exit 1; }
            continue
        fi

        # Rank by preference, not by price: prepend a tier, then sort tier, card count,
        # price. Cheapest-first would have offered 2x RTX 6000 Ada at $1.68 ahead of the
        # single RTX PRO 6000 at $1.89, which is the trade we do not want.
        rows=()
        while IFS= read -r line; do rows=("${rows[@]}" "$line"); done < <(
            printf '%s\n' "${found[@]}" \
                | awk -F'\t' -v prefer="$PREFER_USD_PER_HR" \
                    '{ print (($1 > prefer) ? 3 : (($2 == 1) ? 1 : 2)) "\t" $0 }' \
                | sort -t "$(printf '\t')" -k1,1n -k3,3n -k2,2n
        )

        i=1
        last_tier=""
        for row in "${rows[@]}"; do
            tier="$(cut -f1 <<<"$row")"
            if [ "$tier" != "$last_tier" ]; then
                case "$tier" in
                    1) echo "  single GPU under \$${PREFER_USD_PER_HR}/hr:" ;;
                    2) echo "  multi-GPU under \$${PREFER_USD_PER_HR}/hr:" ;;
                    3) echo "  last resort, up to \$${MAX_USD_PER_HR}/hr:" ;;
                esac
                last_tier="$tier"
            fi
            printf '    [%d] %sx %-44s %4s GB total  $%s/hr  stock=%s\n' "$i" \
                "$(cut -f3 <<<"$row")" "$(cut -f7 <<<"$row")" "$(cut -f5 <<<"$row")" \
                "$(cut -f2 <<<"$row")" "$(cut -f6 <<<"$row")"
            i=$((i + 1))
        done

        # Enter aborts. Nothing here may default to spending money: the eight pods at
        # $21.33/hr behind point 9 came from a success check that guessed.
        choice=""
        read -r -p "  start which one? [number, 1 = best match, anything else aborts] " choice </dev/tty || true
        case "$choice" in
            ""|*[!0-9]*|0) echo "aborted."; exit 1 ;;
        esac
        [ "$choice" -le "${#rows[@]}" ] || { echo "  no option $choice"; continue; }

        row="${rows[$((choice - 1))]}"
        price="$(cut -f2 <<<"$row")"
        count="$(cut -f3 <<<"$row")"
        gpu_id="$(cut -f7 <<<"$row")"
        echo "==> creating $POD_NAME: ${count}x $gpu_id at \$${price}/hr"

        # cloudType is explicit, never defaulted (point 10). Everything the template already
        # decides -- disk, volume, mount path, ports, secrets -- is deliberately absent.
        body="$(jq -nc --arg name "$POD_NAME" --arg image "$IMAGE" --arg gpu "$gpu_id" \
            --arg template "$TEMPLATE_ID" --argjson count "$count" \
            --argjson cmd "$START_CMD_JSON" --argjson entry "$START_ENTRYPOINT_JSON" \
            --argjson env "$ENV_JSON" '{
                name: $name, templateId: $template, imageName: $image,
                cloudType: "SECURE", computeType: "GPU", interruptible: false,
                gpuTypeIds: [$gpu], gpuCount: $count, gpuTypePriority: "custom",
                dockerEntrypoint: $entry, dockerStartCmd: $cmd, env: $env
            }')"

        # One create, then read the result -- never a retry loop (point 9): a probe loop
        # that misparsed its own success check once created eight pods at $21.33/hr.
        if response="$(rp_api POST /pods "$body")"; then
            created_id="$(jq -r '.id // empty' <<<"$response")"
        else
            created_id=""
            echo "  ❌ create failed (HTTP $(rp_code)): $(jq -r '.error // .message // .' <<<"$response" 2>/dev/null | head -3)"
        fi

        if [ -n "$created_id" ]; then
            POD_ID="$created_id"
            echo "  created $POD_ID"
            rp_cost_summary
            # A pod that is billing but misconfigured is the worst outcome, so do not just
            # print an instruction and leave -- offer to stop it here.
            # The secrets are the whole reason for using a template, and whether pod-level
            # env merges with the template's or replaces it is not documented. Read the
            # created pod back and say which template keys actually landed.
            local missing
            missing="$(rp_api GET "/pods/$POD_ID" \
                | jq -r --argjson t "$TEMPLATE_ENV_KEYS" '($t - (.env // {} | keys)) | join(", ")')"
            if [ -n "$missing" ]; then
                echo "  ⚠️  the template's env did not reach the pod: $missing"
                echo "      The run will start without them (HF_TOKEN, S3_BUCKET, AWS_*)."
                echo "      Set them in your shell so the launcher forwards them, or edit the pod in the console."
            fi
            if ! rp_verify_pod_config "$POD_ID" "$IMAGE" "$START_CMD_JSON" "$START_ENTRYPOINT_JSON"; then
                if confirm "  stop $POD_ID now?"; then
                    rp_api POST "/pods/$POD_ID/stop" >/dev/null && echo "  stopped."
                    rp_cost_summary
                fi
                exit 1
            fi
            return 0
        fi
        # Availability is the usual reason, and it changes minute to minute -- so search
        # again rather than give up, but only on another explicit yes.
        confirm "  try again with a fresh availability check?" || { echo "aborted."; exit 1; }
    done
}

# -------------------------------------------------------------------------------- reuse
# A pod only picks up a new image or start command while it is stopped: PATCH on a running
# pod returns 200, echoes the value back, and discards it (point 6). So the running pod
# with everything already correct is the good case -- restart keeps the GPU, and stopping
# risks never getting it back (point 11).
configure_and_start_pod() {
    local pod_id="$1" pod status image current_cmd current_entry current_env desired_env response
    pod="$(rp_api GET "/pods/$pod_id")"
    status="$(jq -r '.desiredStatus' <<<"$pod")"
    image="$(rp_pod_image "$pod")"
    current_cmd="$(jq -c '.dockerStartCmd // []' <<<"$pod")"
    current_entry="$(jq -c '.dockerEntrypoint // []' <<<"$pod")"
    desired_env="$(jq -c 'to_entries | sort_by(.key)' <<<"$ENV_JSON")"
    current_env="$(jq -c --argjson want "$ENV_JSON" \
        '(.env // {}) | with_entries(select(.key | in($want))) | to_entries | sort_by(.key)' <<<"$pod")"

    if [ "$status" = "RUNNING" ] \
        && [ "$image" = "$IMAGE" ] \
        && [ "$current_cmd" = "$START_CMD_JSON" ] \
        && [ "$current_entry" = "$START_ENTRYPOINT_JSON" ] \
        && [ "$current_env" = "$desired_env" ]; then
        echo "==> $pod_id already has the right image, start command and environment."
        # Declining must not abort: the reason to say no is that a run is already going and
        # you want to watch it, so no == attach. Aborting here sent you back to the shell
        # with the job still running and no way to see it.
        if confirm "  restart the container to start the $MODE run? (kills anything running on it)"; then
            rp_api POST "/pods/$pod_id/restart" >/dev/null || {
                echo "❌ restart failed (HTTP $(rp_code))" >&2; exit 1
            }
        else
            echo "  leaving the run alone -- attaching to its log instead."
        fi
        POD_ID="$pod_id"
        return 0
    fi

    if [ "$status" = "RUNNING" ]; then
        echo "==> $pod_id needs a different configuration:"
        [ "$image" = "$IMAGE" ] || echo "     image: $image -> $IMAGE"
        [ "$current_cmd" = "$START_CMD_JSON" ] || echo "     start command differs"
        [ "$current_entry" = "$START_ENTRYPOINT_JSON" ] || echo "     entrypoint differs (image default would swallow the start command)"
        [ "$current_env" = "$desired_env" ] || echo "     environment differs"
        echo "  A running pod discards those on write, so it has to be stopped first --"
        echo "  and a stopped pod is not guaranteed to get its GPU back."
        confirm "  stop $pod_id and reconfigure it?" || { echo "aborted."; exit 1; }
        rp_api POST "/pods/$pod_id/stop" >/dev/null || {
            echo "❌ stop failed (HTTP $(rp_code))" >&2; exit 1
        }
        rp_wait_stopped "$pod_id"
    fi

    echo "==> configuring $pod_id"
    # PATCH takes no templateId, and it replaces `env` wholesale rather than merging. Sending
    # only our overrides would therefore delete the template's HF_TOKEN, S3_BUCKET and AWS
    # keys from an existing pod, so start from what the pod already has and layer on top.
    local merged_env
    merged_env="$(jq -c --argjson want "$ENV_JSON" '(.env // {}) * $want' <<<"$pod")"
    rp_api PATCH "/pods/$pod_id" \
        "$(jq -nc --arg image "$IMAGE" --argjson cmd "$START_CMD_JSON" \
            --argjson entry "$START_ENTRYPOINT_JSON" --argjson env "$merged_env" \
            '{imageName: $image, dockerEntrypoint: $entry, dockerStartCmd: $cmd, env: $env}')" >/dev/null || {
        echo "❌ configuring failed (HTTP $(rp_code))" >&2; exit 1
    }

    rp_verify_pod_config "$pod_id" "$IMAGE" "$START_CMD_JSON" "$START_ENTRYPOINT_JSON" || exit 1

    echo "==> starting $pod_id"
    if response="$(rp_api POST "/pods/$pod_id/start")"; then
        POD_ID="$pod_id"
        return 0
    fi
    echo "  ❌ start failed (HTTP $(rp_code)): $(jq -r '.error // .message // .' <<<"$response" 2>/dev/null | head -3)"
    # A pod-local volume pins the pod to one host machine. If the message above is about
    # free GPUs, that machine is full, start will keep failing, and a new pod is the only
    # way forward (point 11).
    echo "  A stopped pod is pinned to its host machine and only starts if that machine"
    echo "  still has free GPUs. If that is the reason, retrying will not help."
    confirm "  create a new pod instead?" || { echo "aborted."; exit 1; }
    create_pod
}

# ------------------------------------------------------------------------ pick a pod
# `pod list` shows running pods only; a stopped pod looks deleted unless you ask for all
# of them (point 8). The REST collection returns both.
PODS="$(rp_api GET /pods)" || { echo "❌ could not list pods (HTTP $(rp_code)): $PODS" >&2; exit 1; }
# Include a pod unless it is positively identified as a CPU pod. Requiring evidence of a
# GPU looked safer and was much worse: for the first minutes after creation `gpu` is null
# and `machine` is empty, so a freshly created pod is invisible -- and the next run would
# happily create a second one at $2/hr. Absence of evidence is not evidence of absence.
CANDIDATES="$(jq -c '[ .[]
    | select(.desiredStatus != "TERMINATED" and ((.cpuFlavorId // "") == ""))
    | {id, name, status: .desiredStatus, cost: (.costPerHr // 0),
       image: (.imageName // .image // "?"),
       gpuTypeId: (.machine.gpuTypeId // ""),
       gpu: (((.gpu.count // 0) as $c | if $c > 0 then "\($c)x " else "" end)
             + (.gpu.displayName // .machine.gpuDisplayName // .machine.gpuTypeId
                // "GPU not reported yet"))} ]' <<<"$PODS")"
COUNT="$(jq -r 'length' <<<"$CANDIDATES")"

POD_ID=""
if [ "$COUNT" -eq 0 ]; then
    if [ "$ATTACH" = "1" ]; then
        echo "❌ no pod to attach to." >&2
        exit 1
    fi
    echo
    echo "==> no reusable GPU pod found."
    create_pod
else
    echo
    echo "==> existing GPU pods:"
    jq -r 'to_entries[] | "  [\(.key + 1)] \(.value.name)  \(.value.id)  \(.value.status)  \(.value.gpu)  $\(.value.cost)/hr  \(.value.image)"' <<<"$CANDIDATES"
    if [ "$COUNT" -eq 1 ]; then
        POD_TO_USE="$(jq -r '.[0].id' <<<"$CANDIDATES")"
        [ "$ATTACH" = "1" ] && echo "  attaching to $POD_TO_USE" || echo "  reusing $POD_TO_USE"
    else
        CHOICE=""
        read -r -p "  reuse which one? [number, n = create a new pod] " CHOICE </dev/tty || true
        case "$CHOICE" in
            [nN]*) POD_TO_USE="" ;;
            ""|*[!0-9]*|0) echo "aborted."; exit 1 ;;
            *)
                POD_TO_USE="$(jq -r --argjson i "$((CHOICE - 1))" '.[$i].id // empty' <<<"$CANDIDATES")"
                [ -n "$POD_TO_USE" ] || { echo "❌ no option $CHOICE" >&2; exit 1; }
                ;;
        esac
    fi
    if [ -n "$POD_TO_USE" ]; then
        # Reuse is not filtered by the allow-list -- an existing pod is worth more than a
        # perfect one -- but silently running an FP8 test on Ampere is how a wrong result
        # gets believed.
        GPU_TYPE="$(jq -r --arg id "$POD_TO_USE" '.[] | select(.id == $id) | .gpuTypeId' <<<"$CANDIDATES")"
        if [ "$ATTACH" = "0" ] && [ -n "$GPU_TYPE" ] && ! is_fp8_gpu "$GPU_TYPE"; then
            echo "⚠️  $GPU_TYPE predates Ada and has no FP8 tensor cores."
            echo "    bf16 training and eval are fine; the FP8 base model (TODO 2) is not."
        fi
        if [ "$ATTACH" = "1" ]; then
            POD_ID="$POD_TO_USE"
        else
            configure_and_start_pod "$POD_TO_USE"
        fi
    else
        create_pod
    fi
fi

# ------------------------------------------------------------------------------ attach
echo
echo "==> waiting for $POD_ID"
rp_wait_running "$POD_ID"
rp_cost_summary

SSH_CMD="ssh -i $RP_SSH_KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $RP_SSH_PORT root@$RP_SSH_HOST"
echo
echo "  ssh    : $SSH_CMD"
echo "  log    : $POD_LOG"
echo "  stop   : runpodctl stop pod $POD_ID"
echo "  In an SSH session the pipeline's env is missing -- sshd does not inherit it:"
echo "    while IFS= read -r -d '' l; do export \"\$l\"; done < /proc/1/environ"
echo

# The container installs sshd on boot, so the port answers before the daemon does.
# BatchMode, so a key that needs a passphrase fails instead of waiting forever on a prompt
# that '2>/dev/null' would have hidden; and keep the last error to show if we give up.
echo "==> waiting for sshd (key: $RP_SSH_KEY)"
WAITED=0
SSH_ERR="$(mktemp "${TMPDIR:-/tmp}/runpod-ssh.XXXXXX")"
trap 'rm -f "$RP_CODE_FILE" "$SSH_ERR"' EXIT
until rp_ssh "$RP_SSH_HOST" "$RP_SSH_PORT" true 2>"$SSH_ERR"; do
    if [ "$WAITED" -ge 300 ]; then
        echo
        echo "❌ no SSH after ${WAITED}s. Last error:" >&2
        sed 's/^/     /' "$SSH_ERR" >&2
        echo "   The pod is up and the run is unaffected -- this is only the log tail." >&2
        echo "   Try: ssh -i $RP_SSH_KEY -o IdentitiesOnly=yes -p $RP_SSH_PORT root@$RP_SSH_HOST" >&2
        exit 1
    fi
    printf '\r  (%ss)  ' "$WAITED"
    sleep 10
    WAITED=$((WAITED + 10))
done

echo
echo "==> tailing $POD_LOG -- Ctrl+C detaches, the run keeps going"
echo
# The run is setsid-detached from PID 1, not a child of this SSH session, so nothing here
# can take it down.
trap 'echo; echo "detached. The run continues on $POD_ID."; echo "  reattach: $SSH_CMD \"tail -n +1 -F $POD_LOG\""; exit 0' INT
rp_ssh "$RP_SSH_HOST" "$RP_SSH_PORT" "tail -n +1 -F '$POD_LOG'" || true
