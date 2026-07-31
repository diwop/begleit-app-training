# Launcher Hardening

Requirements for a `run train|eval` script that reuses a matching stopped pod or cleanly
creates a new one. Every item below comes from a failure observed on 2026-07-28/29, not
from speculation. The recurring theme: the pipeline assumes properties of the container
image and of RunPod's API that hold for one image and break on the next.

## Container image assumptions

Three images behaved three different ways, and each difference cost a run:

| image | default CMD | what broke |
|---|---|---|
| `axolotlai/axolotl-cloud-uv` | sshd + JupyterLab | nothing; masked every assumption below |
| `lmsysorg/sglang` | server entrypoint that exits | container died on boot, `container is not running` |
| `vllm/vllm-openai` | `vllm serve` on Qwen3-0.6B | container stayed up and **reserved 90% of the GPU** |

1. **Always set `dockerStartCmd` explicitly.** Never inherit the image default. The vLLM
   image's default silently consumed 76 GB of an 80 GB A100, which presented as a memory
   leak and cost roughly two hours of misdiagnosis.
2. **Probe for tools, install what is missing.** `curl`, `wget`, `git`, `openssh-server`.
   The vLLM image ships none of them; `docs/pipeline.md` hardcodes `wget`.
3. **Install and start sshd.** Without it only RunPod's proxy SSH works, and the proxy
   discards remote commands, so no automation can read logs or run diagnostics.
4. **End the start command with `sleep infinity`** so a finished pipeline leaves the
   container reachable rather than exiting.

## RunPod API behaviour

5. **`runpodctl pod create --docker-args` is silently dropped.** It returns success and
   the field is unset. Use the REST API and verify with a fresh `GET`.
6. **`dockerStartCmd` only persists while the pod is stopped.** A `PATCH` on a running pod
   returns HTTP 200, echoes the value back, and discards it.
7. **Never trust a write; verify with an independent read.** Points 5 and 6 both present
   as success.
8. **`pod list` shows running pods only.** Use `-a`, or a stopped pod looks deleted.
9. **Never call `pod create` in a loop.** A probe loop whose success check misparsed the
   response created eight pods at $21.33/hr. One create, verified, then stop.
10. **Cloud type must be explicit.** Never default it. Community Cloud runs on third-party
    hosts outside RunPod's audited GDPR boundary, and this pipeline puts Lebenshilfe source
    texts and live S3 credentials on the pod.
11. **A stopped pod may not get its GPU back.** Pod-local volumes pin a pod to one host
    machine; if that machine's GPUs are taken, `start` fails permanently. Detect
    `not enough free GPUs` and fall back to creating a new pod.

## Environment and process hygiene

12. **SSH sessions do not inherit the container env.** `sshd` forks from its own
    environment, so `$BRANCH`, `$MODE`, `$S3_BUCKET` are empty. Anything run over SSH must
    source `/proc/1/environ` first:
    `while IFS= read -r -d "" line; do export "$line"; done < /proc/1/environ`
13. **Background jobs need `setsid`.** `nohup ... &` over a one-shot SSH connection dies
    when the connection closes.
14. **Kill stale engine processes before starting.** A failed vLLM run orphans
    `VLLM::EngineCore`, which holds the whole card. Match on `VLLM::EngineCore`, not on
    `vllm` -- the latter matches the shell's own `/vllm-workspace` cwd and kills the session.
15. **Assert free VRAM before loading.** One `nvidia-smi` call replaces a 40-line traceback
    after a full model-load attempt.
16. **`cd /` before deleting `/runner/repo`.** `launch.sh` removes a directory the caller
    may be sitting in, and git then fails with `Unable to read current working directory`.
17. **`python3`, never `python`.** The vLLM image has no `python` on PATH; the S3 log
    upload failed silently because of it.

## Storage

18. **`/app` is MooseFS, a network filesystem served from another datacenter.** Large
    sequential reads (model weights) are fine. Treat small-file churn there as suspect.
19. **Cache the model weights on the volume.** `HF_HOME=/app/huggingface_cache` took model
    load from 659 s to 11 s.
20. **Pod-local volumes die with the pod.** Four pods on 2026-07-29 meant paying the 50 GB
    download four times. A network volume is the only fix that survives recreation, at the
    cost of pinning deployments to one datacenter.
21. **Check disk before the merge.** The bf16 merge needs ~130 GB transiently: 50 GB base,
    50 GB merged intermediate, 28 GB FP8 output.

## Artifact identity

22. **Never let "newest prefix wins" choose what is under test.** A stale local adapter and
    three logs that predated the artifact they were supposed to exercise each produced a
    confident wrong conclusion. Pin the artifact explicitly (`SMOKE_ADAPTER_S3`) and print
    what was resolved.
23. **Version merged models per run.** Transformers prefers a stray `model.safetensors`
    over a newer shard index, so overwriting a fixed prefix silently serves old weights.

## Cost control

24. **Set `--stop-after` on every pod.** `KEEP_ALIVE=true` means the pipeline never stops
    the pod itself.
25. **Report running cost after any create or delete**, so an accident is visible within
    seconds rather than minutes.
