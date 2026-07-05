---
name: runpod-ssh
description: Protocols and environment configurations for managing RunPod GPU containers and evaluation pipelines via SSH.
---

# RunPod SSH Skill

This skill defines the standard operating procedures for interacting with RunPod containers, specifically for the diwop/begleit-app-training project.

## SSH Configuration
Always use the following flags to ensure a stable, non-interactive pseudo-terminal:
- `-o StrictHostKeyChecking=no`: Bypasses host key verification prompts.
- `-tt`: Forces pseudo-terminal allocation (critical for interactive shells like zsh on RunPod).
- `-i ~/.ssh/id_ed25519`: Specifies the identity file.

## Workspace Standards
- **Repository Path**: `/app/repo`
- **Output Directory**: `/app/output`
- **HuggingFace Cache**: `/app/huggingface_cache` (exported as `HF_HOME`)

## Environment Variables
For multi-GPU (2x L40S) evaluation:
- `TP_SIZE=2`
- `CUDA_VISIBLE_DEVICES=0,1`
- `NCCL_P2P_DISABLE=1` (Essential for virtualized PCIe stability)
- `NCCL_IB_DISABLE=1`
- `TORCH_NCCL_BLOCKING_WAIT=1`

## Logging and Monitoring
- **Main Log**: `/app/evaluation_run.log`
- **Execution**: Use `bash scripts/eval.sh`
- **Background Execution**: `nohup bash scripts/eval.sh > /app/evaluation_run.log 2>&1 &`
- **Monitoring**: `tail -f /app/evaluation_run.log`

## Critical Insights & Best Practices
- **Persistent Sessions**: One-off commands (`ssh ... "cmd"`) often fail or ignore arguments. Always establish a persistent session (`ssh -o StrictHostKeyChecking=no -tt ...`) and use the `manage_task` tool to send input directly to the active shell.
- **Shell Consistency**: RunPod containers often use `zsh`. Using `bash -c` or direct input ensures consistent command interpretation.
- **Workspace Reliability**: `/app/repo` is the standard location. If `/app` doesn't exist, `mkdir -p /app` as root.
- **Dependency Management**: Always check for `uv` (`uv --version`) before running scripts that depend on it. Fall back to `python3 -m pip` if necessary, but prioritize `uv` for speed.
- **Output Masking**: If `cat` or `ls` returns no output in the logs, it's likely a terminal allocation issue. Use `manage_task` to read the log output from a persistent session or try `BatchMode=yes` for raw data retrieval.
- **PTY Requirement**: RunPod SSH often explicitly requires a pseudo-terminal. Commands failing with `Error: Your SSH client doesn't support PTY` confirm that `-tt` must be used.
- **Process Management**: When restarting evaluations, always ensure `sglang` and `python` processes are terminated to release GPU memory: `pkill -f evaluation.py || true && pkill -f sglang || true`.
- **Log Verification**: If `/app/evaluation_run.log` exists but `tail` shows nothing, the script might be in an initialization phase (DVC pull, pip install). Check `ps aux` for active `eval.sh` or `dvc` processes.
