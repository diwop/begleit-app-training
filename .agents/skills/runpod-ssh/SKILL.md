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

## Common Fixes
- If `ls` or `cat` returns no output, ensure the command is wrapped in a single string and check for shell-specific quoting issues (e.g., zsh on the pod).
- Always `mkdir -p /app` before cloning to ensure the mount point exists.
