---
name: Fine-tuning using Axolotl
description: Tools that help fine-tuning AI models using Axolotl
---

# Show overview and available training methods

```sh
axolotl agent-docs
```

# Topic-specific references

```sh
axolotl agent-docs sft                 # supervised fine-tuning
axolotl agent-docs grpo                # GRPO online RL
axolotl agent-docs preference_tuning   # DPO, KTO, ORPO, SimPO
axolotl agent-docs reward_modelling    # outcome and process reward models
axolotl agent-docs pretraining         # continual pretraining
axolotl agent-docs --list              # list all topics
```

# Dump config schema for programmatic use

```sh
axolotl config-schema
axolotl config-schema --field adapter
```