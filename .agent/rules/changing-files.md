---
trigger: always_on
---

Never make changes in this repository on the `main` branch.

**CRITICAL**: Before performing any edit to any file in this repository, you **MUST** first check the current branch and changes:
```bash
// turbo
git status
```

If the output is `main`, you are **FORBIDDEN** from editing any file. You **MUST** stop and ask the user to switch to a feature branch.
If `main` has pending changes, inform the user about that.

**CRITICAL**: If this repository is on a branch that is not yours, it might be used by another agent!