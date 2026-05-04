---
description: Configure GitHub repository settings (Issues, Wiki, Merge strategies, etc.)
---

# Configure GitHub Repository

Guide the user through the steps you as an agent together with the user have to do to configure the GitHub repository settings.

// turbo-all

## Review Suggested Settings

Explain to the user the following suggested settings for the repository:

- **Delete branch on merge**: Enabled (Keeps the repository clean by automatically deleting feature branches after they are merged)
- **Squash merging**: Enabled (Maintains a clean, linear commit history by squashing all commits into one upon merge)
- **Rebase merging**: Enabled (Provides flexibility if a linear commit history without a merge commit is desired)
- **Merge commits**: Disabled (Forces squash or rebase merging to maintain linear history)
- **Auto merge PRs**: Enabled (Convenient and safe with the other configuration)

## Configure Settings using GitHub CLI

Offer to the user to configure these repository settings using the `gh` CLI.

1.  Check if `gh` is installed (`gh --version`).
2.  If installed, ask the user if they want you to set them automatically.
3.  If yes, run:

    ```bash
    #  Find out <org/repo>
    git remote get-url origin
    # Login check
    gh auth status || gh auth login
    # Configure repository settings
    gh repo edit <org/repo> \
      --delete-branch-on-merge \
      --enable-squash-merge \
      --enable-rebase-merge \
      --enable-merge-commit=false \
      --enable-auto-merge
    ```

## Branch Protection (main)

First, check if there are existing protections or rulesets applied to the `main` branch. This ensures you do not overwrite existing organization-wide or manual rules unintentionally:

```bash
# Check if branch rulesets apply
gh ruleset check main

# Check classic branch protection
gh api /repos/<org/repo>/branches/main/protection || echo "No classic protection configured"
```

Explain the following suggested branch protection rules for the `main` branch to ensure code quality and linear history:

- **Require a pull request before merging**: Enabled
  - **Require review from Code Owners**: Enabled (no need to check for CODEOWNERS file)
  - **Dismiss stale pull request approvals when new commits are pushed**: Enabled
- **Require status checks to pass before merging**: Enabled
  - **Require branches to be up to date before merging**: Enabled
  - **Status checks**: Add the leaf checks (no other check depends on them) from @../../.github/workflows/pull-request.yml
- **Require linear history**: Enabled
- **Allow force pushes**: Disabled
- **Allow deletions**: Disabled

If there are differences, suggest to the user to configure these via `gh api`:

4.  Set branch protection for `main`:

    ```bash
    gh api --method PUT \
      -H "Accept: application/vnd.github+json" \
      /repos/<org/repo>/branches/main/protection \
      --input - <<< '{
        "required_status_checks": {
          "strict": true,
          "contexts": [
            "Check Dependabot",
            "Test Server"
          ]
        },
        "enforce_admins": false,
        "required_pull_request_reviews": {
          "dismiss_stale_reviews": true,
          "require_code_owner_reviews": true,
          "required_approving_review_count": 0
        },
        "restrictions": null,
        "required_linear_history": true,
        "allow_force_pushes": false,
        "allow_deletions": false
      }'
    ```