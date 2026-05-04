---
description: Initial setup guide (GitHub repository, installation)
---

# Initial Setup

Guide the user through the steps you as an agent together with the user have to do to set up the project.

## Connect to GitHub Repository

Check if the local project is connected to a GitHub repository:

```bash
git remote get-url origin
```

If not:

- Ask the user to create a new repository
- You need to ask for the repo-url
- Connect it but do not push anything:

  ```bash
  git remote add origin <repo-url>
  ```

When the GitHub repository is configured suggest to check its configuration via workflow @configure-github-repo.md