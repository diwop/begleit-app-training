---
description: Onboarding guide for new developers (requirements, MCP server setup)
---

# Onboarding

Create a markdown artifact with greeting to developer that you will guide through setup.
That artifact might turn into an implementation plan later.

# Requirements

Run command exactly to check for missing requirements:

// turbo-all

```bash
echo "git" && \
git --version && \
echo "docker" && \
echo "python3" && python3 --version && \
echo "uv" && uv --version && \
echo "done"
```

Your agent terminal might just see paths from `~/.zshenv` (not `~/.zshrc`) or `~/.bash_profile` (not `~/.bashrc`).
If users specify their `$PATH` in the wrong file, you might not see installed tools.
If you see few or no dependencies or if you have installed a tool (especially protoc-gen-openapiv2 via Go) but the script still fails, we may have a `$PATH` configuration mismatch, the run `echo $PATH`.
Check the path and ask the developer wether `$PATH` is defined in the wrong file or what is missing (e.g. `export PATH="$PATH:$(go env GOPATH)/bin`).
If the developer fixed path issues, run `source ~/.zshenv` (or your respective file) and run the requirements check again.

If there are missing dependencies:

- Turn artifact into implementation plan
- Add tasks to install missing dependencies to the plan
- State which you can and will install
- State in detail which need to be installed manually and how

| Tool    | mac                        | linux                                      | win                                                         |
| ------- | -------------------------- | ------------------------------------------ | ----------------------------------------------------------- |
| git     | brew install git           | sudo apt install git                       | winget install Git.Git                                      |
| docker  | brew install --cask docker | sudo apt install docker.io                 | winget install Docker.DockerDesktop                         |
| python3 | brew install python@3.12   | sudo apt install python3 python3-pip       | winget install Python.Python.3.12                           |
| uv      | brew install uv            | curl -LsSf https://astral.sh/uv/install.sh | powershell -c "irm https://astral.sh/uv/install.ps1 \| iex" |

# Git Repo

Check if this repository is a git repository.
If not mention that in the artifact.

# MCP servers

Check which MCP servers are already available by inspecting your available tools.
Do NOT try to read the config file directly as it contains secrets.

Mention briefly the existing suggested MCP servers in artifact.
Explain missing suggested MCP servers from @mcp_server_template.json in artifact.

For each credentials/token explain separately where and how to obtain.

Explain developer has to merge json manually with ~/.gemini/antigravity/mcp_config.json for security reasons.

# Summary

Summarize artifact in final output.
If there are missing dependencies ask to proceed with implementation plan.
Clarify what you do at "proceed" and what dev has to do.
If there are none there is no need to proceed: Workflow done.

If there is no github repository, this is a new project.
Suggest to continue with @initial-setup.md workflow.
