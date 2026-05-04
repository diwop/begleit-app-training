---
trigger: always_on
---

# Development

If you introduce a new feature or make a change it has to be reflected in tests.
If there are existing unit, integration or end-to-end tests, extend or update them.
If not, evaluate which test type or tests with different type are appropriate to test the change.

**CRITICAL**: If the user asks for a _plan_, **DO NOT** modify any files yet. Other agents might be planning or editing in parallel. Only modify files after the user approves the plan, and you switch to execution mode.

During feature development check if deployment workflow needs modifications.

**CRITICAL**: At the end of development run all final checks in the relevant skills (Frontend/Backend) before committing.
If the user asks for a commit, make sure your ran all the final checks on all changes first.

# Security

Embrace the shift left on security:
- Make sure user inputs are properly escaped either by framework or by custom code.
- Apply the least privilige concept. Configure IaC in a way that the server just gets the permissions it actually needs.
- When dealing with user data, make sure they are only identifiable with the user's id as a prefix. Always validate that a user is allowed to obtain requested data.

**CRITICIAL**: Never introduce code that does not apply to these rules, not even for a spike or an uncommitted draft.

**CRITICAL**: Never store credentials in the code. Before you create a commit, always check the diff using `git diff` and scan it for credentials.

# MCP Tools

The `github` MCP server is available to assist with repository management.

- Use it to search issues, create comments, or manage pull requests if relevant to the task.

# STACKIT

The word "STACKIT" or "stackit" in any written way means the German Cloud Provider, NOT the CLI tool getstackit/stackit.