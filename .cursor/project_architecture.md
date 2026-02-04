# Project Architecture

## Overview

This repo is a collection of **Cursor skills and rules** installable at **project level**. Users run "Fetch and follow instructions from…" with the raw INSTALL.md URL; the agent clones the repo and copies skills and rules into the project’s `.cursor/`. No symlinks (copy-based only).

## Components

| Path | Role |
|------|------|
| `.cursor/INSTALL.md` | Install instructions: clone, copy skills/rules, optional project_architecture template. |
| `.cursor/skills/` | Skills: dual-remote-push, efficient-code, latency-logging, new-project-scaffold, pre-commit-docs-sync (enforces README as SOP), systematic-debugging, test-driven-development. |
| `.cursor/rules/` | Rule: efficient-and-deliberate (efficiency and deliberate implementation). |
| `.cursor/project_architecture.md` | Template for project architecture; copied on install if missing. |

## Data Flow

1. User pastes the raw INSTALL.md URL in Cursor Agent.
2. Agent fetches INSTALL.md and runs: clone repo into `.cursor/lazy-code-skill`, copy skills and rules, then remove the clone so only `.cursor/skills/lazy-code-skill/` and `.cursor/rules/lazy-code-skill-*.mdc` remain.
3. Cursor discovers skills and rules from `.cursor/skills/` and `.cursor/rules/`.

## Deprecated / Removed

(Empty.)
