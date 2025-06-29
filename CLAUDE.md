# CLAUDE.md - Project Context

This file contains important context and instructions for Claude Code.

## Project Overview
Python project requiring testing infrastructure setup.

## Commands to Run After Code Changes
- `poetry run test` - Run all tests
- `poetry run lint` - Run linting (if configured)
- `poetry run typecheck` - Run type checking (if configured)

## Testing Infrastructure
- Package manager: Poetry (or UV if detected)
- Test framework: pytest
- Coverage: pytest-cov
- Test directories: tests/unit/, tests/integration/

## Important Notes
- Always run tests after making changes
- Coverage threshold is currently set to 0% for infrastructure setup
- **TODO**: Change `--cov-fail-under=0` to `--cov-fail-under=80` in pyproject.toml after writing tests
- Use conventional commit messages