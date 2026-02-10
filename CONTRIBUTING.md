# Contributing

Thanks for considering contributing.

## Scope
This repository is research code for 3D radiotherapy dose prediction. Contributions are welcome for:
- bug fixes and reproducibility improvements
- clearer documentation
- refactoring for configurability (argparse / YAML configs)
- additional evaluation utilities

## How to contribute
1. Fork the repo and create a feature branch.
2. Make your changes with clear commit messages.
3. Add/adjust tests or minimal sanity checks when appropriate.
4. Open a pull request describing:
   - what you changed
   - why it matters
   - how to reproduce/verify

## Style
- Keep functions small and composable.
- Prefer explicit tensor shapes in comments/docstrings.
- Avoid hard-coded paths in new code (use configs).
