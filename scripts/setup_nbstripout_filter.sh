#!/usr/bin/env bash
set -euo pipefail

git config --local filter.nbstripout.clean "python3 scripts/strip_ipynb_outputs.py"
git config --local filter.nbstripout.smudge cat
git config --local filter.nbstripout.required true
git config --local diff.ipynb.textconv "python3 scripts/strip_ipynb_outputs.py"
