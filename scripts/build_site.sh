#!/usr/bin/env bash
# Copies the code, data, results and paper that the static site (docs/) serves.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p docs/py docs/data
cp src/plu_ggn.py docs/py/plu_ggn.py
cp data/diabetes_25k.csv docs/data/diabetes_25k.csv
cp results/results.json docs/data/results.json
cp paper/main.pdf docs/paper.pdf
touch docs/.nojekyll
echo "site assets updated"
