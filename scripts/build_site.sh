#!/usr/bin/env bash
# Regenerates the static pages and copies the code, data, results and paper the site serves.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 scripts/make_pages.py docs >/dev/null 2>&1 || "${PYTHON:-python}" scripts/make_pages.py docs
mkdir -p docs/py docs/data
cp src/plu_ggn.py src/playground_api.py docs/py/
cp data/diabetes_25k.csv docs/data/diabetes_25k.csv
cp results/results.json docs/data/results.json
cp paper/main.pdf docs/paper.pdf
touch docs/.nojekyll
echo "site assets updated"
