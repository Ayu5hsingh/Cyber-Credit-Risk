#!/usr/bin/env bash
# Launch the cyber stress-test app from the project's .venv (see README, Run: LightGBM on macOS needs libomp).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export OMP_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1
export DYLD_FALLBACK_LIBRARY_PATH="$($PY -c 'import sklearn, os; print(os.path.join(os.path.dirname(sklearn.__file__), ".dylibs"))')"
exec "$PY" -m streamlit run app/app.py "$@"
