#!/usr/bin/env bash
set -e

echo ">>> Upgrade pip and install requirements"
python -m pip install --upgrade pip
pip install -r requirements.txt

echo ">>> Syntax-checking available files (non-fatal)"
python -m py_compile pyterm_singlefile.py || true
python -m py_compile pyterm_rich.py || true
python -m py_compile pyterm/core.py terminal_app.py || true

echo "Build success"
echo "You can now run the app with: python terminal_app.py"
echo "Or create a single-file executable with: python pyterm_singlefile.py"
echo "Or create a rich version with: python pyterm_rich.py"
echo "For more details, see the README.md file."    