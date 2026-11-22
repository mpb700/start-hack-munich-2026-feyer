#!/bin/bash
# Erzeugt die mitgelieferten Bytecode-Dateien neu. Nach JEDER Änderung an differometor/*.py oder argparse.py
# ausführen (mit Python 3.12, im Wurzelverzeichnis des Repos), sonst geht der Gewinn stillschweigend verloren:
#
#   bash runs/philip/bytecode.sh [python]
#
# 1. differometor/_dataclass_code.bin: Code der Dataclass-Methoden, die ein Benchmark-Lauf erzeugt
#    (differometor/_startup.py, DATACLASS_CODE). Vorhandene Einträge bleiben, neue kommen dazu; die Datei darf
#    also nacheinander mit mehreren Python-3.12-Versionen gefüllt werden (hier 3.12.3 und 3.12.11).
# 2. __pycache__ mit Hash-geprüften .pyc für differometor/ und argparse.py (den Träger der Start-Massnahmen).
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${1:-python}"
"$PY" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 nötig"'
"$PY" - <<'PYTHON'
import importlib.util, os, runpy, sys, tempfile
spec = importlib.util.spec_from_file_location("_differometor_startup_tool", os.path.join("differometor", "_startup.py"))
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)
tool.record_dataclass_code()
sys.path.insert(0, os.getcwd())
report = os.path.join(tempfile.mkdtemp(), "bytecode-run.json")
sys.argv = ["benchmark_original.py", "--repeats", "3", "--output", report]
runpy.run_path("benchmark_original.py", run_name="__main__")
PYTHON
find differometor -maxdepth 2 -name '*.cpython-312.pyc' -delete
rm -f __pycache__/argparse.cpython-312.pyc __pycache__/statistics.cpython-312.pyc
"$PY" -m compileall -q --invalidation-mode checked-hash differometor argparse.py >/dev/null
git add -f differometor/_dataclass_code.bin differometor/__pycache__/*.cpython-312.pyc __pycache__/argparse.cpython-312.pyc 2>/dev/null || true
echo "Bytecode neu erzeugt. Mit git status prüfen und committen."
