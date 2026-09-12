#!/bin/zsh
# The benchmark suite, in order.  Assumes the retrospective pipeline has run on
# every scored weekend (make history).  Each stage logs to bench/out/<name>.log.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
run() { echo "### $1 $(date)"; $PY bench/$1.py ${@:2} > bench/out/$1.log 2>&1; echo "$1 exit $?"; }
run bench_accuracy
run bench_strategy
run bench_ablation
run bench_outlook
run bench_live
run bench_stability
run bench_speed barcelona-2026
run bench_apex
echo "### pytest $(date)"; $PY -m pytest tests -q > bench/out/pytest.log 2>&1; echo "pytest exit $?"
run bench_compare
echo "### done $(date)"
