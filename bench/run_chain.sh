#!/bin/zsh
cd /Users/pranav/Documents/coding_projects/degless
PY=.venv/bin/python
echo "### outlook $(date)"; $PY bench/bench_outlook.py > bench/out/outlook.log 2>&1; echo "outlook exit $?"
echo "### ablation $(date)"; $PY bench/bench_ablation.py > bench/out/ablation.log 2>&1; echo "ablation exit $?"
echo "### speed $(date)"; $PY bench/bench_speed.py barcelona-2026 > bench/out/speed.log 2>&1; echo "speed exit $?"
echo "### live $(date)"; $PY bench/bench_live.py > bench/out/live.log 2>&1; echo "live exit $?"
echo "### stability $(date)"; $PY bench/bench_stability.py > bench/out/stability.log 2>&1; echo "stability exit $?"
echo "### done $(date)"
