#!/bin/zsh
# The benchmark suite, in order.  Assumes the retrospective pipeline has run on
# every scored weekend (make history).  Each stage logs to bench/out/<name>.log,
# the run itself to bench/out/run_all.log, and its timings to
# bench/out/runtime.json so the report can be written from bench/out/*.json alone.
# Every stage is offline: nothing here touches the network, and nothing here
# writes to data/processed/ or predictions/sealed/.
# Each script takes --events <keys ...> (default: all seven scored weekends).
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=bench/out/run_all.log
RT=bench/out/runtime.json
T0=$(date +%s)
: > $LOG
: > $RT.parts

say() { echo "$@" | tee -a $LOG; }
run() {
  local name=$1; shift
  local t=$(date +%s)
  say "### $name $(date)"
  $PY bench/$name.py "$@" > bench/out/$name.log 2>&1
  local rc=$?
  local dt=$(( $(date +%s) - t ))
  say "$name exit $rc  [${dt}s]"
  echo "  \"$name\": {\"seconds\": $dt, \"exit\": $rc}," >> $RT.parts
}

run bench_accuracy
run bench_strategy
run bench_ablation
run bench_outlook
run bench_live
run bench_stability
run bench_speed --events barcelona-2026
run bench_apex
# V4: the place-value estimator audit and the extrapolation measurement are
# pure reads of data/processed; the live ablation, the paired tick timing and
# the experiment grid re-run the engine and the search on frozen inputs.
run bench_place_value
run bench_extrapolation
run bench_live --no-race-state
run bench_tick_paired
run bench_experiments
t=$(date +%s)
say "### pytest $(date)"
$PY -m pytest tests -q > bench/out/pytest.log 2>&1
rc=$?
dt=$(( $(date +%s) - t ))
say "pytest exit $rc  [${dt}s]"
echo "  \"pytest\": {\"seconds\": $dt, \"exit\": $rc}," >> $RT.parts
# runtime.json is written *before* bench_compare so the comparison quotes this
# run's stage timings (a previous version wrote it afterwards, and compare.json
# carried the previous suite's numbers); it is rewritten once more at the end
# with bench_compare's own second and the final total.
TOTAL=$(( $(date +%s) - T0 ))
{ echo "{"; cat $RT.parts; echo "  \"total_seconds\": $TOTAL"; echo "}" } > $RT
run bench_compare
run bench_v4_compare

TOTAL=$(( $(date +%s) - T0 ))
{ echo "{"; cat $RT.parts; echo "  \"total_seconds\": $TOTAL"; echo "}" } > $RT
rm -f $RT.parts
say "### done $(date)  [total ${TOTAL}s]"
