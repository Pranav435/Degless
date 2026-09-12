#!/bin/zsh
cd /Users/pranav/Documents/coding_projects/degless
until grep -q "### done" bench/out/chain.log; do sleep 15; done
echo "### apex $(date)"; .venv/bin/python bench/bench_apex.py > bench/out/apex.log 2>&1; echo "apex exit $?"; echo "### apexdone $(date)"
