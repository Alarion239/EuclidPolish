#!/bin/zsh
# Queue (one python process at a time): wait for the running pf 228 job, host injection for 228 and 229,
# summaries + plots, then the optional real-sky gradient arrays (257@15M + 2-tile 238 check) and their profile.
W=/private/tmp/claude-502/-Users-alarion239-Desktop-EuclidPolish/6d444c86-12c6-4a7f-b4eb-7b73fa2e1c66/scratchpad/tests257/outskirts
U=/private/tmp/claude-502/-Users-alarion239-Desktop-EuclidPolish/d1a24d4c-2869-4369-a6ae-08d92714c971/scratchpad/underflux/out
PY=~/miniforge3/envs/EuclidPolishEnv/bin/python
while kill -0 66128 2>/dev/null; do sleep 2; done
echo "=== pf member_228 (orphaned) finished by $(date +%T)" >> $W/runs/chain2.status
export PYTHONPATH=/Users/alarion239/Desktop/EuclidPolish KMP_DUPLICATE_LIB_OK=TRUE
cd /Users/alarion239/Desktop/EuclidPolish
step() {  # name, log, command...
  local name=$1 log=$2; shift 2
  echo "=== $name start $(date +%T)" >> $W/runs/chain2.status
  /usr/bin/time -l "$@" > $log 2>&1
  echo "=== $name exit $? end $(date +%T)" >> $W/runs/chain2.status
}
for spec in member_228 member_229; do
  step "hosts $spec" $W/runs/hosts_$spec.log $PY $W/inject_hosts.py $U "$spec" $W/runs/hosts_$spec.npz
done
step "summary b" $W/runs/summary_b.txt $PY $W/summary.py
step "hosts summary b" $W/runs/hosts_summary_b.txt $PY $W/hosts_summary.py
step "plots b" $W/runs/plots_b.log $PY $W/plot/profile257.py
step "gain 257" $W/runs/gain_257_15M.log $PY $W/gain_run.py $U "milestone:member_257@15000000" $W/runs/gain_257_15M
step "gain 238 check" $W/runs/gain_238_check.log $PY $W/gain_run.py $U member_238 $W/runs/gain_238_check 2
step "gain profile" $W/runs/gain_profile.txt $PY $W/gain_profile.py $U $W/runs
echo "=== chain3 done $(date +%T)" >> $W/runs/chain2.status
