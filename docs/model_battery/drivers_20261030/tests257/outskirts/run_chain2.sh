#!/bin/zsh
# Reordered queue (one python process at a time): wait for the running pf 238 job, then pf 249, the host
# injection (257@15M, 238, 249), summaries + plots, pf 228, summaries again, and last the optional
# real-sky gradient arrays + their profile.
W=/private/tmp/claude-502/-Users-alarion239-Desktop-EuclidPolish/6d444c86-12c6-4a7f-b4eb-7b73fa2e1c66/scratchpad/tests257/outskirts
U=/private/tmp/claude-502/-Users-alarion239-Desktop-EuclidPolish/d1a24d4c-2869-4369-a6ae-08d92714c971/scratchpad/underflux/out
PY=~/miniforge3/envs/EuclidPolishEnv/bin/python
while kill -0 60605 2>/dev/null; do sleep 2; done
echo "=== member_238  (orphaned pf run) finished by $(date +%T)" >> $W/runs/run_pf.status
export PYTHONPATH=/Users/alarion239/Desktop/EuclidPolish KMP_DUPLICATE_LIB_OK=TRUE
cd /Users/alarion239/Desktop/EuclidPolish
step() {  # name, log, command...
  local name=$1 log=$2; shift 2
  echo "=== $name start $(date +%T)" >> $W/runs/chain2.status
  /usr/bin/time -l "$@" > $log 2>&1
  echo "=== $name exit $? end $(date +%T)" >> $W/runs/chain2.status
}
summaries() {
  step "summary $1" $W/runs/summary_$1.txt $PY $W/summary.py
  step "hosts summary $1" $W/runs/hosts_summary_$1.txt $PY $W/hosts_summary.py
  step "plots $1" $W/runs/plots_$1.log $PY $W/plot/profile257.py
}
step "pf member_249" $W/runs/pf_member_249.log $PY $W/inject_deck_pf.py $U member_249 $W/runs/pf_member_249.npz
for spec in "milestone:member_257@15000000" member_238 member_249; do
  tag=${spec//[:@]/_}
  step "hosts $spec" $W/runs/hosts_$tag.log $PY $W/inject_hosts.py $U "$spec" $W/runs/hosts_$tag.npz
done
summaries a
step "pf member_228" $W/runs/pf_member_228.log $PY $W/inject_deck_pf.py $U member_228 $W/runs/pf_member_228.npz
summaries b
step "gain 257" $W/runs/gain_257_15M.log $PY $W/gain_run.py $U "milestone:member_257@15000000" $W/runs/gain_257_15M
step "gain 238 check" $W/runs/gain_238_check.log $PY $W/gain_run.py $U member_238 $W/runs/gain_238_check 2
step "gain profile" $W/runs/gain_profile.txt $PY $W/gain_profile.py $U $W/runs
echo "=== chain2 done $(date +%T)" >> $W/runs/chain2.status
