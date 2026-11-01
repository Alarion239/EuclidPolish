#!/bin/zsh
# Waits for run_chain.sh to finish (one TF process at a time), then the real-tile / synthetic-field host injection.
W=/private/tmp/claude-502/-Users-alarion239-Desktop-EuclidPolish/6d444c86-12c6-4a7f-b4eb-7b73fa2e1c66/scratchpad/tests257/outskirts
U=/private/tmp/claude-502/-Users-alarion239-Desktop-EuclidPolish/d1a24d4c-2869-4369-a6ae-08d92714c971/scratchpad/underflux/out
until grep -q "gain 238 check exit" $W/runs/run_gain.status 2>/dev/null; do sleep 5; done
export PYTHONPATH=/Users/alarion239/Desktop/EuclidPolish KMP_DUPLICATE_LIB_OK=TRUE
cd /Users/alarion239/Desktop/EuclidPolish
for spec in "milestone:member_257@15000000" member_238 member_249; do
  tag=${spec//[:@]/_}
  echo "=== $spec  start $(date +%T)" >> $W/runs/run_hosts.status
  /usr/bin/time -l ~/miniforge3/envs/EuclidPolishEnv/bin/python $W/inject_hosts.py $U "$spec" $W/runs/hosts_$tag.npz > $W/runs/hosts_$tag.log 2>&1
  echo "=== $spec  exit $? end $(date +%T)" >> $W/runs/run_hosts.status
done
