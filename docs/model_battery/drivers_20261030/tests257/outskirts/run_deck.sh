#!/bin/zsh
# Sequential (one TF process at a time) re-run of the outskirt injection deck test.
W=/private/tmp/claude-502/-Users-alarion239-Desktop-EuclidPolish/6d444c86-12c6-4a7f-b4eb-7b73fa2e1c66/scratchpad/tests257/outskirts
U=/private/tmp/claude-502/-Users-alarion239-Desktop-EuclidPolish/d1a24d4c-2869-4369-a6ae-08d92714c971/scratchpad/underflux/out
export PYTHONPATH=/Users/alarion239/Desktop/EuclidPolish KMP_DUPLICATE_LIB_OK=TRUE
cd /Users/alarion239/Desktop/EuclidPolish
for spec in "$@"; do
  tag=${spec//[:@]/_}
  echo "=== $spec  start $(date +%T)" >> $W/runs/run_deck.status
  /usr/bin/time -l ~/miniforge3/envs/EuclidPolishEnv/bin/python $W/inject_deck.py $U "$spec" $W/runs/deck_$tag.npz > $W/runs/deck_$tag.log 2>&1
  echo "=== $spec  exit $? end $(date +%T)" >> $W/runs/run_deck.status
done
