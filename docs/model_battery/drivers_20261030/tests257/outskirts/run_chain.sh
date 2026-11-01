#!/bin/zsh
# Sequential: per-field-sum deck reruns (bootstrap), then the real-sky gain arrays for 257@15M (+ a 2-tile 238 check).
W=/private/tmp/claude-502/-Users-alarion239-Desktop-EuclidPolish/6d444c86-12c6-4a7f-b4eb-7b73fa2e1c66/scratchpad/tests257/outskirts
U=/private/tmp/claude-502/-Users-alarion239-Desktop-EuclidPolish/d1a24d4c-2869-4369-a6ae-08d92714c971/scratchpad/underflux/out
$W/run_pf.sh "milestone:member_257@15000000" member_238 member_249 member_228
export PYTHONPATH=/Users/alarion239/Desktop/EuclidPolish KMP_DUPLICATE_LIB_OK=TRUE
cd /Users/alarion239/Desktop/EuclidPolish
echo "=== gain 257 start $(date +%T)" >> $W/runs/run_gain.status
/usr/bin/time -l ~/miniforge3/envs/EuclidPolishEnv/bin/python $W/gain_run.py $U "milestone:member_257@15000000" $W/runs/gain_257_15M > $W/runs/gain_257_15M.log 2>&1
echo "=== gain 257 exit $? end $(date +%T)" >> $W/runs/run_gain.status
/usr/bin/time -l ~/miniforge3/envs/EuclidPolishEnv/bin/python $W/gain_run.py $U member_238 $W/runs/gain_238_check 2 > $W/runs/gain_238_check.log 2>&1
echo "=== gain 238 check exit $? end $(date +%T)" >> $W/runs/run_gain.status
