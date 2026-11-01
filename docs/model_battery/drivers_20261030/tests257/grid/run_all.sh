#!/bin/zsh
# Sequential: one python/TF process at a time.
cd /Users/alarion239/Desktop/EuclidPolish
G=/private/tmp/claude-502/-Users-alarion239-Desktop-EuclidPolish/6d444c86-12c6-4a7f-b4eb-7b73fa2e1c66/scratchpad/tests257/grid
for spec in "$@"; do
  ~/miniforge3/envs/EuclidPolishEnv/bin/python $G/measure.py "$spec" 2>&1 | grep -v "Warning\|tf_record\|warnings.warn" >> $G/run.log
done
echo ALLDONE >> $G/run.log
