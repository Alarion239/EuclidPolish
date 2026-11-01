#!/bin/zsh
# JWST resolution re-run (current code) for production (238) and 257@15M, one band per process, sequential.
cd /Users/alarion239/Desktop/EuclidPolish || exit 1
OUT=/private/tmp/claude-502/-Users-alarion239-Desktop-EuclidPolish/6d444c86-12c6-4a7f-b4eb-7b73fa2e1c66/scratchpad/tests257/realsky/jwst
export PYTHONPATH=/Users/alarion239/Desktop/EuclidPolish
PY=/Users/alarion239/miniforge3/envs/EuclidPolishEnv/bin/python
MODELS=${MODELS:-production,milestone:member_257@15000000}
for band in H_E J_E Y_E VIS; do
  echo "=== $band start $(date '+%H:%M:%S')"
  $PY scripts/jwst_resolution.py --band $band --models "$MODELS" \
      --cache data/jwst_euclid_overlap/resolution_cache --out $OUT/resolution_$band.json > $OUT/resolution_$band.log 2>&1
  echo "=== $band exit $? $(date '+%H:%M:%S')"
done
