#!/bin/zsh
# Sky > Compare metrics re-run (in memory) for production (238) and 257@15M:
# 13 NEXUS galaxy tiles of record 20261028-134110-710beb, then the poster tile of 20261028-134110-f0d0bb.
cd /Users/alarion239/Desktop/EuclidPolish || exit 1
D=/private/tmp/claude-502/-Users-alarion239-Desktop-EuclidPolish/6d444c86-12c6-4a7f-b4eb-7b73fa2e1c66/scratchpad/tests257/realsky
export PYTHONPATH=/Users/alarion239/Desktop/EuclidPolish KMP_DUPLICATE_LIB_OK=TRUE
PY=/Users/alarion239/miniforge3/envs/EuclidPolishEnv/bin/python
SPECS=production,milestone:member_257@15000000
echo "=== nexus start $(date '+%H:%M:%S')"
$PY $D/holes_run.py --record 20261028-134110-710beb.json --specs $SPECS --out $D/holes_nexus.json > $D/holes_nexus.log 2>&1
echo "=== nexus exit $? $(date '+%H:%M:%S')"
echo "=== poster start $(date '+%H:%M:%S')"
$PY $D/holes_run.py --record 20261028-134110-f0d0bb.json --specs $SPECS --out $D/holes_poster.json --save-sr $D/sr > $D/holes_poster.log 2>&1
echo "=== poster exit $? $(date '+%H:%M:%S')"
