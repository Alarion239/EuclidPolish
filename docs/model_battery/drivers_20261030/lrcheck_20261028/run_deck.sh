export PYTHONPATH=/Users/alarion239/Desktop/EuclidPolish:/private/tmp/claude-502/-Users-alarion239-Desktop-EuclidPolish/d1a24d4c-2869-4369-a6ae-08d92714c971/scratchpad/underflux KMP_DUPLICATE_LIB_OK=TRUE
cd /Users/alarion239/Desktop/EuclidPolish
for m in member_238 member_228 member_229 member_249 member_258; do
  ~/miniforge3/envs/EuclidPolishEnv/bin/python /private/tmp/claude-502/-Users-alarion239-Desktop-EuclidPolish/d7b359b4-16d6-4f7f-aa2b-959d682a91bd/scratchpad/lrcheck/inject_deck.py /private/tmp/claude-502/-Users-alarion239-Desktop-EuclidPolish/d1a24d4c-2869-4369-a6ae-08d92714c971/scratchpad/underflux/out $m /private/tmp/claude-502/-Users-alarion239-Desktop-EuclidPolish/d7b359b4-16d6-4f7f-aa2b-959d682a91bd/scratchpad/lrcheck/deck_$m.npz > /private/tmp/claude-502/-Users-alarion239-Desktop-EuclidPolish/d7b359b4-16d6-4f7f-aa2b-959d682a91bd/scratchpad/lrcheck/deck_$m.log 2>&1
done
