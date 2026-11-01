cd /Users/alarion239/Desktop/EuclidPolish
export PYTHONPATH=/Users/alarion239/Desktop/EuclidPolish/.claude/worktrees/pixel-input-knee-dependency-efc29d KMP_DUPLICATE_LIB_OK=TRUE
IDS=f200w-0070,f200w-0040,f200w-0420,f200w-0141,f200w-0325,f200w-0106,f200w-0080,f200w-0320,f200w-0271,f200w-0260,f200w-0238,f200w-0102,f200w-0210
for B in H_E J_E Y_E VIS; do
  /Users/alarion239/miniforge3/envs/EuclidPolishEnv/bin/python -u /Users/alarion239/Desktop/EuclidPolish/.claude/worktrees/pixel-input-knee-dependency-efc29d/scripts/jwst_resolution.py --band $B --tile-ids $IDS     --models lr,universal:member_249,universal:member_218,universal:member_227,production,milestone:member_257@4000000,milestone:member_257@8000000     --cache data/jwst_euclid_overlap/resolution_cache --out data/jwst_euclid_overlap/galaxies/resolution_$B.json
done
