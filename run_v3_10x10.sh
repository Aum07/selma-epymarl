#!/usr/bin/env bash
# =============================================================================
# SELMA V3 -- target allocation on 10x10-3p-3f (plain, NOT -coop)
#
# Branch: selma-v3 @ ac36e7a   Worktree: ~/selma/v3   847 tests green
#
# WHAT IS DIFFERENT FROM THE V2 RUNS
#   V2 asked: "which of the 3 cells around THIS food should each agent take?"
#             -> candidate cells are always neighbours of one food, so the
#                largest distance between any two offered plans is 2 units,
#                measured on 100% of 2,861 real detector events.
#                Phi spread across the answer set = 2/19 = 0.105.
#   V3 asks:  "which agent should go to which FOOD?"
#             -> a wrong choice costs a traversal, not two steps.
#                Phi spread across the answer set is up to ~1.0.
#   That is roughly a 10x larger lever. It is the change that makes selection
#   quality able to matter at all.
#
# TWO FLAGS THAT MUST NOT BE OMITTED
#   selma_guidance_contract=v2_target_allocation
#       Defaults to v1_loading_cell. Omit it and you silently run the SMALL
#       lever -- the runs complete, the logs look normal, and the experiment
#       answers the question you already answered.
#   selma_ttl_steps=20
#       Defaults to 15, which is the frozen 8x8 value. d_max on 10x10 is 19,
#       so with TTL 15 an agent assigned to a far food loses its guidance
#       BEFORE arriving. Under target allocation the journeys are long by
#       design, so this default would gut the effect being tested.
#       20 = d_max(19) + 1 load allowance. Not calibrated from a shadow run --
#       there was no time -- so DISCLOSE it as a design choice, not a
#       measurement, and report the realised journey lengths from the logs.
#
#   NOTE: selma_potential_mode does NOT exist on this branch. That flag came
#   from the V2 patch; V3 replaces it with the contract above. Passing it
#   fails the run.
#
# SPLIT BY SEED, NOT BY CONDITION
#   requirements.txt is unpinned, so two machines get different torch/numpy
#   builds. Splitting by condition would confound library version with
#   condition. Splitting by seed keeps every paired contrast inside one machine.
#
#     LAPTOP A:  SEEDS="1 2" bash run_v3_10x10.sh baseline heuristic random llm
#     LAPTOP B:  SEEDS="3 4" bash run_v3_10x10.sh baseline heuristic random llm
#
# USAGE
#   bash run_v3_10x10.sh smoke      # ALWAYS first. Proves the big lever exists.
# =============================================================================
set -euo pipefail

REPO="${REPO:-$HOME/selma/v3}"
PY="${PY:-$HOME/selma/venv/bin/python3}"
OUT="${OUT:-$REPO/analysis/out/v3_10x10}"

ENVKEY="${ENVKEY:-lbforaging:Foraging-10x10-3p-3f-v3}"   # PLAIN. never -coop.
TMAX="${TMAX:-2500000}"
SEEDS="${SEEDS:-1 2}"
BETA="${BETA:-0.05}"
TTL="${TTL:-20}"
CONTRACT="${CONTRACT:-v2_target_allocation}"
MAXPAR="${MAXPAR:-2}"

mkdir -p "$OUT"
cd "$REPO"

case "$ENVKEY" in
  *-coop-*) echo "REFUSING: $ENVKEY is -coop; target allocation collapses to 1 plan."; exit 2 ;;
esac

launch () {
  local cond="$1" seed="$2"; shift 2
  local dir="$OUT/${cond}_seed${seed}"
  if [ -f "$dir/DONE" ]; then echo "skip  $cond seed=$seed (already done)"; return; fi
  mkdir -p "$dir"
  echo "start $cond seed=$seed  ->  $dir"
  (
    "$PY" src/main.py --config=qmix --env-config=gymma with \
      env_args.key="$ENVKEY" env_args.time_limit=50 \
      seed="$seed" t_max="$TMAX" use_cuda=False common_reward=True \
      selma_log_csv="$dir/episodes.csv" \
      local_results_path="$dir/results" \
      test_interval=50000 test_nepisode=100 log_interval=50000 \
      learner_log_interval=50000 runner_log_interval=50000 \
      "$@" > "$dir/run.log" 2>&1 && touch "$dir/DONE" \
      || echo "FAILED $cond seed=$seed -- see $dir/run.log"
  ) &
  while [ "$(jobs -rp | wc -l)" -ge "$MAXPAR" ]; do sleep 20; done
}

guided_args () {
  local dir="$1" provider="$2"
  echo "selma_mode=active selma_variant_id=wf15_m3_t5_ibon \
selma_provider=$provider selma_guidance_contract=$CONTRACT \
selma_ttl_steps=$TTL selma_beta=$BETA \
selma_guidance_log_csv=$dir/guidance.csv selma_decision_log_csv=$dir/decisions.csv"
}

# Caps raised for the longer horizon and the larger V3 payload.
# ~3.1 decisions per 1,000 steps measured => ~7,750 calls at 2.5M.
# A cap hit ABORTS the run, so these are set with headroom, not to the bone.
llm_args () {
  local dir="$1"
  echo "selma_cache_mode=read_write selma_cache_path=$dir/guidance_cache.sqlite3 \
selma_live_call_cap=20000 selma_cost_cap_usd=5.00 \
selma_price_table_date=2026-08-17 selma_price_input_usd_per_mtok=0.2 \
selma_price_cached_input_usd_per_mtok=0.02 selma_price_output_usd_per_mtok=1.25 \
selma_provider_timeout_s=30.0 selma_provider_max_retries=1 \
selma_provider_max_output_tokens=512"
}

# ---------------------------------------------------------------- smoke ------
if [ "${1:-}" = "smoke" ]; then
  d="$OUT/smoke"; rm -rf "$d"; mkdir -p "$d"
  PROV="${PROVIDER:-heuristic_min_travel}"
  EXTRA=""
  case "$PROV" in
    openai*) : "${OPENAI_API_KEY:?source ~/.config/selma/openai.env first}"
             EXTRA="$(llm_args "$d")" ;;
  esac
  echo "smoke: 30k steps, seed 91, provider=$PROV, contract=$CONTRACT, ttl=$TTL"
  "$PY" src/main.py --config=qmix --env-config=gymma with \
    env_args.key="$ENVKEY" env_args.time_limit=50 \
    seed=91 t_max=30000 use_cuda=False common_reward=True \
    selma_log_csv="$d/episodes.csv" local_results_path="$d/results" \
    test_interval=15000 test_nepisode=10 log_interval=15000 \
    learner_log_interval=15000 runner_log_interval=15000 \
    $(guided_args "$d" "$PROV") $EXTRA > "$d/run.log" 2>&1 \
    || { echo; echo "SMOKE FAILED -- tail of log:"; tail -40 "$d/run.log"; exit 1; }

  if [ ! -s "$d/decisions.csv" ]; then
    echo "SMOKE FAILED: no decisions logged. STOP -- do not launch."
    tail -20 "$d/run.log"; exit 1
  fi

  python3 - "$d/decisions.csv" "$d/guidance.csv" <<'PY'
import csv, sys, collections, os, ast
rows = list(csv.DictReader(open(sys.argv[1])))
print(f"\ndecisions logged        : {len(rows)}")
cols = rows[0].keys() if rows else []

# candidate-set size
cand = []
for r in rows:
    for k in r:
        if 'candidate' in k and ('count' in k or 'n_' in k) and str(r[k]).isdigit():
            cand.append(int(r[k]))
            break
if cand:
    c = collections.Counter(cand)
    print(f"answer-set size dist    : {sorted(c.items())}")
    multi = sum(v for k, v in c.items() if k > 1)
    print(f"multi-plan decisions    : {multi}/{len(cand)}")

# THE test that matters: do offered plans send agents to DIFFERENT foods?
foodcols = [k for k in cols if 'food' in k and ('target' in k or 'ids' in k)]
print(f"target-food columns     : {foodcols}")
distinct = 0
for r in rows:
    for k in foodcols:
        try:
            v = ast.literal_eval(r[k])
            if hasattr(v, '__len__') and len(set(v)) > 1:
                distinct += 1
                break
        except Exception:
            pass
print(f"decisions whose chosen plan spans >1 food : {distinct}/{len(rows)}")
print("""
WHAT TO CHECK
  answer-set size > 1 on most decisions  -> a choice exists
  plans spanning more than one food      -> the BIG lever is active
If every plan sits on a single food you are still on v1_loading_cell.
Check the run log for selma_guidance_contract before launching.
""")
PY
  grep -o "selma_guidance_contract[^,}]*" "$d/run.log" | head -2 || true
  grep -o "selma_ttl_steps[^,}]*" "$d/run.log" | head -2 || true
  echo "SMOKE COMPLETE -- read the two checks above before launching."
  exit 0
fi

# ------------------------------------------------------------- conditions ----
[ $# -ge 1 ] || { echo "usage: bash $0 smoke | baseline heuristic random llm"; exit 2; }

echo "contract=$CONTRACT  ttl=$TTL  beta=$BETA  t_max=$TMAX  seeds='$SEEDS'  out=$OUT"
for cond in "$@"; do
  case "$cond" in
    baseline)  for s in $SEEDS; do launch baseline "$s" selma_mode=off; done ;;
    heuristic) for s in $SEEDS; do d="$OUT/heuristic_seed${s}"; mkdir -p "$d"
                 launch heuristic "$s" $(guided_args "$d" heuristic_min_travel); done ;;
    random)    for s in $SEEDS; do d="$OUT/random_seed${s}"; mkdir -p "$d"
                 launch random "$s" $(guided_args "$d" random_valid_plan); done ;;
    llm)       : "${OPENAI_API_KEY:?source ~/.config/selma/openai.env first}"
               for s in $SEEDS; do d="$OUT/llm_seed${s}"; mkdir -p "$d"
                 launch llm "$s" $(guided_args "$d" openai) $(llm_args "$d"); done ;;
    *) echo "unknown condition: $cond"; exit 2 ;;
  esac
done
wait
echo
echo "complete: $*   ->  $OUT"
echo "progress:  ls $OUT/*/DONE 2>/dev/null | wc -l"
