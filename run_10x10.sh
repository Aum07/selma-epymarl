#!/usr/bin/env bash
# =============================================================================
# SELMA V2 mechanism replication on 10x10-3p-3f  (plain, NOT -coop)
#
# WHY THIS RUN EXISTS
#   Not to win on performance. The offered plans on 10x10 are still permutations
#   over the loading cells of ONE food, and those cells are always neighbours of
#   that food: measured over 2,861 real detector events, the max distance between
#   candidate cells is 2 units on 100% of them. With d_max=19 instead of 15, the
#   per-agent Phi spread is 0.1053 on 10x10 against 0.1333 on 8x8 -- a SMALLER
#   lever than the one that did nothing on 8x8.
#
#   What this DOES buy, and it is worth buying:
#     * the plan-conditionality result replicated on a second layout, with 3
#       agents and 12-plan answer sets instead of 2 agents and 6
#     * a second environment in the paper, which is currently its single most
#       prominent limitation
#     * an honest floor-effect report at 2M, consistent with everything already
#       claimed
#
#   Report it as MECHANISM, not as a performance win. If it happens to lift
#   success, that is a bonus, not the thesis.
#
# NO CODE IS MODIFIED. Same repo, same branch, same commit, same tests.
# Only the environment key, the horizon and the output directory change.
#
# USAGE -- run smoke FIRST, on one laptop, and read its output before anything
#          else. It takes about two minutes.
#
#   bash run_10x10.sh smoke                 # proves the 3-agent path works
#   PROVIDER=openai_ascii bash run_10x10.sh smoke   # same, via the v4 payload
#
#   # then split across the two laptops:
#   LAPTOP A:  bash run_10x10.sh baseline random
#   LAPTOP B:  source ~/.config/selma/openai.env
#              bash run_10x10.sh llm llm_ascii
#
#   Raise parallelism only while watching memory:  MAXPAR=3 bash run_10x10.sh ...
# =============================================================================
set -euo pipefail

REPO="${REPO:-$HOME/selma/epymarl}"
PY="${PY:-$HOME/selma/venv/bin/python3}"
OUT="${OUT:-$REPO/analysis/out/v2_10x10}"

ENVKEY="${ENVKEY:-lbforaging:Foraging-10x10-3p-3f-v3}"   # PLAIN. never -coop.
TMAX="${TMAX:-2000000}"
SEEDS="${SEEDS:-1 2 3}"
BETA="${BETA:-0.05}"
MAXPAR="${MAXPAR:-2}"          # memory pressure killed a Stage E wave at ~57%

mkdir -p "$OUT"
cd "$REPO"

# ---- guard: -coop collapses the answer set to one plan. refuse it outright. --
case "$ENVKEY" in
  *-coop-*) echo "REFUSING: $ENVKEY is a -coop variant, which leaves nothing to select."; exit 2 ;;
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
selma_provider=$provider selma_potential_mode=plan selma_beta=$BETA \
selma_guidance_log_csv=$dir/guidance.csv selma_decision_log_csv=$dir/decisions.csv"
}

llm_args () {
  local dir="$1"
  echo "selma_cache_mode=read_write selma_cache_path=$dir/guidance_cache.sqlite3 \
selma_live_call_cap=13500 selma_cost_cap_usd=2.50 \
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
  echo "smoke: 20k steps, seed 91, provider=$PROV. ~2 minutes."
  "$PY" src/main.py --config=qmix --env-config=gymma with \
    env_args.key="$ENVKEY" env_args.time_limit=50 \
    seed=91 t_max=20000 use_cuda=False common_reward=True \
    selma_log_csv="$d/episodes.csv" local_results_path="$d/results" \
    test_interval=10000 test_nepisode=10 log_interval=10000 \
    learner_log_interval=10000 runner_log_interval=10000 \
    $(guided_args "$d" "$PROV") $EXTRA > "$d/run.log" 2>&1 \
    || { echo; echo "SMOKE FAILED -- tail of log:"; tail -30 "$d/run.log"; exit 1; }

  echo
  if [ ! -s "$d/decisions.csv" ]; then
    echo "SMOKE FAILED: no decisions were logged."
    echo "The detector may not fire in 20k steps on this layout -- retry with"
    echo "  TMAX=60000 bash run_10x10.sh smoke"
    echo "If it still logs nothing, STOP. Do not launch the wave."
    exit 1
  fi
  python3 - "$d/decisions.csv" <<'PY'
import csv, sys, collections
rows = list(csv.DictReader(open(sys.argv[1])))
print(f"decisions logged      : {len(rows)}")
cand = [int(r[k]) for r in rows for k in r if 'candidate' in k and 'count' in k and r[k].isdigit()]
if cand:
    c = collections.Counter(cand)
    print(f"candidate-count dist  : {sorted(c.items())}")
    print(f"multi-plan decisions  : {sum(v for k,v in c.items() if k>1)}/{len(cand)}")
print("\nSMOKE PASSED -- the 3-agent plan path works on 10x10. Launch the wave.")
PY
  exit 0
fi

# ------------------------------------------------------------- conditions ----
[ $# -ge 1 ] || { echo "usage: bash $0 smoke | <condition...>   (baseline heuristic random llm llm_ascii)"; exit 2; }

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
    llm_ascii) : "${OPENAI_API_KEY:?source ~/.config/selma/openai.env first}"
               for s in $SEEDS; do d="$OUT/llm_ascii_seed${s}"; mkdir -p "$d"
                 launch llm_ascii "$s" $(guided_args "$d" openai_ascii) $(llm_args "$d"); done ;;
    *) echo "unknown condition: $cond"; exit 2 ;;
  esac
done
wait
echo
echo "complete: $*   ->  $OUT"
echo "progress at any time:  ls $OUT/*/DONE 2>/dev/null | wc -l"
