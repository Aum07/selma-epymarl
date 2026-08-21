#!/usr/bin/env python3
"""Did the OpenAI path actually work under the V3 contract?"""
import collections, csv, os, sys
d = sys.argv[1] if len(sys.argv) > 1 else "."
rows = list(csv.DictReader(open(os.path.join(d, "decisions.csv"))))
print(f"decisions: {len(rows)}\n")

def dist(col):
    c = collections.Counter(r.get(col, "") for r in rows)
    return sorted(c.items(), key=lambda x: -x[1])[:6]

for col in ("decision", "outcome", "cache_result", "provider_considered",
            "validator_valid", "validator_reason", "physically_valid",
            "activated", "returned_model", "fallback_reason"):
    if rows and col in rows[0]:
        print(f"{col:22}: {dist(col)}")

def ints(col):
    out = []
    for r in rows:
        v = str(r.get(col, "")).strip()
        if v in ("", "0", "n/a", "N/A", "None", "-"):
            continue
        try:
            out.append(int(float(v)))
        except ValueError:
            pass
    return out

toks, outs = ints("tokens_input"), ints("tokens_output")
print(f"\nrows with non-zero input tokens : {len(toks)}")
if toks:
    print(f"  mean input tokens  : {sum(toks)/len(toks):.0f}")
    print(f"  mean output tokens : {sum(outs)/len(outs):.0f}" if outs else "  NO OUTPUT TOKENS")
    cost = sum(toks)/1e6*0.20 + sum(outs)/1e6*1.25
    print(f"  smoke cost         : ${cost:.4f}")
    n = len(toks)
    print(f"  calls so far       : {n}")
else:
    print("  *** NO LIVE CALLS. The provider was never reached. ***")

sel = [r for r in rows if r.get("selected_plan_id", "").strip()]
print(f"\nrows with a selected_plan_id    : {len(sel)}")
bad = [r for r in rows if r.get("validator_valid", "").strip().lower() in ("false", "0")]
print(f"validator failures              : {len(bad)}")
if bad:
    print("  reasons:", collections.Counter(r.get("validator_reason") for r in bad).most_common(5))
print("""
PASS if: non-zero input tokens, output tokens > 0, validator failures ~0,
         and selected_plan_id present on the multi-plan decisions.
FAIL if: 'NO LIVE CALLS', or every outcome is an error/fallback.
""")
