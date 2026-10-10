import json, sys
for f in sys.argv[1:]:
    s = json.load(open(f))["summary"]
    print(f)
    for k in ("all", "jevbench-easy", "jevbench-original", "jevbench-hard", "internal-eval-v1", "type:choice", "type:noul", "type:score"):
        v = s[k]
        print(f"  {k:18s} n={v['n']:4d} agree={v['argmax_agreement_pct']:.2f} meanDp={v['mean_abs_dp']:.4f} maxDp={v['max_abs_dp']:.4f} "
              f"accRef={v['acc_ref_pct']:.2f} accFork={v['acc_fork_pct']:.2f} p50 {v['lat_ref_p50_ms']:.1f}/{v['lat_fork_p50_ms']:.1f} "
              f"p95 {v['lat_ref_p95_ms']:.1f}/{v['lat_fork_p95_ms']:.1f}")
