"""Compare label-logit gaps: ref (vocabulary logprobs) vs fork (T*ln p ratios)."""
import json, math, sys, collections
T = 1.65
analysis = json.load(open(sys.argv[1]))
buckets = collections.Counter(); worst = []
for item in analysis["per_item"]:
    ref, fork = item["ref"], item["fork"]
    lp = ref["logprobs"]; names = list(lp)
    top = max(names, key=lambda n: lp[n])
    deltas = []
    for n in names:
        d_ref = lp[n] - lp[top]
        pf, pt = fork["vec"][n], fork["vec"][top]
        if pf <= 0 or pt <= 0:
            continue
        d_fork = T * math.log(pf / pt)
        deltas.append(abs(d_fork - d_ref))
    m = max(deltas)
    for edge, label in ((1e-3, "0 (identical)"), (0.0625 + 1e-3, "<=1/16"), (0.125 + 1e-3, "<=1/8"), (0.25 + 1e-3, "<=1/4"), (0.5 + 1e-3, "<=1/2"), (1e9, ">1/2")):
        if m <= edge:
            buckets[label] += 1; break
    worst.append((m, item["id"]))
n = len(analysis["per_item"])
order = ["0 (identical)", "<=1/16", "<=1/8", "<=1/4", "<=1/2", ">1/2"]
print({k: f"{buckets[k]} ({100*buckets[k]/n:.1f}%)" for k in order})
print(sorted(worst)[-5:])
