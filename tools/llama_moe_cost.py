"""llama.cpp pinned stack: routed-MoE ms/token = 1000/tps(normal) - 1000/tps(nomoe), with a normal run on each side
(control, candidate, control). Writes results/llamacpp_moe.json from ../runs/2026-11-16-moecost/bench_*.json."""
import json, statistics, sys
D = "../runs/2026-11-16-moecost"
def tps(m):
    r = json.load(open(f"{D}/bench_{m}.json"))
    return statistics.median(x["decode_tok_s"] for x in r["runs"])
n1, nm, n2 = tps("normal"), tps("nomoe"), tps("normal2")
normal = (n1 + n2) / 2
moe_ms = 1000 / normal - 1000 / nm
out = {"tps_normal_before": n1, "tps_nomoe": nm, "tps_normal_after": n2,
       "ms_per_token_normal": 1000 / normal, "ms_per_token_nomoe": 1000 / nm, "moe_ms_per_token": moe_ms}
json.dump(out, open("results/llamacpp_moe.json", "w"), indent=1)
print(json.dumps(out))
