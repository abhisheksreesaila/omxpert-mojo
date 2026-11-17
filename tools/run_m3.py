"""usage: run_m3.py <binary> <layer> <n_tokens> <cold_pct>"""
import json, subprocess, sys
b, layer, n, cold = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
m = json.load(open(f"results/layer{layer}.json"))
args = [b, m["gate"]["file"], m["gate"]["offset"], m["up"]["file"], m["up"]["offset"], m["down"]["file"], m["down"]["offset"],
        int(m["down"]["type"] == "Q8_0"), m["gate"]["expert_bytes"], m["down"]["expert_bytes"],
        m["router"]["file"], m["router"]["offset"], n, cold]
r = subprocess.run([str(x) for x in args], capture_output=True, text=True, timeout=600)
print(r.stdout.strip() if r.stdout.strip() else r.stderr[-3000:])
