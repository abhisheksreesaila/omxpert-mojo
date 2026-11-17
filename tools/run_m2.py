"""usage: run_m2.py <binary> <layer> <n_cases>"""
import json, subprocess, sys
b, layer, n = sys.argv[1], int(sys.argv[2]), sys.argv[3]
m = json.load(open(f"results/layer{layer}.json"))
args = [b, m["gate"]["file"], m["gate"]["offset"], m["up"]["file"], m["up"]["offset"], m["down"]["file"], m["down"]["offset"],
        int(m["down"]["type"] == "Q8_0"), m["gate"]["expert_bytes"], m["down"]["expert_bytes"], n]
r = subprocess.run([str(x) for x in args], capture_output=True, text=True)
print(r.stdout.strip() if r.stdout.strip() else r.stderr[-2000:])
