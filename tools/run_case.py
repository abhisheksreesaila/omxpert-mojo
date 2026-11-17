"""Run a compiled test binary on reference cases of a layer.  usage: run_case.py <binary> <layer> <case> <out.bin>"""
import json, subprocess, sys
b, layer, case, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
extra = sys.argv[5:]
m = json.load(open(f"results/layer{layer}.json"))
args = [b, m["gate"]["file"], m["gate"]["offset"], m["up"]["file"], m["up"]["offset"], m["down"]["file"], m["down"]["offset"],
        int(m["down"]["type"] == "Q8_0"), m["gate"]["expert_bytes"], m["down"]["expert_bytes"], f"results/layer{layer}_case{case}.bin", out] + extra
r = subprocess.run([str(x) for x in args], capture_output=True, text=True)
print(r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-2000:])
