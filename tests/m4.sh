#!/bin/bash
# M4: routed MoE of all 48 layers, one token at a time, real routing (owner's held-out chat trace), owner's profile
# for the hot set under llama.cpp's 35000 MiB budget. MoE ms/token must be <= 70% of llama.cpp pinned stack's routed-MoE
# ms/token (measured by scripts/n7_moe_cost.sh: normal minus EDX_NOMOE, controls on both sides).
set -e; cd "$(dirname "$0")/.."
[ -f results/model.txt ] || pixi run python tools/build_model.py > /dev/null
[ -f results/llamacpp_moe.json ] || pixi run python tools/llama_moe_cost.py > /dev/null
pixi run mojo build -I src src/test_m4.mojo -o build/test_m4 2>&1 | grep -E ' error' && exit 1
timeout 900 ./build/test_m4 300 100 | tail -1 > results/m4.json
python3 - <<'PY'
import json, sys
m = json.load(open("results/m4.json")); b = json.load(open("results/llamacpp_moe.json"))
r = m["moe_ms_per_token"] / b["moe_ms_per_token"]
print(f"Mojo MoE {m['moe_ms_per_token']:.1f} ms/token vs llama.cpp {b['moe_ms_per_token']:.1f} ({100*r:.0f}%); "
      f"{m['cold_per_token']:.1f} cold experts/token on the CPU ({m['cpu_ms_per_token']:.1f} ms)")
sys.exit(0 if r <= 0.70 else 1)
PY
