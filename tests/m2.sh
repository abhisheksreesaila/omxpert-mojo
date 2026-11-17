#!/bin/bash
# M2: same Mojo source on CPU and GPU gives BIT-IDENTICAL output on 200 random tokens (layer 10: Q4_K + Q5_1, and
# layer 4: Q4_K + Q8_0), and the CPU reads at >= 60% of measured RAM bandwidth (26 GB/s -> 15.6), median of 5 runs of
# 200 random routings over the whole 1.57 GB layer (25x the L3, so RAM speed, not cache).
set -e; cd "$(dirname "$0")/.."
# timing is only meaningful on a quiet machine: wait (load < 3 for 3 min), as the llama.cpp benchmarks do
bash ../scripts/wait_quiet.sh
for L in 10 4; do [ -f results/layer$L.json ] || pixi run python tools/extract.py --layer $L --cases 1 > /dev/null; done
pixi run mojo build -I src src/test_m2.mojo -o build/test_m2 2>&1 | grep -E ' error' && exit 1
: > results/m2.jsonl
for i in 1 2 3 4 5; do pixi run python tools/run_m2.py build/test_m2 10 200 | tail -1 >> results/m2.jsonl; done
pixi run python tools/run_m2.py build/test_m2 4 200 | tail -1 | sed 's/^{/{"layer": 4, /' >> results/m2.jsonl
python3 - <<'PY'
import json, sys
rows = [json.loads(l) for l in open("results/m2.jsonl")]
l10 = [r for r in rows if "layer" not in r]; l4 = [r for r in rows if "layer" in r]
bits = all(r["bit_identical_cases"] == r["cases"] == 200 for r in rows)
med = sorted(r["cpu_gbs"] for r in l10)[len(l10) // 2]
print(f"bit-identical {sum(r['bit_identical_cases'] for r in rows)}/{sum(r['cases'] for r in rows)} tokens (Q5_1 and Q8_0 layers), "
      f"CPU median {med:.1f} GB/s ({100*med/26:.0f}% of 26)")
sys.exit(0 if bits and med >= 15.6 else 1)
PY
