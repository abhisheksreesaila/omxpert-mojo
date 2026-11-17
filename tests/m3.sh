#!/bin/bash
# M3: hybrid layer (router + top-k on GPU, hot experts GPU, cold experts CPU via the pinned-memory mailbox ring, no
# host sync per layer) is BIT-IDENTICAL to the all-GPU layer: 200 tokens, ~10% of experts cold, on layer 10 (Q5_1 down)
# and layer 4 (Q8_0 down). Records latency per token-layer next to llama.cpp's per-layer cost (41 ms / 48 = 0.85 ms).
set -e; cd "$(dirname "$0")/.."
for L in 10 4; do [ -f results/layer$L.json ] || pixi run python tools/extract.py --layer $L --cases 1 > /dev/null; done
pixi run mojo build -I src src/test_m3.mojo -o build/test_m3 2>&1 | grep -E ' error' && exit 1
: > results/m3.jsonl
for L in 10 4; do timeout 300 pixi run python tools/run_m3.py build/test_m3 $L 200 10 | tail -1 | sed "s/^{/{\"layer\": $L, /" >> results/m3.jsonl; done
python3 - <<'PY'
import json, sys
rows = [json.loads(l) for l in open("results/m3.jsonl")]
ok = all(r["bit_identical"] == r["tokens"] == 200 for r in rows)
print("; ".join(f"L{r['layer']}: {r['bit_identical']}/{r['tokens']} identical, all-GPU {r['all_gpu_us']:.0f} us, "
                f"hybrid {r['hybrid_us']:.0f} us/token-layer (llama.cpp ~850 incl. attention)" for r in rows))
sys.exit(0 if ok else 1)
PY
