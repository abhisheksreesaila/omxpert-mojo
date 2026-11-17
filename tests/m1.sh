#!/bin/bash
# M1: one real layer (10) of Qwen3.8-Flash-Next on the GPU: rel. error vs float64 reference <= 1e-3 and the expert bytes
# read at >= 50% of measured VRAM bandwidth (838 GB/s -> 419 GB/s), timed over 256 random top-10 routings.
set -e; cd "$(dirname "$0")/.."
# timing is only meaningful on a quiet machine: wait (load < 3 for 3 min), as the llama.cpp benchmarks do
bash ../scripts/wait_quiet.sh
[ -f results/layer10.json ] || pixi run python tools/extract.py --layer 10 --cases 1 > /dev/null
pixi run mojo build -I src src/test_gpu.mojo -o build/test_gpu 2>&1 | grep -E ' error' && exit 1
out=$(pixi run python tools/run_case.py build/test_gpu 10 0 results/gpu_l10_c0.bin)
echo "$out" > results/m1.json
python3 -c "
import json,sys; r=json.loads(open('results/m1.json').read())
ok = r['rel_err'] <= 1e-3 and r['gbs'] >= 419
print(f\"rel_err {r['rel_err']:.2e}, {r['us_per_token_layer']:.1f} us/layer, {r['gbs']:.0f} GB/s ({100*r['gbs']/838:.0f}% of 838)\")
sys.exit(0 if ok else 1)"
