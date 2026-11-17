#!/bin/bash
# M8: the decode layer runs on llama.cpp's own CUDA stream (no CPU stop per layer, CUDA graphs on).
# Pass: (1) harness tools/probe/stream_layer.c: in-stream output bit-identical to the host path per layer, in a chain,
# unwaited, and under CUDA graph replay; (2) recorded back-to-back runs (../scripts/n8_final.sh): the new default
# decodes faster than last night's default on both new runs, and greedy text is byte-identical across in-stream /
# host path / bigger hot set (../scripts/n8_stream_run.sh). The harness needs both GPUs idle (~1 min).
set -e; cd "$(dirname "$0")/.."
pgrep -x llama-server > /dev/null && { echo "a llama-server is running; stop it first"; exit 1; }
gcc -O1 tools/probe/stream_layer.c -I/opt/cuda/include -L/opt/cuda/lib64 -lcudart -ldl -lpthread -o build/stream_layer
LD_LIBRARY_PATH=/opt/cuda/lib64:.pixi/envs/default/lib ./build/stream_layer $PWD/build/libomxmoe.so \
  results/model_full.txt results/pins_full.bin | grep -E "^[1-4]\." | tee build/m8_harness.txt
python3 - <<'PY'
import json, re, statistics, sys
h = open("build/m8_harness.txt").read()
fr = [tuple(map(int, m)) for m in re.findall(r"(\d+) / (\d+)", h)]
bits = len(fr) == 3 and all(a == b for a, b in fr) and "2. 24-layer chain on GPU 0: bit-identical" in h
R, S = "../runs/2026-11-17-m8final", "../runs/2026-11-17-stream"
def med(f): return statistics.median(x["decode_tok_s"] for x in json.load(open(f))["runs"])
new, old, new2 = (med(f"{R}/bench_{c}.json") for c in ("new", "old", "new2"))
t = {n: [x["text"] for x in json.load(open(f"{S}/text_{n}.json"))] for n in ("host", "stream", "full")}
same = t["host"] == t["stream"] == t["full"]
print(f"harness bit-identical: {bits}; decode new {new:.1f} / old {old:.1f} / new {new2:.1f} tok/s; greedy text identical: {same}")
sys.exit(0 if bits and new > old and new2 > old and same else 1)
PY
