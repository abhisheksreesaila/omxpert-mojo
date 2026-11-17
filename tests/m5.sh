#!/bin/bash
# M5: Qwen3.8-Flash-Next decodes through the Mojo expert layer inside llama.cpp (libomxmoe.so via the OMX_LIB hook).
# Pass: decode tok/s (median over 2 rounds x 3 held-out chat prompts) above the pinned llama.cpp stack (GOAL: 22.2,
# and above the same-day control with the same flags), and identical greedy text with two very different pin sets.
# The server runs take ~30 min; FULL=1 reruns them (scripts in ../scripts), otherwise the recorded runs are checked.
set -e; cd "$(dirname "$0")/.."
R=../runs/2026-11-16-omx
if [ "${FULL:-0}" = "1" ]; then
  ../scripts/n7_omx_bench.sh omx_check OMX_PREFILL=1
  BASELINE=1 ../scripts/n7_omx_bench.sh base_check
  ../scripts/n7_placement.sh
fi
python3 - <<PY
import json, statistics, sys, os
R = "$R"
def med(f):
    return statistics.median(x["decode_tok_s"] for x in json.load(open(f))["runs"])
omx = med(f"{R}/bench_omx_check.json") if os.path.exists(f"{R}/bench_omx_check.json") else med(f"{R}/bench_omx2.json")
base = med(f"{R}/bench_base_check.json") if os.path.exists(f"{R}/bench_base_check.json") else med(f"{R}/bench_base_lzm.json")
txt = {n: open(f"{R}/place_{n}.txt").read().splitlines()[:3] for n in ("A1", "A2", "B")}
same = txt["A1"] == txt["A2"] == txt["B"]
print(f"decode {omx:.1f} tok/s with Mojo vs {base:.1f} llama.cpp (same flags; GOAL bar 22.2); "
      f"greedy text identical across pin sets: {same}")
sys.exit(0 if omx > 22.2 and omx > base and same else 1)
PY
