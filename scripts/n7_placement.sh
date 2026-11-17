#!/bin/bash
# omxpert-mojo M5 placement invariance: greedy text with pin set A (normal), A again, and B (about half the hot
# experts: thousands of experts move from GPU to CPU). Texts must be identical.
set -u
D=~/Projects/expert-dispatch; M=$D/omxpert-mojo; O=$D/runs/2026-11-16-omx; PORT=18094
for run in A1:_m5 A2:_m5 B:_half; do
  name=${run%%:*}; tag=${run##*:}
  bash $D/scripts/wait_quiet.sh
  OMX_PREFILL=1 STACK_ARGS="-fit off -ngl 99 --n-cpu-moe 48 -lzm on -t 6 --poll 0" \
    STACK_ENV="OMX_LIB=$M/build/libomxmoe.so OMX_MODEL=$M/results/model$tag.txt OMX_PINS=$M/results/pins$tag.bin" \
    PORT=$PORT bash $D/omxpert/run.sh > $O/server_place_$name.log 2>&1 &
  SPID=$!
  for i in $(seq 1 600); do curl -sf http://127.0.0.1:$PORT/health | grep -q ok && break; sleep 2; done
  LPID=$(pgrep -n -x llama-server)
  : > $O/place_$name.txt
  for p in "The three most important ideas in the history of computing are" "def merge_sort(xs):" "Explain why the sky is blue to a ten year old."; do
    curl -s http://127.0.0.1:$PORT/completion -H 'Content-Type: application/json' \
      -d "$(python3 -c 'import json,sys; print(json.dumps({"prompt": sys.argv[1], "n_predict": 96, "temperature": 0, "cache_prompt": False}))' "$p")" \
      | python3 -c 'import json,sys; print(json.dumps(json.load(sys.stdin)["content"]))' >> $O/place_$name.txt
  done
  grep 'omx:' $O/server_place_$name.log | tail -1 >> $O/place_$name.txt
  kill $SPID; sleep 20; kill -0 $LPID 2>/dev/null && kill -9 $LPID
done
echo done > $O/place_done
