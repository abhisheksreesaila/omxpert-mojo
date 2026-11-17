#!/bin/bash
# Daily-driver candidates: Mojo layer (2048-token chunks, dual cold path) with and without MTP speculation
# (draft head's experts on the CPU). Short chat decode + decode after a 24K-token prompt.
set -u
D=~/Projects/expert-dispatch; M=$D/omxpert-mojo; O=$D/runs/2026-11-17-daily; PORT=18096; mkdir -p $O
MTPF=~/models/Qwen3.8-Flash-Next-GGUF/mtp-Qwen3.8-Flash-Next-Q8_0.gguf
BASEARGS="-fit off -ngl 99 --n-cpu-moe 48 -lzm on -t 6 --poll 0 -b 2048 -ub 2048"
for cfg in omx omx_mtp; do
  bash $D/scripts/wait_quiet.sh
  ARGS="$BASEARGS"
  [ $cfg = omx_mtp ] && ARGS="$BASEARGS -md $MTPF -ngld 99 -otd exps=CPU --spec-type draft-mtp --spec-draft-n-max 2"
  env OMX_PREFILL=1 OMX_BMAXT=2048 CTX=40960 STACK_ARGS="$ARGS" \
    STACK_ENV="OMX_LIB=$M/build/libomxmoe.so OMX_MODEL=$M/results/model_ub2k.txt OMX_PINS=$M/results/pins_ub2k.bin" \
    PORT=$PORT bash $D/omxpert/run.sh > $O/server_$cfg.log 2>&1 &
  SPID=$!
  for i in $(seq 1 600); do curl -sf http://127.0.0.1:$PORT/health | grep -q ok && break; kill -0 $SPID 2>/dev/null || { echo "$cfg failed to start" >> $O/status; break; }; sleep 2; done
  LPID=$(pgrep -n -x llama-server)
  cd $D && ROUNDS=1 NPRED=64 WARM=5 PROMPTS=chat pixi run python scripts/n6_bench.py http://127.0.0.1:$PORT warm $O/warm_$cfg.json > /dev/null 2>&1
  ROUNDS=2 NPRED=128 WARM=0 PROMPTS=chat pixi run python scripts/n6_bench.py http://127.0.0.1:$PORT $cfg $O/bench_$cfg.json > $O/bench_$cfg.out 2>&1
  echo "$cfg short: $(tail -1 $O/bench_$cfg.out)" >> $O/status
  pixi run python scripts/n7_longdecode.py http://127.0.0.1:$PORT $cfg $O/long_$cfg.json >> $O/status 2>&1
  kill $SPID; sleep 20; kill -0 $LPID 2>/dev/null && kill -9 $LPID
done
echo done > $O/done
