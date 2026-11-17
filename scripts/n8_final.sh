#!/bin/bash
# M8 final: the candidate (STACK=mojo + OMX_STREAM=1 + full hot set) vs last night's default (host round trip per
# layer, _ub2k pins), back to back: new, old, new2. Short chat, 25K-context decode, and the 26.5K needle on new.
set -u
D=~/Projects/expert-dispatch; M=$D/omxpert-mojo/results; O=$D/runs/2026-11-17-m8final; PORT=18097; mkdir -p $O
for cfg in new old new2; do
  bash $D/scripts/wait_quiet.sh
  if [ $cfg = old ]; then
    env STACK=mojo OMX_STREAM=0 OMX_MODEL=$M/model_ub2k.txt OMX_PINS=$M/pins_ub2k.bin PORT=$PORT bash $D/omxpert/run.sh > $O/server_$cfg.log 2>&1 &
  else
    env STACK=mojo OMX_STREAM=1 OMX_MODEL=$M/model_full.txt OMX_PINS=$M/pins_full.bin PORT=$PORT bash $D/omxpert/run.sh > $O/server_$cfg.log 2>&1 &
  fi
  SPID=$!
  for i in $(seq 1 600); do curl -sf http://127.0.0.1:$PORT/health | grep -q ok && break; kill -0 $SPID 2>/dev/null || { echo "$cfg failed to start" >> $O/status; break; }; sleep 2; done
  LPID=$(pgrep -n -x llama-server)
  nvidia-smi --query-gpu=index,power.draw,power.limit --format=csv,noheader -lms 500 > $O/power_$cfg.csv 2>&1 &
  NPID=$!
  echo "$cfg vram: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')" >> $O/status
  cd $D && ROUNDS=1 NPRED=64 WARM=5 PROMPTS=chat pixi run python scripts/n6_bench.py http://127.0.0.1:$PORT warm $O/warm_$cfg.json > /dev/null 2>&1
  ROUNDS=2 NPRED=128 WARM=0 PROMPTS=chat pixi run python scripts/n6_bench.py http://127.0.0.1:$PORT $cfg $O/bench_$cfg.json > $O/bench_$cfg.out 2>&1
  echo "$cfg short: $(tail -1 $O/bench_$cfg.out)" >> $O/status
  pixi run python scripts/n7_longdecode.py http://127.0.0.1:$PORT $cfg $O/long_$cfg.json >> $O/status 2>&1
  [ $cfg = new ] && pixi run python scripts/n7_long.py http://127.0.0.1:$PORT $cfg $O/needle_$cfg.json >> $O/status 2>&1
  echo "$cfg peak vram: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')" >> $O/status
  kill $NPID; kill $SPID; sleep 20; kill -0 $LPID 2>/dev/null && kill -9 $LPID
done
echo done > $O/done
