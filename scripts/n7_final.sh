#!/bin/bash
# Overnight final: STACK=mojo (the launcher option) vs llama.cpp's pinned stack, short chat + 25K-context decode.
set -u
D=~/Projects/expert-dispatch; O=$D/runs/2026-11-17-final; PORT=18097; mkdir -p $O
for cfg in mojo base mojo2; do
  bash $D/scripts/wait_quiet.sh
  if [ $cfg = base ]; then
    env STACK=pinned PIN_MIB=35000 CTX=40960 PORT=$PORT bash $D/omxpert/run.sh > $O/server_$cfg.log 2>&1 &
  else
    env STACK=mojo PORT=$PORT bash $D/omxpert/run.sh > $O/server_$cfg.log 2>&1 &
  fi
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
