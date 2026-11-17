#!/bin/bash
# omxpert-mojo M4 baseline: what does the routed MoE cost llama.cpp's pinned stack per token? Decode the held-out chat
# prompts twice: normal, and with EDX_NOMOE=1 (routed experts + router dropped from the graph; timing only).
# MoE ms/token = 1000/tps(normal) - 1000/tps(nomoe). runs/<date>-moecost/MEASURING exists while timing runs.
set -u
D=~/Projects/expert-dispatch; O=$D/runs/2026-11-16-moecost; PORT=18091
for mode in normal nomoe normal2; do
  bash $D/scripts/wait_quiet.sh
  if [ "$mode" = "nomoe" ]; then export EDX_NOMOE=1; else unset EDX_NOMOE; fi
  PIN_MIB=35000 PORT=$PORT STACK=pinned bash $D/omxpert/run.sh > $O/server_$mode.log 2>&1 &
  SPID=$!
  for i in $(seq 1 600); do curl -sf http://127.0.0.1:$PORT/health | grep -q ok && break; kill -0 $SPID 2>/dev/null || { echo "$mode failed" >> $O/status; break; }; sleep 2; done
  LPID=$(pgrep -n -x llama-server)
  touch $O/MEASURING
  cd $D && ROUNDS=1 NPRED=64 WARM=5 PROMPTS=chat pixi run python scripts/n6_bench.py http://127.0.0.1:$PORT warm $O/warm_$mode.json > /dev/null 2>&1
  ROUNDS=2 NPRED=128 WARM=0 PROMPTS=chat pixi run python scripts/n6_bench.py http://127.0.0.1:$PORT $mode $O/bench_$mode.json > $O/bench_$mode.out 2>&1
  rm -f $O/MEASURING
  kill $SPID; sleep 20; kill -0 $LPID 2>/dev/null && kill -9 $LPID
  echo "$mode done $(tail -1 $O/bench_$mode.out)" >> $O/status
done
unset EDX_NOMOE
echo done > $O/done
