#!/bin/bash
# M8: one server run in a mode, greedy text check (+ optional bench). usage: n8_stream_run.sh <tag> [env...]
set -u
D=~/Projects/expert-dispatch; O=$D/runs/2026-11-17-stream; PORT=18096; TAG=$1; shift
env "$@" STACK=mojo PORT=$PORT bash $D/omxpert/run.sh > $O/server_$TAG.log 2>&1 &
SPID=$!
for i in $(seq 1 600); do curl -sf http://127.0.0.1:$PORT/health | grep -q ok && break; kill -0 $SPID 2>/dev/null || { echo "$TAG failed to start" >> $O/status; exit 1; }; sleep 2; done
LPID=$(pgrep -n -x llama-server)
cd $D && timeout 900 pixi run python scripts/n8_stream_check.py http://127.0.0.1:$PORT $O/text_$TAG.json > $O/check_$TAG.out 2>&1 || echo "$TAG check failed" >> $O/status
if [ "${BENCH:-0}" = "1" ]; then
  touch $O/MEASURING
  ROUNDS=1 NPRED=64 WARM=5 PROMPTS=chat pixi run python scripts/n6_bench.py http://127.0.0.1:$PORT warm $O/warm_$TAG.json > /dev/null 2>&1
  ROUNDS=2 NPRED=128 WARM=0 PROMPTS=chat pixi run python scripts/n6_bench.py http://127.0.0.1:$PORT $TAG $O/bench_$TAG.json > $O/bench_$TAG.out 2>&1
  rm -f $O/MEASURING
fi
kill $SPID; sleep 15; kill -0 $LPID 2>/dev/null && kill -9 $LPID
echo "$TAG done $(tail -1 $O/bench_$TAG.out 2>/dev/null)" >> $O/status
