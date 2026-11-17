#!/bin/bash
# omxpert-mojo M8 step 1: what does the per-layer CPU stop cost? Same STACK=mojo flags, four modes back to back:
#   mojo    real Mojo expert layer (custom op on the CPU, 48 stops per token)
#   nullop  OMX_NULLOP=1: the 48 stops stay, the expert work is skipped (output meaningless)
#   nomoe   EDX_NOMOE=1: routed experts dropped from the graph, no stops (output meaningless)
#   mojo2   control repeat
# stop cost per token = nullop - nomoe (ms/token). runs/2026-11-17-split/MEASURING exists while timing runs.
set -u
D=~/Projects/expert-dispatch; O=$D/runs/2026-11-17-split; PORT=18095
for mode in ${MODES:-mojo nullop nomoe mojo2}; do
  bash $D/scripts/wait_quiet.sh
  unset OMX_NULLOP EDX_NOMOE
  [ "$mode" = "nullop" ] && export OMX_NULLOP=1
  [ "$mode" = "nomoe" ] && export EDX_NOMOE=1
  STACK=mojo PORT=$PORT bash $D/omxpert/run.sh > $O/server_$mode.log 2>&1 &
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
echo done > $O/done
