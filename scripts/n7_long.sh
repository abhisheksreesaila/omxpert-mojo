#!/bin/bash
# omxpert-mojo long-context runs: Mojo at 512- and 2048-token chunks, and llama.cpp's own pinned stack.
set -u
D=~/Projects/expert-dispatch; M=$D/omxpert-mojo; O=$D/runs/2026-11-17-long; PORT=18095; mkdir -p $O
MODEL=$M/results/model_ub2k.txt; PINS=$M/results/pins_ub2k.bin
for cfg in omx_ub2k omx_ub512 base; do
  bash $D/scripts/wait_quiet.sh
  case $cfg in
    omx_ub2k)  ENVX="OMX_PREFILL=1 OMX_BMAXT=2048"; ARGS="-fit off -ngl 99 --n-cpu-moe 48 -lzm on -t 6 --poll 0 -b 2048 -ub 2048"
               STACK_ENV="OMX_LIB=$M/build/libomxmoe.so OMX_MODEL=$MODEL OMX_PINS=$PINS" ;;
    omx_ub512) ENVX="OMX_PREFILL=1 OMX_BMAXT=512"; ARGS="-fit off -ngl 99 --n-cpu-moe 48 -lzm on -t 6 --poll 0"
               STACK_ENV="OMX_LIB=$M/build/libomxmoe.so OMX_MODEL=$MODEL OMX_PINS=$PINS" ;;
    base)      ENVX="STACK=pinned PIN_MIB=35000"; ARGS="-fit off -ngl 99 --n-cpu-moe 48 --moe-cache-mib 35000 -lzm on -t 6 --poll 0"
               STACK_ENV="" ;;
  esac
  if [ "$cfg" = base ]; then
    env $ENVX CTX=40960 STACK_ARGS="$ARGS" PORT=$PORT bash $D/omxpert/run.sh > $O/server_$cfg.log 2>&1 &
  else
    env $ENVX CTX=40960 STACK_ARGS="$ARGS" STACK_ENV="$STACK_ENV" PORT=$PORT bash $D/omxpert/run.sh > $O/server_$cfg.log 2>&1 &
  fi
  SPID=$!
  for i in $(seq 1 600); do curl -sf http://127.0.0.1:$PORT/health | grep -q ok && break; kill -0 $SPID 2>/dev/null || { echo "$cfg failed to start" >> $O/status; break; }; sleep 2; done
  LPID=$(pgrep -n -x llama-server)
  cd $D && pixi run python scripts/n7_long.py http://127.0.0.1:$PORT $cfg $O/long_$cfg.json >> $O/status 2>&1
  kill $SPID; sleep 20; kill -0 $LPID 2>/dev/null && kill -9 $LPID
done
echo done > $O/done
