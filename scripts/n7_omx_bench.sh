#!/bin/bash
# omxpert-mojo M5: llama.cpp with the Mojo expert layer (libomxmoe.so) for decode, same client/prompts as
# n7_moe_cost.sh. usage: n7_omx_bench.sh <tag> [extra env...]
set -u
D=~/Projects/expert-dispatch; M=$D/omxpert-mojo; O=$D/runs/2026-11-16-omx; PORT=18093; TAG=$1; shift
bash $D/scripts/wait_quiet.sh
if [ "${BASELINE:-0}" = "1" ]; then
  # control: llama.cpp's own pinned stack (EDX patch, 35000 MiB of pins), same extra flags, no Mojo
  env "$@" STACK=pinned PIN_MIB=35000 STACK_ARGS="-fit off -ngl 99 --n-cpu-moe 48 --moe-cache-mib 35000 ${OMX_ARGS:--lzm off}" \
    PORT=$PORT bash $D/omxpert/run.sh > $O/server_$TAG.log 2>&1 &
else
env "$@" STACK_ARGS="-fit off -ngl 99 --n-cpu-moe 48 ${OMX_ARGS:--lzm off}" \
  STACK_ENV="OMX_LIB=$M/build/libomxmoe.so OMX_MODEL=${OMX_MODEL:-$M/results/model_m5.txt} OMX_PINS=${OMX_PINS:-$M/results/pins_m5.bin}" \
  PORT=$PORT bash $D/omxpert/run.sh > $O/server_$TAG.log 2>&1 &
fi
SPID=$!
for i in $(seq 1 600); do curl -sf http://127.0.0.1:$PORT/health | grep -q ok && break; kill -0 $SPID 2>/dev/null || { echo "$TAG failed" >> $O/status; exit 1; }; sleep 2; done
LPID=$(pgrep -n -x llama-server)
touch $O/MEASURING
echo "majflt_before $(awk '{print $12}' /proc/$LPID/stat)" > $O/faults_$TAG.txt
cd $D && ROUNDS=1 NPRED=64 WARM=5 PROMPTS=chat pixi run python scripts/n6_bench.py http://127.0.0.1:$PORT warm $O/warm_$TAG.json > /dev/null 2>&1
ROUNDS=2 NPRED=128 WARM=0 PROMPTS=chat pixi run python scripts/n6_bench.py http://127.0.0.1:$PORT $TAG $O/bench_$TAG.json > $O/bench_$TAG.out 2>&1
echo "majflt_after $(awk '{print $12}' /proc/$LPID/stat)" >> $O/faults_$TAG.txt
rm -f $O/MEASURING
kill $SPID; sleep 20; kill -0 $LPID 2>/dev/null && kill -9 $LPID
echo "$TAG done $(tail -1 $O/bench_$TAG.out)" >> $O/status
