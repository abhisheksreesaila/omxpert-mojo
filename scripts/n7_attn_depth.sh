#!/bin/bash
# Attention-side decode cost vs context depth: llama-bench with the routed experts dropped (EDX_NOMOE), so what is
# left is attention, linear attention, the indexer, shared expert, n-gram rows and the LM head.
set -u
D=~/Projects/expert-dispatch; O=$D/runs/2026-11-17-attn; mkdir -p $O
bash $D/scripts/wait_quiet.sh
GG=$(ls ~/models/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL/*00001-of-*.gguf)
EDX_NOMOE=1 $D/llama.cpp/build/bin/llama-bench -m $GG -ngl 99 -ncmoe 48 -lzm on -t 6 -p 0 -n 32 -d 0,8192,24576 -r 2 -o json > $O/nomoe_depth.json 2> $O/nomoe_depth.err
echo done > $O/done
