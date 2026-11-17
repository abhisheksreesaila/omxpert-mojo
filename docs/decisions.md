# Decisions

- D-01 · own git repo inside expert-dispatch over a new top-level repo · the parent is not git and holds the model tools and data · yes
- D-02 · Mojo 1.2.0 + MAX 26.7.0 release over the Oct nightly in ../max · matches the Modular source checked out today · yes
- D-03 · reuse llama.cpp for attention/linear attention/n-gram, own only the expert layer · a full qwen4exp forward pass in Mojo is weeks; the measured gap is in the MoE dispatch · yes
- D-04 · write our own Q4_K/Q5_1 GPU GEMV · MAX has Q4_K only on CPU and Q4_0/GPTQ on GPU · yes
- D-05 · fp32 activations with per-element dequant over int8 activations + integer dots · keeps error ~3e-7 vs float64 (milestone bar 1e-3) and matches the reference; int8 would make order irrelevant (exact) but changes numerics; revisit if CPU or GPU becomes ALU-bound · yes
- D-06 · fold Q5_1's offset as d*sum(q h) + m*sum(h) on both devices · one fma per value instead of two; it's a definition of the order, applied identically on CPU and GPU · yes
- D-07 · gate/up: one row pair per warp with x in registers, over multi-row and split-warp variants · measured best (32.8 vs 38-47 us) · yes
- D-08 · M2 CPU speed judged on the median of 5 runs, not one · desktop load (Chromium) moves single runs by +-1.5 GB/s; the skill's noise rule; threshold unchanged · yes
- D-09 · keep two parallelize calls per layer for now; persistent pthread workers come with the mailbox in M3 · fused spin-barrier version was 2.8x slower with the runtime's task model · yes
