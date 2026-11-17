# GOAL: omxpert's expert layer in Mojo

**End goal (owner's words, condensed):** build the expert-dispatch ideas in Mojo, going beneath the usual (fusion, compiler
tricks, below the compiler), and get speed improvements; then run a gigantic MoE like GLM-5.3-Flash on this box
(2× RTX 3090 = 48 GB VRAM, 62 GB RAM). Code the owner can inspect, benchmark and tweet about.

## Milestones (each proven by `scripts/check`)
1. **GPU expert kernel.** One Mojo kernel runs a token's routed experts for one real layer of Qwen3.8-Flash-Next
   (gate+up Q4_K, SiLU×up, down Q5_1/Q8_0, weighted sum), read straight from the GGUF. Matches a float64 NumPy
   reference (max rel. error ≤ 1e-3) and reads its bytes at ≥ 50% of measured VRAM bandwidth (838 GB/s).
2. **Same source on the CPU, same bits.** The expert math compiles for the CPU (SIMD + all cores) from the same Mojo
   source and gives **bit-identical** output to the GPU on 200 random routings; ≥ 60% of measured RAM bandwidth (26 GB/s).
3. **Hybrid layer, routing stays on the device.** Router + top-k on the GPU; hot experts on the GPU, cold on the CPU,
   handed over through a pinned-memory mailbox, no host sync per layer. Output bit-identical to the all-GPU layer.
   Latency per layer recorded next to llama.cpp's per-layer cost.
4. **All 48 MoE layers, one token, the owner's real profile** (`omxpert/pins/profile.bin`), with the profile-specialised
   layout. MoE time per token ≤ 70% of the llama.cpp pinned stack's MoE time per token (same profile, same VRAM budget).
5. **End to end.** Qwen3.8-Flash-Next decodes through the Mojo expert layer: decode tok/s on the held-out chat prompts
   above the pinned llama.cpp stack measured the same way (22.2 tok/s), and the same greedy tokens whether an expert runs on
   CPU or GPU (placement invariance).
6. **Batched CPU verify.** Verifying k = 4 drafted tokens costs ≤ 2× one token's cold-expert time (cold experts read once
   per batch), the foundation for exact self-speculation.
7. **GLM-5.3-Flash on this box.** A measured per-layer cost model (from 3-4) projects GLM-5.3-Flash speed here; running it
   for real needs ~120-200 GB of disk (74 GB free) and is gated on the owner's yes to free space.
8. **In-stream decode** (added 17 Nov, owner: "proceed"). The decode layer runs on llama.cpp's own CUDA stream: no CPU
   stop per layer, CUDA graphs stay on. Output bit-identical to the host path (per layer, chained, under graph replay);
   back-to-back decode faster than the previous default; greedy text byte-identical.

## Done means
`scripts/check` prints PASS for 1-6 and 7 has either a real run or a projection plus the exact disk blocker.

## Out of bounds
No deploying, no paid services or sign-ups, no secrets, no pushing anywhere, nothing deleted outside this folder
(the Qwen model and other data stay unless the owner says yes). No editing milestones or the checker to make them pass.

## Assumptions (made instead of asking)
- Work lives in `~/Projects/expert-dispatch/omxpert-mojo` (own git repo), reading the model from `~/models` and tools
  from the parent folder.
- Mojo 1.2.0 + MAX 26.7.0 (conda.modular.com), Modular source at `../engines/modular` for reference.
- Qwen3.8-Flash-Next is the test model (only MoE on disk); GLM is the stretch goal.
- "Same output" means bit-identical across CPU/GPU placement inside our engine, not bit-identical to stock llama.cpp
  (different kernels already diverge there, night 6).
- Integration with the rest of the model (attention, linear attention, n-gram table) reuses llama.cpp; only the expert
  layer is ours.

## Budget
120 turns (owner gave this week's token budget), checked in at each milestone.
