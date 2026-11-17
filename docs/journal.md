# Journal

## 2026-11-16
- Setup. Expert tensors: gate/up Q4_K (47 layers) or Q5_K (1), down Q5_1 (43) or Q8_0 (5); expert = 2560->640->2560, ~3.1 MB; 512 experts/layer, top-10.
- Handoff microbench (../tools/mailbox/roundtrip.cu): launch+sync 6.3 us, mapped-memory mailbox 2.2 us.
- First llama.cpp profile run died: GPU0 OOM by 1 GB (desktop uses ~2.7 GB on GPU1, and a Mojo test ran at the same time). Rerun with PIN_MIB=35000.
- CPU expert layer (src/qexperts.mojo + src/cpu_moe.mojo, W=32) first run: rel. error 2.6e-7 vs float64, 10 experts in 1.41 ms
  (21.8 GB/s). Caveat: 31 MB nearly fits the 5900X's 2x32 MB L3, so this is cache-flattered; M2 must time random routings
  over the whole layer.
- Mojo 1.2.0 gotchas: `out` and `case` are keywords; positional ptr[i] deprecated (works); alloc[T](n) works with a warning;
  closures for parallelize are `def f(i: Int) {imm}:` passed as parallelize(f, n); mutable pointer params need
  `Pointer[mut=True, T, _]`; two pointers from one List trip the aliasing checker (copy into alloc'd buffers).
- parallelize wraps each task in FlushDenormals: the CPU flushes denormals, the GPU does not by default. Possible source of
  CPU/GPU bit differences for tiny values; check in M2.
- wait_quiet.sh never fires while I compile (load > 3); profile and Mojo builds must not overlap.
- Profile: pinned stack 24.5 tok/s chat; GPUs busy 23%/39%; llama.cpp main thread 98% busy -> host-bound. M3-M5 premise holds.

## M1: GPU expert kernel (PASS: 60.4 us/layer, 509 GB/s = 61% of 838, rel. err 2.96e-7, bit-identical to CPU)
Path, per-layer time for 10 experts (31 MB):
- 140 us naive (byte loads, runtime loops) -> 85 (aligned word loads + comptime unroll) -> 81 (down: 1 warp per expert)
- int->float via the exact 2^23 magic trick: no GPU change (not I2F-bound), CPU +10%.
- 72 us: Q4_K scale decode was DIVERGENT per lane (j<4 branch); lane t decodes sub-block t and the rest come by shuffle.
- Probe (src/bw_probe.mojo): plain reads of 12/18 MB take 16.5/23.5 us (~760-800 GB/s): the practical floor.
- Probe (src/pattern_probe.mojo): Q5_1 weight loads alone hit the floor (16.6 us); the per-row reloads of h (2.5 KB, L1
  hits) doubled it. Keeping h in registers and reusing it across 8 rows: down 36.6 -> 26.9 us.
- Same for x in gate/up: more rows per warp got SLOWER (fewer warps in flight); 1 row (gate+up) per warp with x in
  registers is best (32.8 us). Splitting gate and up onto separate warps: 44 us (worse). Array[] spilled; SIMD[64] did not.
Every step kept the per-lane arithmetic order, so CPU/GPU stayed bit-identical throughout.

## M2: same source, same bits (PASS: 1200/1200 tokens bit-identical on Q5_1 and Q8_0 layers; CPU median 16.2 GB/s = 62%)
- Bit-identical from the first try, and through every optimisation, because only loads were specialised per device;
  the per-lane arithmetic (fma chain order, butterfly tree, own exp) is one definition.
- CPU speed with random routing over the full layer: 13.5 -> ~16.5 GB/s median. Probe (src/cpu_bw_probe.mojo): plain
  sequential reads 26 GB/s; MoE pattern (10 random 3 MB pieces/token, one parallelize) 22.6 GB/s = the practical ceiling.
- Helped: shuffle-built scale vectors, 16-byte block loads, h arranged in lane order once per expert, expert-major down
  loop + software prefetch. Didn't: task count (12-96 all within noise of +-1.5 GB/s from desktop load).
- Fused single parallel region with spin barriers: 5.9 GB/s (worse): parallelism_level is 24 (SMT) and spinning
  tasks wait on workers not yet scheduled. Persistent workers need real threads (pthreads via FFI) -> M3.
- Hot-cache CPU (same 10 experts): 28 GB/s, so math is not the limit; memory/compute overlap is.

## M3: hybrid layer (PASS: 400/400 tokens bit-identical to all-GPU, layers 10 and 4; no host sync per layer)
- Mojo mailbox probe (src/mailbox_probe.mojo): pinned host buffers are directly GPU-addressable; volatile store +
  threadfence[SYSTEM] then spin: 2.06 us round trip (same as the CUDA C probe).
- Routing (llama.cpp qwen4exp): softmax over 512, top-10, renormalise -> w = exp(l - lmax) / sum over the top-10.
- First run hung: when a token has no cold experts the GPU doesn't wait, races ahead and overwrote the single mailbox
  before the CPU read it. Fix: a ring of 16 mailbox slots (> the 8-token enqueue window), slot = seq % 16.
- Latency per token-layer: all-GPU 85-101 us (router 5 MB F32 read + top-k + experts + combine). Hybrid with ~1 cold
  expert per token: 470-580 us. The CPU request path costs ~0.4 ms for one 3 MB expert (memory alone ~0.19 ms):
  2 parallelize dispatches, serial permute/arrange, copies. Next: persistent CPU workers (pthreads) + split one cold
  expert's rows across all cores without per-request dispatch.

## M4 (in progress): 48 layers, real routing, owner's profile
- tools/build_model.py: pins from profile.bin under 35000 MiB split by free VRAM (21.0/18.3 GB), layers 0-23 GPU0,
  24-47 GPU1 -> 11,704 experts (48%); held-out chat picks on the GPU 92.8% (night 6 llama.cpp: ~94% at 36500 MiB).
  Real routing: 400 tokens sampled evenly from traces/.../chat_heldout.npz (93.3% hot). The first tokens of the
  trace are atypically cold (54%), so don't take the head of the file.
- Bugs on the way (all deadlocks of "GPU waits on CPU, CPU blocked on GPU"):
  1. first launch of a kernel variant loads its module, which waits for the device to idle, while a combine kernel
     spins on the CPU -> pre-launch every kernel variant with harmless args before the token loop;
  2. a ctx.synchronize() at the warm-up/measure boundary while queued kernels wait on the CPU -> timestamp only.
- 72.9 ms/token MoE -> 47.2 (warm) -> profile: empty parallelize(48) costs 236 us mean (60 best), 2 per cold request.
- Persistent CPU co-processor (src/cpu_pool.mojo): 12 pthreads created from Mojo via the C FFI (abi("C") entry),
  spinning; worker 0 polls the GPU mailbox rings itself; atomic barriers. Dispatch+sync 1.04 us (pthread_probe).
  -> 35.7 ms/token.
- Cold experts were being read from the SSD (50k major faults/s, 650 MB/s): loading 35 GB of hot experts through the
  page cache evicted the cold ones. madvise(MADV_PAGEOUT) on each hot expert after its VRAM copy (= llama.cpp's
  EDX_PAGEOUT) -> no major faults in the loop -> 11.7 ms/token MoE (CPU 8.5 ms for ~20 cold experts/token).
- llama.cpp baseline (scripts/n7_moe_cost.sh, control/candidate/control): normal 27.2 / 28.8 tok/s (35.7 ms/token),
  EDX_NOMOE 91.2 tok/s (11.0 ms) -> llama.cpp's routed MoE = 24.7 ms/token; everything else = 11.0 ms.
- M4 PASS: Mojo MoE 8.6 ms/token (35% of 24.7) on a quiet machine (11.7 with desktop load), 19.9 cold experts/token
  served by the CPU pool in 5.4 ms. Projection if M5 adds nothing: 11.0 + 8.6 = 19.6 ms -> ~51 tok/s (today 28).

## M5: end to end inside llama.cpp (PASS: 42.8 tok/s decode vs 25.9 same-flag control / 27-29 llama.cpp best; text identical across pin sets)
- libomxmoe.so (src/omx_lib.mojo: @export C entry points, integer handle, initialize_runtime) + ~60-line hook in
  llama.cpp build_moe_ffn (patches/llama.cpp-omx-hook.patch): dlopen OMX_LIB, a ggml custom op per layer calls
  omx_moe(layer, x, y). llama.cpp keeps attention, linear attention, shared expert, n-gram table, sampling, server.
- Pins for M5 had to fit beside llama.cpp (+ a 2 GB node process and the desktop): 16.3 + 15.3 GiB, 10,825 experts,
  90.8% of held-out picks (llama.cpp baseline: 35000 MiB). Mojo runs with the SMALLER budget.
- First run: 19 tok/s decode, 8 tok/s prompts. The Mojo MoE took 27-50 ms/token in-process (8.6 standalone):
  -lzm off kept the 28.8 GB n-gram table resident, llama.cpp's prompt path read all experts from mmap (evicting the
  cold ones), and llama.cpp's CPU threads spun against our 12 workers.
- Fix (and start of C): -lzm on (n-gram rows read on demand, llama.cpp's prefetch), OMX_PREFILL=1 (prompts through
  Mojo too, token by token, so llama.cpp never touches expert weights), -t 6 --poll 0.
  -> 42.8 tok/s decode (35.3-47.7), 112 tok/s prompts; in-process MoE 6.7-8.8 ms/token (~150 us/layer call).
- Control, llama.cpp pinned stack with the same flags: 25.9 tok/s decode, 227 tok/s prompts (flags don't help it).
  Earlier same-day llama.cpp default flags: 27.2 / 28.8. => +53% to +65% decode; prompts 2x slower (token-by-token MoE).
- Placement invariance: greedy 96 tokens x 3 prompts, pins A, A again, and B (22% hot, 68% of picks: thousands of
  experts moved to the CPU): byte-identical text.
- Ops lesson: T3 restarts kill setsid/nohup children; run long jobs with systemd-run --user (pass PATH/HOME).

## M6: premise falsified (FAIL, by measurement, before building)
- M6 assumed consecutive tokens share cold experts, so a k=4 verify batch would read them once (<= 2x one token).
  On the held-out chat trace (26,915 consecutive tokens, M5 pins): distinct cold experts per batch of 1/2/3/5/8 tokens
  = 44 / 77 / 106 / 160 / 233 -> 1.00 / 1.74 / 2.40 / 3.62 / 5.28x one token. The profile concentrates the HOT set;
  cold picks are a diverse long tail. The CPU is bandwidth-bound, so verify cold time ~3.6x at k=4: the bar can't be met.
- What still holds: speculation amortises llama.cpp's ~16 ms/token of non-MoE work and the GPU hot path over k+1
  tokens; only the CPU cold part doesn't amortise. Rough model for k=4 at ~76% acceptance (~3.5 tokens/step):
  ~45 ms per step -> ~13 ms/token -> ~75 tok/s. And batching does fix prompts: at 512 tokens per batch every expert
  is read once and the CPU becomes compute-bound (~1 ms/token of cold work).

## Overnight (autopilot): speculation + batched layer
- MTP speculation (Qwen's draft head, 2 drafts) with the one-token Mojo path: 33.9 tok/s (< 42.8). Acceptance 64-76%,
  2.3-2.5 tokens/step, but verify = 3 separate token passes, and the draft head's ~3 GB of VRAM cut pins to 87.8%.
  Hook now skips layer >= 48 (the MTP head's own experts stay on llama.cpp).
- Batched expert layer (src/qmulti.mojo, layer48 batch kernels, cpu_pool batch job, omx_moe_batch): GPU kernels take a
  token index; the CPU groups a batch's cold picks by expert (counting sort) and reads each cold expert ONCE for all
  its tokens (dequantise once, per-token fma chains unchanged). test_batch: 425/425 tokens bit-identical to the
  one-token path across Q4_K/Q5_1, Q5_K/Q8_0, Q4_K/Q8_0 layers.
- In llama.cpp (OMX_PREFILL + batch): prompts 112 -> 190 tok/s (llama.cpp 227), decode 43.4. Prompt chunks are now
  bound by the CPU reading the chunk's cold experts; the GPU re-reads hot weights per token (~0.85 s per 512 tokens).
- MTP + batched verify: 40.6 tok/s median (36.7-53.9; 52-54 where the draft is accepted 76%). Not yet above plain
  decode (43.4): verify still ~2.4x the MoE work (hot experts overlap only 0.76 across 3 tokens, cold 0.8), and the
  draft head's VRAM costs pins. Next: -otd to keep the draft head's experts on the CPU.
- Long prompts (docs text, 8K tokens): Mojo 52-61 tok/s vs llama.cpp 136. Docs text routes away from the chat profile
  (many more cold picks) and my batch path computed every cold pick on the CPU. llama.cpp uploads big-batch expert
  weights to the GPU instead. Bigger chunks (2048) didn't help a CPU-bound path.
- Dual cold path for batches >= 64 tokens (moe_chunk): routing on the GPU, one sync, the host splits the cold picks by
  tokens per expert: the busiest experts (>= 3 tokens, up to OMX_NSTAGE = 96) are copied to pinned staging (parallel)
  and DMA'd to a VRAM staging pool, then run by the same batch kernels (slot >= STAGE_BASE); the rest go to the CPU
  pool at the same time. Still 425/425 bit-identical; uniformly routed 512-token chunk 9.1 -> 4.7 s.
- Needle test fixed: raw /completion on a 26K-token doc ended at once (EOS) on every stack incl. stock; now uses
  /v1/chat/completions with thinking off.
- Long context, fixed needle (26.5K tokens, chat API): code found by all three. Prompt speed: Mojo 2048-chunks + dual
  path 214 tok/s, Mojo 512 116, llama.cpp 164 -> long prompts 1.3x faster than llama.cpp.
- Attention vs context (llama-bench, experts dropped): 11.5 ms/token at 24K depth, same as 8K. The QSA sparse
  attention in llama.cpp masks over all cells (no sparsity saving) but reading the KV is cheap at this size.
  The "15 tok/s at 26K" in the needle test was an 8-token sample after a 26K prefill, not real.
- Daily-driver candidates (2048 chunks, dual path), short chat / 25K-token prompt:
  plain: 44.6 tok/s decode, 159 short prompts, 212 tok/s 25K prompt, 30.7 tok/s decode after it.
  + MTP with the draft head's experts on the CPU (-otd exps=CPU): 35.7 short, 32.7 after 25K (79% accepted).
  -> speculation stays off by default (D-21).
- Short prompts were slower at 2048 chunks (159 vs 190): the upload path's sync + copies don't pay for a short chunk.
  Upload now only for chunks >= OMX_UPLOAD_MIN (default 1024).
- omxpert/run.sh: STACK=mojo starts the whole setup (Mojo layer, 2048 chunks, dual path, -lzm on, 40K context).
- Final back-to-back (runs/2026-11-17-final, STACK=mojo / llama.cpp pinned / STACK=mojo): short decode 42.7 / 31.9 /
  41.9 tok/s; short prompts 180 / 224 / 189; 25K prompt 194 / 163 / 171; decode after 25K 29.3 / 26.9 / 25.8.
  The long-context decode advantage fades; suspect -lzm on (on-demand n-gram rows vs cold experts in RAM).
- M7: GLM-5.3-Flash projection from its config (45 layers, 288 experts top-8 + 1 shared, expert FFN 2048) and our
  measured tiers: 14.5 tok/s at UD-IQ3_XXS (120 GB), RAM- and SSD-bound on cold experts; blocked on disk (74 GB free).
