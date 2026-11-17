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
