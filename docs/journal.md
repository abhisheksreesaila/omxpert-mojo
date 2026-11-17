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
