# omxpert-mojo

**A 125B mixture-of-experts model on a desktop, 1.5–1.7× faster than llama.cpp, with output that stays bit-identical.**

Qwen3.8-Flash-Next (125B MoE, 512 experts per layer, top-10, plus a 51B n-gram table, 111 GB at 4 bits) on 2× RTX 3090
(48 GB VRAM), 62 GB RAM and a Ryzen 9 5900X. A ~2,700-line Mojo expert layer replaces llama.cpp's MoE: your personal
"hot" experts live in VRAM, the cold ones run on the CPU, and the two meet through a mailbox in pinned memory.

**Visual explainer (15 frames + 2 isometric diagrams):** [omxpert Frame by Frame](https://claude.ai/artifact/8svxBeMBKyVZrCJPtcTyig)
(private link, opens when you're signed in). The same page is in [`docs/explainer/index.html`](docs/explainer/index.html).

## Results

Back to back on held-out chat prompts, 17 Nov 2026 (`runs/2026-11-17-m8final`, median of 2 rounds × 3 prompts):

| | Mojo now (2 runs) | Mojo, night before | llama.cpp pinned stack |
|---|---|---|---|
| Decode, short chat | **49.3 / 53.9 tok/s** | 41.0 | 31.9 |
| Reading short prompts (~550 tokens) | **221 / 258 tok/s** | 189 | 224 |
| Reading a 25K-token prompt | **181 / 187 tok/s** | 177 | 163 |
| Decode after a 25K-token prompt | **37.8 / 38.1 tok/s** | 29.7 | 26.9 |
| Code hidden at 26.5K tokens | found | found | found |

The llama.cpp column is from the previous night's back-to-back run (`runs/2026-11-17-final`).

## Findings

1. **Your profile decides what lives on the GPU.** Run over the owner's own chat history, half the experts serve 93% of
   the picks. A generic coding profile covered only 44%.
2. **llama.cpp's cost was coordination, not math.** It spent 24.7 of 35.7 ms per token on experts while the GPUs idled
   60–80% of the time. The Mojo layer does the same expert work in ~9 ms.
3. **Same bits on CPU and GPU.** The expert kernels are written once in Mojo with a width parameter (1 lane per GPU
   thread, 32 lanes per CPU vector), with a fixed summation order, so an expert gives identical bits wherever it runs.
   Greedy text is byte-identical when half the hot experts move to the CPU.
4. **Measure before building.** The per-layer hand-off was estimated at 5 ms. Measured: llama.cpp's CPU stops cost 1.2 ms
   per token, and Mojo's own wait after each layer about 3 ms.
5. **Never stop the GPU.** Mojo adopts llama.cpp's CUDA stream (same CUDA context), so the expert kernels run in
   llama.cpp's queue and CUDA graphs record and replay them. The GPU numbers its own CPU requests from a counter in
   pinned RAM, so a replay stays correct. Decode went from 42.6 to 48.0 tok/s.
6. **Freed VRAM, more hot experts.** The new path left ~4 GB idle per card. It now holds 1,700 more hot experts: 92% of
   picks are on the GPUs (was 88%). That gave 54.3 tok/s and the long-context gains above.
7. **Speculation doesn't pay here yet.** Qwen's draft head is right 64–79% of the time, but each extra token routes to
   new cold experts: checking 3 tokens costs 2.4× the expert work. It should start paying once ~96% of picks are hot.
8. **Falsified:** the hope that consecutive tokens reuse cold experts (milestone 6). Five tokens need 3.6× the cold
   experts of one.

## Run it

```bash
pixi run mojo build --emit shared-lib -I src src/omx_lib.mojo -o build/libomxmoe.so
pixi run python tools/build_model.py --budget-gib 17.6,15.9 --tag _full    # hot set from your profile
cd .. && STACK=mojo omxpert/run.sh                                         # OpenAI-compatible server on :8080, 40K context
```

The llama.cpp side is in [`patches/llama.cpp-omx-hook.patch`](patches/llama.cpp-omx-hook.patch): a ggml custom-op hook
in `build_moe_ffn` and a CUDA-backend hook that runs the Mojo layer on llama.cpp's stream.

## Milestones (`scripts/check`)

| # | Milestone | Status |
|---|---|---|
| 1 | GPU expert kernel ≥ 50% of VRAM bandwidth | pass (61%) |
| 2 | CPU kernel from the same source, bit-identical | pass (1200/1200; speed bar on a quiet machine) |
| 3 | Hybrid layer, routing stays on the GPU | pass (400/400 bit-identical) |
| 4 | 48 layers, real profile, ≤ 70% of llama.cpp's MoE time | pass (35%) |
| 5 | End to end, faster than llama.cpp, placement-invariant | pass |
| 6 | Batched verify ≤ 2× one token's cold time | **fail** (falsified by the trace: 3.6× at 5 tokens) |
| 7 | GLM-5.3-Flash on this box | projection only: ~14.5 tok/s, needs 120 GB of disk |
| 8 | In-stream decode, bit-identical, faster | pass |

## Where things are

- [`docs/report.md`](docs/report.md): the short report
- [`docs/journal.md`](docs/journal.md): every step and number, in order
- [`docs/decisions.md`](docs/decisions.md): decisions D-01 to D-27 with reasons
- `src/`: Mojo (`qexperts` kernels, `gpu_moe`, `cpu_pool` worker threads, `layer48` kernels, `omx_lib` the library)
- `tools/probe/`: C harnesses (external-stream probe, in-stream vs host-path bit check)
- `runs/`: raw benchmark results
