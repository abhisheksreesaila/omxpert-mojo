# omxpert-mojo: report (16-17 Nov 2026)

**Goal:** build the expert-dispatch ideas in Mojo, below the usual level, and make a big MoE fast on 2× RTX 3090 + 62 GB RAM.

## Where it landed (updated 17 Nov, midday)
Qwen3.8-Flash-Next (125B MoE + 51B n-gram table) runs through a ~2,700-line Mojo expert layer inside llama.cpp. Since
this morning the decode layers run **on llama.cpp's own CUDA stream** (no CPU stop per layer, CUDA graphs on) and the
freed VRAM holds more hot experts (92% of picks). Back to back (`runs/2026-11-17-m8final/`, desktop busy):

| | Mojo now, 2 runs | Mojo last night | llama.cpp pinned (last night) |
|---|---|---|---|
| decode, short chat | **49.3 / 53.9 tok/s** | 41.0 | 31.9 |
| prompts, ~550 tokens | **221 / 258 tok/s** | 189 | 224 |
| prompt, 25K tokens | **181 / 187 tok/s** | 177 | 163 |
| decode after a 25K prompt | **37.8 / 38.1 tok/s** | 29.7 | 26.9 |
| 26.5K-token needle | found | found | found |

Greedy text is byte-identical across the old path, the in-stream path and the bigger hot set; the in-stream layer is
bit-identical to the host path per layer, chained, and under CUDA graph replay (`tests/m8.sh`).

What it took: (1) measuring first: ggml's CPU stops cost only 1.2 ms/token, Mojo's own per-layer wait the rest;
(2) Mojo wraps llama.cpp's cudart stream (same CUDA context, probe-verified); (3) a graph-safe GPU/CPU handshake (the
GPU numbers its own requests from a shared pinned counter); (4) the freed VRAM filled with ~1,700 more hot experts.

## Checker (`scripts/check`)
- PASS 1 GPU expert kernel (61% of VRAM bandwidth), 3 hybrid layer, 5 end to end, 7 GLM projection (14.5 tok/s, disk-blocked),
  8 in-stream decode.
- 2 and 4 passed on a quiet machine; they failed once under heavy desktop load and now wait for quiet before timing.
- FAIL 6: falsified by the trace (cold experts don't repeat across consecutive tokens: 3.6× at 5 tokens).

## 30-second demo
```bash
cd ~/Projects/expert-dispatch/omxpert-mojo && pixi run mojo build --emit shared-lib -I src src/omx_lib.mojo -o build/libomxmoe.so
pixi run python tools/build_model.py --budget-gib 17.6,15.9 --tag _full
cd .. && STACK=mojo omxpert/run.sh          # OpenAI-compatible server on :8080, 40K context
```
Explainer (13 frames + 2 isometric diagrams): `docs/explainer/index.html`.

## Three decisions worth a second look
- **D-16 `-lzm on`:** n-gram (Engram) rows read on demand freed ~29 GB of RAM for cold experts. It may be why the Mojo
  advantage fades at long context (on-demand reads compete with cold experts). Worth an A/B.
- **D-20 dual cold path:** big prompt chunks upload their busiest cold experts to the GPU while the CPU takes the rest.
  Long prompts beat llama.cpp; short ones (< 1024 tokens) still trail it.
- **D-21 speculation off:** Qwen's MTP head is accepted 64-79% of the time, but verifying 3 tokens costs ~2.4× the MoE
  work here, so it only helped after long prompts (32.7 vs 30.7 tok/s).

## Incident
At 10:37 the PC powered off mid-benchmark (no error logged, off for 21 min). GPUs were then capped at 300 W and power
logged: peak 280 W, so power draw looks unlikely as the cause; no repeat in three runs since. Cap resets on reboot.

## Next, in one line
The CPU computing cold experts is now most of the remaining expert time: raise the CPU kernel from ~16 to ~25 GB/s, or
push hot coverage past ~96% (which would also make speculation pay, IDEAS #33).
