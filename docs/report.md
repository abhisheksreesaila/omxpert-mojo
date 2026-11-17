# omxpert-mojo: report (16-17 Nov 2026)

**Goal:** build the expert-dispatch ideas in Mojo, below the usual level, and make a big MoE fast on 2× RTX 3090 + 62 GB RAM.

## Where it landed
Qwen3.8-Flash-Next (125B MoE + 51B n-gram table) runs through a ~2,500-line Mojo expert layer inside llama.cpp.
Final comparison, back to back on the owner's held-out chat prompts (`runs/2026-11-17-final/`):

| | Mojo (`STACK=mojo`), 2 runs | llama.cpp pinned stack |
|---|---|---|
| decode, short chat | **42.7 / 41.9 tok/s** | 31.9 |
| prompts, ~550 tokens | 180 / 189 tok/s | 224 |
| prompt, 25K tokens | **194 / 171 tok/s** | 163 |
| decode after a 25K prompt | 29.3 / 25.8 tok/s | 26.9 |
| 26.5K-token needle | found | found |

Best single numbers during the night: 47.7 tok/s decode on one prompt, 214 tok/s on a 26.5K prompt.
Expert output is bit-identical on CPU and GPU; greedy text is byte-identical when half the hot experts move to the CPU.

## Checker (`scripts/check`)
- PASS 1 GPU expert kernel (61% of VRAM bandwidth), 3 hybrid layer, 5 end to end, 7 GLM projection (14.5 tok/s, disk-blocked).
- 2 and 4 passed on a quiet machine; they failed once under heavy desktop load and now wait for quiet before timing.
- FAIL 6: falsified by the trace (cold experts don't repeat across consecutive tokens: 3.6× at 5 tokens).

## 30-second demo
```bash
cd ~/Projects/expert-dispatch/omxpert-mojo && pixi run mojo build --emit shared-lib -I src src/omx_lib.mojo -o build/libomxmoe.so
pixi run python tools/build_model.py --budget-gib 14.8,13.8 --tag _ub2k
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

## Next, in one line
Run the Mojo kernels inside llama.cpp's own CUDA stream (no CPU split per layer, CUDA graphs back on): the ~5 ms per
token of split overhead is now the largest cost that isn't attention or expert math.
