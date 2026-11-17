# omxpert-mojo

**Can a 125B mixture-of-experts model run well on a desktop if it learns which experts *you* use?**

Big MoE models only run a few of their experts for each token, and the ones a single person needs turn out to be a
fairly small, stable set. So we're experimenting with this: keep *your* experts on the graphics cards, run the rest on
the CPU, and borrow every good trick we can find along the way. This repo is the Mojo part of that experiment. It's
early: one machine, one person's history. What do you think?

**Visual walk-through, frame by frame:** <https://abhisheksreesaila.github.io/omxpert-mojo/> (also in
[`docs/explainer/`](docs/explainer/index.html)).

## Setup

- **Model:** Qwen3.8-Flash-Next: 125B parameters, 512 experts per layer (10 run per token), 48 layers, plus a 51B
  n-gram table. 111 GB at 4 bits.
- **Machine:** 2× RTX 3090 (48 GB VRAM), 62 GB RAM, Ryzen 9 5900X, NVMe.
- **Profile:** the owner's own chat history, used to count which experts fire. Half the experts serve 93% of the picks;
  a generic coding profile covered only 44%. Speeds below are on held-out chats the profile never saw.
- **Engines:** llama.cpp keeps attention, the n-gram table, sampling and the server. A ~2,700-line Mojo library takes
  over the experts: router and hot experts on the GPUs, cold experts on 12 CPU threads, the two meeting in a mailbox in
  pinned memory.

## What has worked so far

Decode speed on short chats, each step measured on the same box:

| Step | Borrowed from | Decode |
|---|---|---|
| llama.cpp with your hot experts pinned in VRAM, cold ones in RAM | llama.cpp; exclusive tiers from 0xSero's glm-flash-lite; CPU/GPU overlap from FreeToken | 31.9 tok/s |
| Mojo expert layer: one source for CPU and GPU kernels (identical bits), GPU↔CPU mailbox, always-awake CPU threads | Modular's Mojo/MAX | 42.6 |
| Batched prompts; big chunks upload their busiest cold experts to the GPU while the CPU does the rest | llama.cpp's own upload path | (prompts 112 → 190 tok/s) |
| Mojo kernels run inside llama.cpp's own GPU queue, recorded in CUDA graphs | standard CUDA practice | 48.0 |
| The VRAM that freed up holds 1,700 more hot experts (92% of picks, was 88%) | | 54.3 |

Back to back against the previous setup, on a busy desktop (`runs/2026-11-17-m8final`):

| | Now (2 runs) | Previous Mojo setup | llama.cpp pinned |
|---|---|---|---|
| Decode, short chat | 49.3 / 53.9 tok/s | 41.0 | 31.9 |
| Reading short prompts | 221 / 258 tok/s | 189 | 224 |
| Decode after a 25K-token prompt | 37.8 / 38.1 tok/s | 29.7 | 26.9 |
| Code hidden at 26.5K tokens | found | found | found |

The llama.cpp column is from the previous night's run (`runs/2026-11-17-final`). Promising, not settled: these are one
machine and one person's chats.

One thing we care about: the output doesn't change. An expert gives the same bits on the CPU or the GPU, and greedy
text is byte-identical whichever experts sit where, and with or without CUDA graphs.

## What didn't work

- **Reusing cold experts across tokens.** We hoped consecutive tokens would share cold experts, so checking several
  drafted tokens at once would be cheap. The trace says no: 5 tokens need 3.6× the cold experts of one.
- **Speculative decoding (so far).** Qwen's draft head is right 64–79% of the time, but each extra token routes to new
  cold experts, so checking 3 tokens costs 2.4× the expert work. It only helped after long prompts. Our guess is that it
  starts to pay once ~96% of picks are hot.
- **Our own estimate.** We expected the per-layer GPU↔CPU hand-off to cost ~5 ms per token. Measured, llama.cpp's part
  was 1.2 ms and Mojo's own waiting about 3 ms. The fix was the same, but measuring first changed what we built.
- **GLM-5.3-Flash.** It projects to about 14.5 tok/s here but needs 120 GB of disk we don't have free yet.

## It carries on

Next questions: can the CPU's cold-expert kernel get closer to the RAM's 26 GB/s (it runs at ~16)? Does hot coverage
past 96% make speculation pay? Do the same ideas carry over to Engram-style models like Tencent's Hy4 flash?

Ideas, questions and results from other machines are very welcome: [open an issue](https://github.com/abhisheksreesaila/omxpert-mojo/issues).

**Try it:**

```bash
pixi run mojo build --emit shared-lib -I src src/omx_lib.mojo -o build/libomxmoe.so
pixi run python tools/build_model.py --budget-gib 17.6,15.9 --tag _full    # hot set from your profile
cd .. && STACK=mojo omxpert/run.sh                                         # OpenAI-compatible server on :8080, 40K context
```

The llama.cpp changes are in [`patches/llama.cpp-omx-hook.patch`](patches/llama.cpp-omx-hook.patch). The full story,
step by step with every number, is in [`docs/journal.md`](docs/journal.md); the reasons behind each choice are in
[`docs/decisions.md`](docs/decisions.md); the milestone checks are in `scripts/check` and [`GOAL.md`](GOAL.md).
