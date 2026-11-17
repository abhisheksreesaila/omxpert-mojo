"""M4 inputs: every layer's expert/router tensors (in place in the GGUF), the pinned (hot) set from the owner's profile
under llama.cpp's VRAM budget, and real routing from the held-out chat trace.
  pixi run python tools/build_model.py [--pin-mib 35000] [--tokens 400]
Writes results/model.txt (one line per layer, for Mojo), results/pins.bin (int32 [48][512]: slot or -1),
results/trace_chat.bin (int32 [T][48][10], each row sorted) and prints the pinned share of held-out picks.

Pin rule, mirroring the EDX patch: the budget is split between the GPUs by free memory (llama.cpp: 'by default by free
memory'); layers 0-23 live on GPU 0 and 24-47 on GPU 1; within a GPU, experts are taken in order of profile score until
the budget is used."""
import argparse, glob, json, os, re
import numpy as np
from gguf import GGUFReader

MODEL = os.path.expanduser("~/models/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL")
PROFILE = os.path.expanduser("~/Projects/expert-dispatch/omxpert/pins/profile.bin")
TRACE = os.path.expanduser("~/Projects/expert-dispatch/traces/Qwen3.8-Flash-Next/chat_heldout.npz")
ap = argparse.ArgumentParser()
ap.add_argument("--pin-mib", type=int, default=35000)
ap.add_argument("--free-gb", default="21.0,18.3", help="free VRAM per GPU after dense weights (llama.cpp splits by this)")
ap.add_argument("--tokens", type=int, default=400)
a = ap.parse_args()
os.makedirs("results", exist_ok=True)

L, E = 48, 512
info = {}
for path in sorted(glob.glob(f"{MODEL}/*.gguf")):
    for t in GGUFReader(path).tensors:
        m = re.match(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$", t.name)
        r = re.match(r"blk\.(\d+)\.ffn_gate_inp\.weight$", t.name)
        if m:
            info.setdefault(int(m.group(1)), {})[m.group(2)] = (path, int(t.data_offset), t.tensor_type.name, int(t.n_bytes) // E)
        elif r:
            info.setdefault(int(r.group(1)), {})["router"] = (path, int(t.data_offset), t.tensor_type.name, 0)
assert len(info) == L and all(len(v) == 4 for v in info.values())
files = sorted({v[k][0] for v in info.values() for k in v})

prof = np.fromfile(PROFILE, dtype=np.float32, offset=8).reshape(L, E)
free = [float(x) for x in a.free_gb.split(",")]
budget = [a.pin_mib * 2**20 * f / sum(free) for f in free]
pins = np.full((L, E), -1, dtype=np.int32)
nhot = np.zeros(L, dtype=np.int32)
for g, layers in enumerate([range(0, 24), range(24, 48)]):
    cand = sorted(((prof[l, e], l, e) for l in layers for e in range(E)), key=lambda c: -c[0])
    used = 0
    for score, l, e in cand:
        size = info[l]["gate"][3] + info[l]["up"][3] + info[l]["down"][3]
        if used + size > budget[g]:
            continue
        used += size
        pins[l, e] = nhot[l]
        nhot[l] += 1
pins.tofile("results/pins.bin")

tr = np.load(TRACE)["topk"].astype(np.int32)
hot_share = float((pins[np.arange(L)[None, :, None], tr] >= 0).mean())
idx = np.linspace(0, tr.shape[0] - 1, a.tokens).astype(int)   # evenly across the held-out chats, not just the start
sel = np.sort(tr[idx], axis=2).astype(np.int32)
sample_share = float((pins[np.arange(L)[None, :, None], tr[idx]] >= 0).mean())
sel.tofile("results/trace_chat.bin")

with open("results/model.txt", "w") as f:
    f.write(" ".join(files) + "\n")
    for l in range(L):
        row = [l, 0 if l < 24 else 1, int(nhot[l])]
        for k in ("gate", "up", "down", "router"):
            p, off, ty, eb = info[l][k]
            row += [files.index(p), off, ty, eb]
        f.write(" ".join(str(x) for x in row) + "\n")
json.dump({"pin_mib": a.pin_mib, "budget_gib": [b / 2**30 for b in budget], "hot_per_layer": nhot.tolist(),
           "pinned_experts": int(nhot.sum()), "heldout_hot_share": hot_share, "sample_hot_share": sample_share, "trace_tokens": int(sel.shape[0])},
          open("results/pins.json", "w"), indent=1)
print(f"pinned {int(nhot.sum())} experts ({100*nhot.sum()/(L*E):.0f}%), held-out chat picks on the GPU: {100*hot_share:.1f}% "
      f"(the {sel.shape[0]} sampled tokens: {100*sample_share:.1f}%)")
print("hot per layer:", nhot.tolist())
