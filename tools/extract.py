"""Locate one layer's routed-expert tensors inside the GGUF (no copy: Mojo reads them in place) and build a float64
reference case for that layer: random activation x, a top-10 routing, and the exact MoE output.
  pixi run python tools/extract.py --layer 10 [--cases 200]
Writes results/layer<L>.json (file, offsets, types, strides) and results/layer<L>_case<i>.bin (x, ids, weights, ref)."""
import argparse, glob, json, os, re, struct
import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize

MODEL = os.path.expanduser("~/models/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL")
ap = argparse.ArgumentParser(); ap.add_argument("--layer", type=int, default=10); ap.add_argument("--cases", type=int, default=1)
ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
os.makedirs("results", exist_ok=True)

want = {f"blk.{a.layer}.ffn_{k}_exps.weight": k for k in ("gate", "up", "down")}
found = {}
for path in sorted(glob.glob(f"{MODEL}/*.gguf")):
    r = GGUFReader(path)
    for t in r.tensors:
        if t.name in want:
            found[want[t.name]] = (path, t)
assert len(found) == 3, found.keys()

man = {"layer": a.layer, "n_experts": 512, "top_k": 10, "d_model": 2560, "d_ff": 640}
for k, (path, t) in found.items():
    n_exp_bytes = int(t.n_bytes) // 512
    man[k] = {"file": path, "offset": int(t.data_offset), "type": t.tensor_type.name, "shape": [int(s) for s in t.shape],
              "expert_bytes": n_exp_bytes, "row_bytes": n_exp_bytes // int(t.shape[1])}
json.dump(man, open(f"results/layer{a.layer}.json", "w"), indent=1)
print(json.dumps({k: (v["type"], v["row_bytes"]) for k, v in man.items() if isinstance(v, dict)}))

def expert(k, e):
    path, t = found[k]
    eb = man[k]["expert_bytes"]
    raw = np.asarray(t.data).reshape(-1).view(np.uint8)[e * eb:(e + 1) * eb]
    rows = t.shape[1]
    return dequantize(raw.reshape(rows, -1), t.tensor_type).astype(np.float64)   # [rows, cols]

rng = np.random.default_rng(a.seed)
for c in range(a.cases):
    x = rng.standard_normal(2560).astype(np.float32)
    ids = np.sort(rng.choice(512, 10, replace=False)).astype(np.int32)   # sorted: the fixed combine order
    w = rng.random(10).astype(np.float32); w /= w.sum()
    out = np.zeros(2560)
    for e, we in zip(ids, w):
        g = expert("gate", e) @ x.astype(np.float64); u = expert("up", e) @ x.astype(np.float64)
        h = g / (1 + np.exp(-g)) * u
        out += float(we) * (expert("down", e) @ h)
    with open(f"results/layer{a.layer}_case{c}.bin", "wb") as f:
        f.write(x.tobytes()); f.write(ids.tobytes()); f.write(w.tobytes()); f.write(out.astype(np.float64).tobytes())
    if c == 0: print("case0 ids", ids.tolist(), "out[:4]", out[:4].round(5).tolist(), "norm", round(float(np.linalg.norm(out)), 4))
