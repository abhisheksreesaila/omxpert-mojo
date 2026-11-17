"""M8 check: greedy text from a running server for fixed prompts (raw completions, temperature 0), saved to json.
usage: n8_stream_check.py <base_url> <out.json>"""
import json, sys, urllib.request
from pathlib import Path
base, out = sys.argv[1].rstrip("/"), Path(sys.argv[2])
W = Path.home() / "expert-dispatch"
code = (W / "corpus/coding_heldout.txt").read_text()
prompts = ["Write a Python function that returns the n-th Fibonacci number iteratively.\n\n```python\n",
           "The three main causes of the French Revolution were",
           code[5000:6500]]
res = []
for p in prompts:
    body = {"prompt": p, "n_predict": 96, "temperature": 0, "cache_prompt": False}
    r = json.loads(urllib.request.urlopen(urllib.request.Request(base + "/completion", json.dumps(body).encode(),
                                                                 {"Content-Type": "application/json"}), timeout=1200).read())
    res.append({"text": r["content"], "decode_tok_s": r["timings"]["predicted_per_second"]})
    print(round(r["timings"]["predicted_per_second"], 1), repr(r["content"][:80]))
out.write_text(json.dumps(res, indent=1))
