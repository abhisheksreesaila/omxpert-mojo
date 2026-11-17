"""Decode speed after a long prompt: a ~24K-token document, then 128 generated tokens (chat, thinking off).
usage: n7_longdecode.py <base_url> <name> <out.json>"""
import json, sys, time, urllib.request
base, name, out = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3]
text = open("/home/abhisheksreesaila/Projects/expert-dispatch/corpus/docs_heldout.txt").read()[200000:296000]
body = {"messages": [{"role": "user", "content": text + "\n\nWrite a detailed, multi-paragraph summary of the document above."}],
        "max_tokens": 128, "temperature": 0, "cache_prompt": False, "chat_template_kwargs": {"enable_thinking": False}}
req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
t0 = time.time(); r = json.loads(urllib.request.urlopen(req, timeout=3600).read())
tm = r.get("timings", {})
res = {"name": name, "prompt_tokens": tm.get("prompt_n"), "prompt_tok_s": tm.get("prompt_per_second"),
       "gen_tokens": tm.get("predicted_n"), "decode_tok_s": tm.get("predicted_per_second"), "wall_s": time.time() - t0,
       "draft": {k: tm.get(k) for k in ("draft_n", "draft_n_accepted")}}
json.dump(res, open(out, "w"), indent=1); print(json.dumps(res))
