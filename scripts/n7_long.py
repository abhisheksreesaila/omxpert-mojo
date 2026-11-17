"""Long-context check for a running server: an ~8K-token prompt (prompt speed) and a needle in a ~24K-token
document (retrieval at long context). usage: n7_long.py <base_url> <name> <out.json>"""
import json, sys, time, urllib.request
base, name, out = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3]
text = open("/home/abhisheksreesaila/Projects/expert-dispatch/corpus/docs_heldout.txt").read()

def complete(prompt, n):
    body = {"prompt": prompt, "n_predict": n, "temperature": 0, "cache_prompt": False}
    req = urllib.request.Request(base + "/completion", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time(); r = json.loads(urllib.request.urlopen(req, timeout=3600).read()); r["wall_s"] = time.time() - t0
    return r

res = {"name": name}
r = complete(text[:32000] + "\n\nSummary of the above in one sentence:", 64)
res["p8k"] = {"prompt_tokens": r["timings"]["prompt_n"], "prompt_tok_s": r["timings"]["prompt_per_second"],
              "decode_tok_s": r["timings"]["predicted_per_second"]}
code = "48217-KESTREL"
doc = text[100000:148000]
doc = doc[:24000] + f"\n\nImportant: the vault access code is {code}. Remember it.\n\n" + doc[24000:] + doc[:48000]
body = {"messages": [{"role": "user", "content": doc + "\n\nWhat is the vault access code mentioned in the document above? Reply with the code only."}],
        "max_tokens": 400, "temperature": 0, "cache_prompt": False, "chat_template_kwargs": {"enable_thinking": False}}
req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
t0 = time.time(); r = json.loads(urllib.request.urlopen(req, timeout=3600).read())
msg = r["choices"][0]["message"]; ans = (msg.get("content") or "") + " " + (msg.get("reasoning_content") or "")
tm = r.get("timings", {})
res["needle"] = {"prompt_tokens": tm.get("prompt_n"), "prompt_tok_s": tm.get("prompt_per_second"),
                 "decode_tok_s": tm.get("predicted_per_second"), "answer": ans.strip()[:200], "found": code in ans, "wall_s": time.time() - t0}
json.dump(res, open(out, "w"), indent=1); print(json.dumps(res))
