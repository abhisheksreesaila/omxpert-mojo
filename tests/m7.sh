#!/bin/bash
# M7: GLM-5.3-Flash on this box: a real run, or a projection from its config + our measured tiers plus the exact blocker.
set -e; cd "$(dirname "$0")/.."
python3 tools/project_glm.py > /dev/null
python3 -c "
import json, sys; m = json.load(open('results/m7_projection.json'))
ok = m.get('projected_tok_s', 0) > 0 and (not m['blocked'] or len(m.get('blocker', '')) > 20)
print(f\"projection {m['projected_tok_s']} tok/s ({m['quant']}, {m['file_gb']} GB); blocked: {m['blocker']}\")
sys.exit(0 if ok else 1)"
