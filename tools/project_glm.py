"""M7: GLM-5.3-Flash on this box, projected from GLM's real config and our measured tiers (not run: needs disk).
  python3 tools/project_glm.py   -> results/m7_projection.json
Every assumption is a named number below; change one and rerun."""
import json
# GLM-5.3-Flash (zai-org config.json, 2026-11-17): 45 layers, first 3 dense, 288 routed experts top-8 + 1 shared,
# hidden 4096, expert FFN 2048, 18B active of 321B. Quant: unsloth UD-IQ3_XXS = 120 GB (2.99 bits/param on average).
moe_layers, experts, topk, d, f = 42, 288, 8, 4096, 2048
bits = 120e9 * 8 / 321e9
expert_params = 3 * d * f
expert_bytes = expert_params * bits / 8
routed_bytes_total = moe_layers * experts * expert_bytes
routed_per_token = moe_layers * topk * expert_bytes
nonrouted_per_token = (18e9 - moe_layers * topk * expert_params) * 0.85 * 4.5 / 8   # attention/shared/LM head, kept near 4.5 bpw
# Measured on this box (omxpert-mojo M1-M5): VRAM 838 GB/s, CPU cold path ~20 GB/s from RAM, NVMe ~3 GB/s random 9 MB reads.
vram_for_experts = 48e9 - 3.2e9 - nonrouted_per_token - 3.5e9        # desktop, dense weights, KV + buffers
resident = vram_for_experts / routed_bytes_total
# ASSUMPTION: GLM routing is as skewed for the owner as Qwen's is (held-out chat: 30% -> 79%, 40% -> 88% of picks).
hit = 0.79 + (resident - 0.30) * (0.88 - 0.79) / 0.10
cold_per_token = (1 - hit) * routed_per_token
ram_for_cold = 62e9 - 12e9                                            # OS, desktop, llama.cpp, buffers
cold_set = routed_bytes_total - vram_for_experts
ram_share = min(1.0, 0.6 + 0.4 * ram_for_cold / cold_set)            # warmer cold experts stay cached
gpu_ms = (nonrouted_per_token + hit * routed_per_token) / 838e9 * 1e3 * 2.0      # x2: measured kernel efficiency ~50%
cpu_ms = cold_per_token * ram_share / 20e9 * 1e3
ssd_ms = cold_per_token * (1 - ram_share) / 3e9 * 1e3
overhead_ms = 15.0                                                    # llama.cpp non-MoE glue measured on Qwen (attention excl.)
token_ms = max(gpu_ms, cpu_ms + ssd_ms) + overhead_ms
out = {"quant": "UD-IQ3_XXS", "file_gb": 120, "disk_free_gb": 74, "blocked": True,
       "blocker": "120 GB file vs 74 GB free disk; running it needs a new drive or the owner's yes to delete the 111 GB Qwen3.8-Flash-Next",
       "expert_mb": round(expert_bytes / 1e6, 2), "routed_gb_total": round(routed_bytes_total / 1e9, 1),
       "experts_resident_share": round(resident, 3), "assumed_hit_share": round(hit, 3),
       "cold_gb_per_token": round(cold_per_token / 1e9, 3), "cold_from_ram_share": round(ram_share, 2),
       "ms": {"gpu": round(gpu_ms, 1), "cpu_ram": round(cpu_ms, 1), "ssd": round(ssd_ms, 1), "glue": overhead_ms},
       "projected_tok_s": round(1000 / token_ms, 1),
       "reference": "0xSero glm-flash-lite: 19.5 tok/s on one 3090 with an 8-channel EPYC (~150 GB/s RAM) and 4 NVMe RAID0"}
json.dump(out, open("results/m7_projection.json", "w"), indent=1)
print(json.dumps(out, indent=1))
