# Journal

## 2026-11-16
- Setup. Expert tensors: gate/up Q4_K (47 layers) or Q5_K (1), down Q5_1 (43) or Q8_0 (5); expert = 2560->640->2560, ~3.1 MB; 512 experts/layer, top-10.
- Handoff microbench (../tools/mailbox/roundtrip.cu): launch+sync 6.3 us, mapped-memory mailbox 2.2 us.
- First llama.cpp profile run died: GPU0 OOM by 1 GB (desktop uses ~2.7 GB on GPU1, and a Mojo test ran at the same time). Rerun with PIN_MIB=35000.
