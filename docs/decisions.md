# Decisions

- D-01 · own git repo inside expert-dispatch over a new top-level repo · the parent is not git and holds the model tools and data · yes
- D-02 · Mojo 1.2.0 + MAX 26.7.0 release over the Oct nightly in ../max · matches the Modular source checked out today · yes
- D-03 · reuse llama.cpp for attention/linear attention/n-gram, own only the expert layer · a full qwen4exp forward pass in Mojo is weeks; the measured gap is in the MoE dispatch · yes
- D-04 · write our own Q4_K/Q5_1 GPU GEMV · MAX has Q4_K only on CPU and Q4_0/GPTQ on GPU · yes
