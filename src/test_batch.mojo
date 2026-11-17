# M6b proof: the batched expert layer (prompt chunks, speculative verify) gives the SAME BITS as running its tokens
# one at a time, on layers with every format mix, and times batches of 3 (verify) and 512 (a prompt chunk).
# args: model_txt pins_bin
from std.sys import argv
from std.time import perf_counter_ns
from std.random import random_float64, seed
from std.memory import alloc
from omx_lib import Engine
from gpu_moe import D_MODEL

comptime NL = 48


def main() raises:
    var a = argv()
    var e = Engine(String(a[1]), String(a[2]))
    var TMAX = 512
    var x = alloc[Float32](TMAX * D_MODEL).as_unsafe_any_origin()
    var y1 = alloc[Float32](TMAX * D_MODEL).as_unsafe_any_origin()
    var y2 = alloc[Float32](TMAX * D_MODEL).as_unsafe_any_origin()
    seed(11)
    for i in range(TMAX * D_MODEL):
        x[i] = Float32(random_float64(-0.6, 0.6))
    var same = 0
    var total = 0
    for layer in [10, 2, 4, 30, 47]:
        for T in [1, 3, 17, 64]:
            for t in range(T):
                e.moe(layer, x + t * D_MODEL, y1 + t * D_MODEL)
            e.moe_batch(layer, x, y2, T)
            for t in range(T):
                var ok = True
                for i in range(D_MODEL):
                    if y1[t * D_MODEL + i].to_bits() != y2[t * D_MODEL + i].to_bits():
                        ok = False
                total += 1
                if ok:
                    same += 1
    print("bit-identical tokens:", same, "/", total)
    # timing over all 48 layers
    for T in [1, 3, 512]:
        var best_one = 1 << 62
        var best_b = 1 << 62
        for _ in range(3):
            var t0 = perf_counter_ns()
            for l in range(NL):
                for t in range(min(T, 3) if T == 512 else T):
                    e.moe(l, x + t * D_MODEL, y1 + t * D_MODEL)
            var one = perf_counter_ns() - t0
            if T == 512:
                one = one * 512 // 3
            best_one = min(best_one, one)
            t0 = perf_counter_ns()
            for l in range(NL):
                e.moe_batch(l, x, y2, T)
            best_b = min(best_b, perf_counter_ns() - t0)
        print("{\"tokens\": ", T, ", \"one_at_a_time_ms\": ", Float64(best_one) / 1e6, ", \"batched_ms\": ", Float64(best_b) / 1e6,
              ", \"batched_tok_s\": ", Float64(T) * 1e9 / Float64(best_b), "}")
    print("{\"bit_identical\": ", same, ", \"total\": ", total, "}")
