# Where does one cold expert's CPU time go? moe_cpu with k = 1 on random experts of layer 10 (whole layer in RAM).
from std.time import perf_counter_ns
from std.random import random_ui64, seed
from std.memory import alloc
from max.algorithm import parallelize
from cpu_moe import moe_cpu, HV_PER_EXPERT, D_MODEL, D_FF


def main() raises:
    var eb_gu = 921600
    var eb_d = 1228800
    var n = 512
    var g = alloc[UInt8](n * eb_gu)
    var u = alloc[UInt8](n * eb_gu)
    var d = alloc[UInt8](n * eb_d)
    for i in range(0, n * eb_gu, 64):
        g[i] = UInt8(i % 13)
        u[i] = UInt8(i % 7)
    for i in range(0, n * eb_d, 64):
        d[i] = UInt8(i % 5)
    var x = alloc[Float32](D_MODEL)
    for i in range(D_MODEL):
        x[i] = 0.01
    var xp = alloc[Float32](D_MODEL)
    var h = alloc[Float32](10 * D_FF)
    var hv = alloc[Float32](10 * HV_PER_EXPERT)
    var s = alloc[Float32](10 * D_MODEL)
    var y = alloc[Float32](D_MODEL)
    var w = alloc[Float32](10)
    w[0] = 1.0
    var og = alloc[Int](10)
    var od = alloc[Int](10)
    seed(1)
    for k in [1, 2, 4, 10]:
        for stages in [1, 2, 3]:
            var best = 1 << 62
            var tot = 0
            for it in range(200):
                for j in range(k):
                    var e = Int(random_ui64(0, 511))
                    og[j] = e * eb_gu
                    od[j] = e * eb_d
                var t0 = perf_counter_ns()
                moe_cpu(g, u, d, False, og, od, k, w, x, xp, h, hv, s, y, stages)
                var dt = perf_counter_ns() - t0
                best = min(best, dt)
                tot += dt
            print("k", k, "stages", stages, ": mean", Float64(tot) / 200000.0, "us, best", Float64(best) / 1000.0, "us")
    # an empty parallel region
    var t0 = perf_counter_ns()
    for _ in range(200):
        def nothing(i: Int) {imm}:
            pass
        parallelize(nothing, 48)
    print("empty parallelize(48):", Float64(perf_counter_ns() - t0) / 200000.0, "us")
