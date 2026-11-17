# What read bandwidth can Mojo's parallelize reach from RAM here? Sum a 1.5 GB buffer with different task counts,
# and the same with random 3 MB "experts" (the MoE access pattern).
from std.time import perf_counter_ns
from std.memory import alloc
from std.random import random_ui64, seed
from max.algorithm import parallelize


def main() raises:
    var n = 1536 * 1024 * 1024
    var buf = alloc[UInt8](n)
    for i in range(0, n, 4096):
        buf[i] = UInt8(i % 7)
    var sums = alloc[UInt64](256)
    for tasks in [12, 24, 48, 96]:
        var best = 1 << 62
        for _ in range(3):
            var t0 = perf_counter_ns()

            def work(c: Int) {imm}:
                var acc = SIMD[DType.uint64, 4](0)
                var lo = c * (n // tasks)
                var hi = (c + 1) * (n // tasks)
                for i in range(lo, hi, 128):
                    var v = (buf + i).bitcast[UInt64]().load[width=16]()
                    acc += v.slice[4, offset=0]() ^ v.slice[4, offset=4]() ^ v.slice[4, offset=8]() ^ v.slice[4, offset=12]()
                sums[c] = acc.reduce_add()

            parallelize(work, tasks)
            best = min(best, perf_counter_ns() - t0)
        print("sequential, tasks", tasks, ":", Float64(n) / Float64(best), "GB/s")
    # random 3 MB pieces, 31 MB per "token", like 10 experts
    seed(3)
    var pieces = 10
    var psz = 3 * 1024 * 1024
    var offs = alloc[Int](pieces * 200)
    for i in range(pieces * 200):
        offs[i] = Int(random_ui64(0, UInt64(n // psz - 1))) * psz
    for tasks in [12, 24, 48]:
        var t0 = perf_counter_ns()
        for tok in range(200):
            def work2(c: Int) {imm}:
                var acc = SIMD[DType.uint64, 4](0)
                var total = pieces * psz
                for j in range(c * total // tasks, (c + 1) * total // tasks, 128):
                    var p = offs[tok * pieces + j // psz] + j % psz
                    var v = (buf + p).bitcast[UInt64]().load[width=16]()
                    acc += v.slice[4, offset=0]() ^ v.slice[4, offset=4]() ^ v.slice[4, offset=8]() ^ v.slice[4, offset=12]()
                sums[c] = acc.reduce_add()

            parallelize(work2, tasks)
        var dt = perf_counter_ns() - t0
        print("random 10x3MB per token, tasks", tasks, ":", Float64(200 * pieces * psz) / Float64(dt), "GB/s,", Float64(dt) / 200000.0, "us/token")
