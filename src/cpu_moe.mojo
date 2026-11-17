# CPU side of the expert layer: same qexperts.mojo math with W = 32 (one AVX2 lane vector), rows spread over all cores.
from std.math import fma
from max.algorithm import parallelize
from std.runtime import parallelism_level
from std.atomic import Atomic
from std.sys.intrinsics import prefetch
from qexperts import row_dot_q5k, row_dot_q4k, row_dot_q51, row_dot_q80, tree32, silu_mul, permute_x_q4k, arrange_h_q51

comptime D_MODEL = 2560
comptime D_FF = 640
comptime Q4K_ROW = 1440  # 10 blocks x 144
comptime Q51_ROW = 480   # 20 blocks x 24
comptime Q5K_ROW = 1760  # 10 blocks x 176
comptime Q80_ROW = 680   # 20 blocks x 34
comptime HV_PER_EXPERT = 3 * 8 * 32  # arranged h for Q5_1 down (3 iterations x 8 steps x 32 lanes)


def moe_cpu(
    gate: Pointer[UInt8, _], up: Pointer[UInt8, _], down: Pointer[UInt8, _], down_q80: Bool,
    expert_off_gu: Pointer[Int, _], expert_off_d: Pointer[Int, _], k: Int,
    w: Pointer[Float32, _], x: Pointer[Float32, _],
    xp: Pointer[mut=True, Float32, _], h: Pointer[mut=True, Float32, _], hv: Pointer[mut=True, Float32, _], s: Pointer[mut=True, Float32, _], y: Pointer[mut=True, Float32, _], stages: Int = 3, gate_q5k: Bool = False,
):
    """out = sum_e w[e] * down_e(silu(gate_e x) * up_e x), experts in the given (sorted) order.
    expert_off_*: byte offset of each selected expert inside gate/up and down. s: per-expert down sums [k, D_MODEL]."""
    permute_x_q4k(x, xp, D_MODEL)

    comptime CHUNKS = 48   # a few tasks per hardware thread: task overhead stays small, load stays balanced
    var na = k * D_FF

    def stage_a(c: Int) {imm}:
        for t in range(c * na // CHUNKS, (c + 1) * na // CHUNKS):
            var e = t // D_FF
            var r = t % D_FF
            var ahead = min(t + 4, na - 1)
            var pa = expert_off_gu[ahead // D_FF] + (ahead % D_FF) * (Q5K_ROW if gate_q5k else Q4K_ROW)
            comptime for cl in range(0, Q4K_ROW, 64):
                prefetch((gate + pa + cl).bitcast[Int8]())
                prefetch((up + pa + cl).bitcast[Int8]())
            var g: Float32
            var u: Float32
            if gate_q5k:
                g = tree32(row_dot_q5k[32, D_MODEL // 256](gate + expert_off_gu[e] + r * Q5K_ROW, xp, 0))
                u = tree32(row_dot_q5k[32, D_MODEL // 256](up + expert_off_gu[e] + r * Q5K_ROW, xp, 0))
            else:
                g = tree32(row_dot_q4k[32, D_MODEL // 256](gate + expert_off_gu[e] + r * Q4K_ROW, xp, 0))
                u = tree32(row_dot_q4k[32, D_MODEL // 256](up + expert_off_gu[e] + r * Q4K_ROW, xp, 0))
            h[t] = silu_mul[1](g, u)[0]

    if stages & 1:
        parallelize(stage_a, CHUNKS)
    if not down_q80:
        for e in range(k):
            arrange_h_q51(h + e * D_FF, hv + e * HV_PER_EXPERT, D_FF // 32)

    def stage_b(c: Int) {imm}:
        var i0 = c * D_MODEL // CHUNKS
        var i1 = (c + 1) * D_MODEL // CHUNKS
        for e in range(k):
            for i in range(i0, i1):
                var v: Float32
                if down_q80:
                    v = tree32(row_dot_q80[32, D_FF // 32](down + expert_off_d[e] + i * Q80_ROW, h + e * D_FF, 0))
                else:
                    var nxt = down + expert_off_d[e] + min(i + 4, D_MODEL - 1) * Q51_ROW
                    comptime for cl in range(0, Q51_ROW, 64):
                        prefetch((nxt + cl).bitcast[Int8]())
                    v = tree32(row_dot_q51[32, D_FF // 32](down + expert_off_d[e] + i * Q51_ROW, hv + e * HV_PER_EXPERT, 0))
                s[e * D_MODEL + i] = v
        for i in range(i0, i1):
            var o: Float32 = 0
            for e in range(k):
                o = fma(w[e], s[e * D_MODEL + i], o)
            y[i] = o

    if stages & 2:
        parallelize(stage_b, CHUNKS)



@inline(.always)
def spin_barrier(ctr: Pointer[mut=True, Int64, _], target: Int64):
    _ = Atomic.fetch_add(ctr, Int64(1))
    while Atomic.load(ctr) < target:
        pass


def moe_cpu_fused(
    gate: Pointer[UInt8, _], up: Pointer[UInt8, _], down: Pointer[UInt8, _], down_q80: Bool,
    expert_off_gu: Pointer[Int, _], expert_off_d: Pointer[Int, _], k: Int,
    w: Pointer[Float32, _], x: Pointer[Float32, _],
    xp: Pointer[mut=True, Float32, _], h: Pointer[mut=True, Float32, _], hv: Pointer[mut=True, Float32, _],
    s: Pointer[mut=True, Float32, _], y: Pointer[mut=True, Float32, _], bar: Pointer[mut=True, Int64, _],
):
    """Same math as moe_cpu, one parallel region: T workers (= runtime threads) do gate/up, meet at a spin barrier,
    worker 0..k-1 arrange h, meet again, then do down. One dispatch per layer instead of two (bar: 2 zeroed Int64)."""
    var T = parallelism_level()
    permute_x_q4k(x, xp, D_MODEL)
    var na = k * D_FF

    def worker(c: Int) {imm}:
        for t in range(c * na // T, (c + 1) * na // T):
            var e = t // D_FF
            var r = t % D_FF
            var g = tree32(row_dot_q4k[32, D_MODEL // 256](gate + expert_off_gu[e] + r * Q4K_ROW, xp, 0))
            var u = tree32(row_dot_q4k[32, D_MODEL // 256](up + expert_off_gu[e] + r * Q4K_ROW, xp, 0))
            h[t] = silu_mul[1](g, u)[0]
        spin_barrier(bar, Int64(T))
        if not down_q80:
            var e = c
            while e < k:
                arrange_h_q51(h + e * D_FF, hv + e * HV_PER_EXPERT, D_FF // 32)
                e += T
        spin_barrier(bar + 1, Int64(T))
        for i in range(c * D_MODEL // T, (c + 1) * D_MODEL // T):
            for e in range(k):
                var v: Float32
                if down_q80:
                    v = tree32(row_dot_q80[32, D_FF // 32](down + expert_off_d[e] + i * Q80_ROW, h + e * D_FF, 0))
                else:
                    v = tree32(row_dot_q51[32, D_FF // 32](down + expert_off_d[e] + i * Q51_ROW, hv + e * HV_PER_EXPERT, 0))
                s[e * D_MODEL + i] = v
            var o: Float32 = 0
            for e in range(k):
                o = fma(w[e], s[e * D_MODEL + i], o)
            y[i] = o

    parallelize(worker, T)
