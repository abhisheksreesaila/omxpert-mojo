# CPU side of the expert layer: same qexperts.mojo math with W = 32 (one AVX2 lane vector), rows spread over all cores.
from std.math import fma
from max.algorithm import parallelize
from qexperts import row_dot_q4k, row_dot_q51, row_dot_q80, tree32, silu_mul, permute_x_q4k

comptime D_MODEL = 2560
comptime D_FF = 640
comptime Q4K_ROW = 1440  # 10 blocks x 144
comptime Q51_ROW = 480   # 20 blocks x 24
comptime Q80_ROW = 680   # 20 blocks x 34


def moe_cpu(
    gate: Pointer[UInt8, _], up: Pointer[UInt8, _], down: Pointer[UInt8, _], down_q80: Bool,
    expert_off_gu: Pointer[Int, _], expert_off_d: Pointer[Int, _], k: Int,
    w: Pointer[Float32, _], x: Pointer[Float32, _],
    xp: Pointer[mut=True, Float32, _], h: Pointer[mut=True, Float32, _], s: Pointer[mut=True, Float32, _], y: Pointer[mut=True, Float32, _],
):
    """out = sum_e w[e] * down_e(silu(gate_e x) * up_e x), experts in the given (sorted) order.
    expert_off_*: byte offset of each selected expert inside gate/up and down. s: per-expert down sums [k, D_MODEL]."""
    permute_x_q4k(x, xp, D_MODEL)

    def stage_a(t: Int) {imm}:
        var e = t // D_FF
        var r = t % D_FF
        var g = tree32(row_dot_q4k[32, D_MODEL // 256](gate + expert_off_gu[e] + r * Q4K_ROW, xp, 0))
        var u = tree32(row_dot_q4k[32, D_MODEL // 256](up + expert_off_gu[e] + r * Q4K_ROW, xp, 0))
        h[t] = silu_mul[1](g, u)[0]

    parallelize(stage_a, k * D_FF)

    def stage_b(i: Int) {imm}:
        for e in range(k):
            var v: Float32
            if down_q80:
                v = tree32(row_dot_q80[32, D_FF // 32](down + expert_off_d[e] + i * Q80_ROW, h + e * D_FF, 0))
            else:
                v = tree32(row_dot_q51[32, D_FF // 32](down + expert_off_d[e] + i * Q51_ROW, h + e * D_FF, 0))
            s[e * D_MODEL + i] = v
        var o: Float32 = 0
        for e in range(k):
            o = fma(w[e], s[e * D_MODEL + i], o)
        y[i] = o

    parallelize(stage_b, D_MODEL)
