# GPU side of the expert layer: the same qexperts.mojo math with W = 1 (one lane per thread, 32 lanes per warp).
# The warp's 32 lane sums are combined with shuffle_xor 16, 8, 4, 2, 1: the same tree as tree32 on the CPU.
from std.math import fma
from max.gpu import block_idx, block_dim, thread_idx
from max.gpu.primitives.warp import shuffle_xor
from qexperts import row_dot_q4k, row_dot_q51, row_dot_q80, silu_mul

comptime D_MODEL = 2560
comptime D_FF = 640
comptime Q4K_ROW = 1440
comptime Q51_ROW = 480
comptime Q80_ROW = 680
comptime WARPS_PER_BLOCK = 8
comptime TPB = WARPS_PER_BLOCK * 32


@inline(.always)
def warp_tree(v: Float32) -> Float32:
    var a = v + shuffle_xor(v, 16)
    a = a + shuffle_xor(a, 8)
    a = a + shuffle_xor(a, 4)
    a = a + shuffle_xor(a, 2)
    a = a + shuffle_xor(a, 1)
    return a


def permute_x_kernel(x: Pointer[Float32, ImmutAnyOrigin], xp: Pointer[Float32, MutAnyOrigin]):
    """xp[blk*256 + k*32 + l] = x[blk*256 + 64*(l/8) + 4*(l%8) + (k%4) + 32*(k/4)] (same permutation as the CPU)."""
    var t = Int(block_idx.x * block_dim.x + thread_idx.x)
    if t < D_MODEL:
        var blk = t // 256
        var k = (t % 256) // 32
        var l = t % 32
        xp[t] = x[blk * 256 + 64 * (l // 8) + 4 * (l % 8) + (k % 4) + 32 * (k // 4)]


def gate_up_kernel(
    gate: Pointer[UInt8, ImmutAnyOrigin], up: Pointer[UInt8, ImmutAnyOrigin],
    ids: Pointer[Int32, ImmutAnyOrigin], expert_bytes: Int, k: Int,
    xp: Pointer[Float32, ImmutAnyOrigin], h: Pointer[Float32, MutAnyOrigin],
):
    """One warp per (expert slot e, row r): h[e, r] = silu(gate_e[r] . x) * (up_e[r] . x)."""
    var warp = Int(block_idx.x) * WARPS_PER_BLOCK + Int(thread_idx.x) // 32
    var lane = Int(thread_idx.x) % 32
    if warp >= k * D_FF:
        return
    var e = warp // D_FF
    var r = warp % D_FF
    var off = Int(ids[e]) * expert_bytes + r * Q4K_ROW
    var g = warp_tree(row_dot_q4k[1](gate + off, D_MODEL // 256, xp, lane)[0])
    var u = warp_tree(row_dot_q4k[1](up + off, D_MODEL // 256, xp, lane)[0])
    if lane == 0:
        h[warp] = silu_mul[1](g, u)[0]


def down_kernel(
    down: Pointer[UInt8, ImmutAnyOrigin], down_q80: Bool,
    ids: Pointer[Int32, ImmutAnyOrigin], w: Pointer[Float32, ImmutAnyOrigin], expert_bytes: Int, k: Int,
    h: Pointer[Float32, ImmutAnyOrigin], y: Pointer[Float32, MutAnyOrigin],
):
    """One warp per output row i: y[i] = sum_e w[e] * (down_e[i] . h_e), experts in slot order (sorted ids)."""
    var i = Int(block_idx.x) * WARPS_PER_BLOCK + Int(thread_idx.x) // 32
    var lane = Int(thread_idx.x) % 32
    if i >= D_MODEL:
        return
    var o: Float32 = 0
    for e in range(k):
        var v: Float32
        if down_q80:
            v = warp_tree(row_dot_q80[1](down + Int(ids[e]) * expert_bytes + i * Q80_ROW, D_FF // 32, h + e * D_FF, lane)[0])
        else:
            v = warp_tree(row_dot_q51[1](down + Int(ids[e]) * expert_bytes + i * Q51_ROW, D_FF // 32, h + e * D_FF, lane)[0])
        o = fma(w[e], v, o)
    if lane == 0:
        y[i] = o
