# GPU side of the expert layer: the same qexperts.mojo math with W = 1 (one lane per thread, 32 lanes per warp).
# The warp's 32 lane sums are combined with shuffle_xor 16, 8, 4, 2, 1: the same tree as tree32 on the CPU.
from std.math import fma
from max.gpu import block_idx, block_dim, thread_idx
from max.gpu.primitives.warp import shuffle_xor
from max.gpu.sync import barrier
from std.memory import unsafe_stack_allocation
from qexperts import row_dot_q4k, row_dot_q51, row_dot_q80, silu_mul, q51_block, Q51_BYTES, Q4K_BYTES, scale_min_k4, u2f
from max.gpu.primitives.warp import shuffle_idx
from std.memory import bitcast

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
    ids: Pointer[Int32, ImmutAnyOrigin], expert_bytes_: Int64, k_: Int64,
    xp: Pointer[Float32, ImmutAnyOrigin], h: Pointer[Float32, MutAnyOrigin],
):
    """One warp per (expert slot e, row r): h[e, r] = silu(gate_e[r] . x) * (up_e[r] . x)."""
    var expert_bytes = Int(expert_bytes_)
    var k = Int(k_)
    var warp = Int(block_idx.x) * WARPS_PER_BLOCK + Int(thread_idx.x) // 32
    var lane = Int(thread_idx.x) % 32
    if warp >= k * D_FF:
        return
    var e = warp // D_FF
    var r = warp % D_FF
    var off = Int(ids[e]) * expert_bytes + r * Q4K_ROW
    var g = warp_tree(row_dot_q4k[1, D_MODEL // 256](gate + off, xp, lane)[0])  # xp is x itself on the GPU
    var u = warp_tree(row_dot_q4k[1, D_MODEL // 256](up + off, xp, lane)[0])
    if lane == 0:
        h[warp] = silu_mul[1](g, u)[0]


def down_kernel(
    down: Pointer[UInt8, ImmutAnyOrigin], down_q80: Int64,
    ids: Pointer[Int32, ImmutAnyOrigin], w: Pointer[Float32, ImmutAnyOrigin], expert_bytes_: Int64, k_: Int64,
    h: Pointer[Float32, ImmutAnyOrigin], y: Pointer[Float32, MutAnyOrigin],
):
    """One warp per output row i: y[i] = sum_e w[e] * (down_e[i] . h_e), experts in slot order (sorted ids)."""
    var expert_bytes = Int(expert_bytes_)
    var k = Int(k_)
    var i = Int(block_idx.x) * WARPS_PER_BLOCK + Int(thread_idx.x) // 32
    var lane = Int(thread_idx.x) % 32
    if i >= D_MODEL:
        return
    var o: Float32 = 0
    for e in range(k):
        var v: Float32
        if down_q80 == 1:
            v = warp_tree(row_dot_q80[1, D_FF // 32](down + Int(ids[e]) * expert_bytes + i * Q80_ROW, h + e * D_FF, lane)[0])
        else:
            v = warp_tree(row_dot_q51[1, D_FF // 32](down + Int(ids[e]) * expert_bytes + i * Q51_ROW, h + e * D_FF, lane)[0])
        o = fma(w[e], v, o)
    if lane == 0:
        y[i] = o


comptime ROWS_PER_BLOCK = 8
comptime MAX_K = 16


def down_kernel_par(
    down: Pointer[UInt8, ImmutAnyOrigin], down_q80: Int64,
    ids: Pointer[Int32, ImmutAnyOrigin], w: Pointer[Float32, ImmutAnyOrigin], expert_bytes_: Int64, k_: Int64,
    h: Pointer[Float32, ImmutAnyOrigin], y: Pointer[Float32, MutAnyOrigin],
):
    """Block = k warps (one per expert) x ROWS_PER_BLOCK output rows. Each lane loads its 24 values of h_e into
    registers once and reuses them for every row (the h reloads, not the weights, were half the time). Then one thread
    per row combines the k expert values in slot order. Same per-lane arithmetic as the CPU (q51_block)."""
    var expert_bytes = Int(expert_bytes_)
    var k = Int(k_)
    var e = Int(thread_idx.x) // 32
    var lane = Int(thread_idx.x) % 32
    var i0 = Int(block_idx.x) * ROWS_PER_BLOCK
    var sv = unsafe_stack_allocation[ROWS_PER_BLOCK * MAX_K, Float32, address_space=.SHARED]()
    var base = down + Int(ids[e]) * expert_bytes
    var he = h + e * D_FF
    if down_q80 == 1:
        comptime for r in range(ROWS_PER_BLOCK):
            var v = warp_tree(row_dot_q80[1, D_FF // 32](base + (i0 + r) * Q80_ROW, he, lane)[0])
            if lane == 0:
                sv[unsafe_offset=r * MAX_K + e] = v
    else:
        var p = lane % 4
        var hl0 = he.load[width=4](0 * 256 + (lane // 4) * 32 + 4 * p)
        var hh0 = he.load[width=4](0 * 256 + (lane // 4) * 32 + 16 + 4 * p)
        var hl1 = he.load[width=4](1 * 256 + (lane // 4) * 32 + 4 * p)
        var hh1 = he.load[width=4](1 * 256 + (lane // 4) * 32 + 16 + 4 * p)
        var hl2 = SIMD[DType.float32, 4](0)
        var hh2 = SIMD[DType.float32, 4](0)
        if lane < 16:
            hl2 = he.load[width=4](2 * 256 + (lane // 4) * 32 + 4 * p)
            hh2 = he.load[width=4](2 * 256 + (lane // 4) * 32 + 16 + 4 * p)
        comptime for r in range(ROWS_PER_BLOCK):
            var row = base + (i0 + r) * Q51_ROW
            var acc: Float32 = 0
            acc = q51_block(row + (0 * 8 + lane // 4) * Q51_BYTES, p, hl0, hh0, acc)
            acc = q51_block(row + (1 * 8 + lane // 4) * Q51_BYTES, p, hl1, hh1, acc)
            if lane < 16:
                acc = q51_block(row + (2 * 8 + lane // 4) * Q51_BYTES, p, hl2, hh2, acc)
            var v = warp_tree(acc)
            if lane == 0:
                sv[unsafe_offset=r * MAX_K + e] = v
    barrier()
    if Int(thread_idx.x) < ROWS_PER_BLOCK:
        var r = Int(thread_idx.x)
        var o: Float32 = 0
        for ee in range(k):
            o = fma(w[ee], sv[unsafe_offset=r * MAX_K + ee][0], o)
        y[i0 + r] = o


comptime GU_ROWS = 1  # gate rows (and the matching up rows) per warp


@inline(.always)
def q4k_block_gpu(b: Pointer[UInt8, _], lane: Int, xl: SIMD[DType.float32, 4], xh: SIMD[DType.float32, 4], acc_in: Float32) -> Float32:
    """One lane's 8 values of one Q4_K block with x already in registers. Same arithmetic, in the same order, as
    row_dot_q4k (CPU): acc = fma(fma(d1, q, -m1), x, acc) over low nibbles k = 0..3, then high with d2, m2."""
    var hv = b.bitcast[UInt32]().load[width=4]()
    var d = bitcast[DType.float16, 1](UInt16(hv[0] & 0xFFFF)).cast[DType.float32]()
    var dmin = bitcast[DType.float16, 1](UInt16(hv[0] >> 16)).cast[DType.float32]()
    var mine = scale_min_k4(hv[1], hv[2], hv[3], lane % 8)
    var my_d = d * Float32(mine[0])
    var my_m = dmin * Float32(mine[1])
    var j = lane // 8
    var d1 = shuffle_idx(my_d, UInt32(2 * j))
    var m1 = shuffle_idx(my_m, UInt32(2 * j))
    var d2 = shuffle_idx(my_d, UInt32(2 * j + 1))
    var m2 = shuffle_idx(my_m, UInt32(2 * j + 1))
    var q = (b + 16).bitcast[UInt32]()[lane]
    var acc = acc_in
    comptime for k in range(4):
        acc = fma(fma(d1, u2f[1]((q >> UInt32(8 * k)) & 0xF)[0], -m1), xl[k], acc)
    comptime for k in range(4):
        acc = fma(fma(d2, u2f[1]((q >> UInt32(8 * k + 4)) & 0xF)[0], -m2), xh[k], acc)
    return acc


def gate_up_kernel_reg(
    gate: Pointer[UInt8, ImmutAnyOrigin], up: Pointer[UInt8, ImmutAnyOrigin],
    ids: Pointer[Int32, ImmutAnyOrigin], expert_bytes_: Int64, k_: Int64,
    x: Pointer[Float32, ImmutAnyOrigin], h: Pointer[Float32, MutAnyOrigin],
):
    """One warp per (expert slot, GU_ROWS consecutive rows). Each lane holds its 80 values of x in registers for all
    rows, so per row only the weights are loaded."""
    var expert_bytes = Int(expert_bytes_)
    var k = Int(k_)
    var warp = Int(block_idx.x) * WARPS_PER_BLOCK + Int(thread_idx.x) // 32
    var lane = Int(thread_idx.x) % 32
    if warp >= k * (D_FF // GU_ROWS):
        return
    var e = warp // (D_FF // GU_ROWS)
    var r0 = (warp % (D_FF // GU_ROWS)) * GU_ROWS
    var xbase = 64 * (lane // 8) + 4 * (lane % 8)
    var xl = SIMD[DType.float32, 64](0)  # 10 blocks x 4 values, padded to a power of two (stays in registers)
    var xh = SIMD[DType.float32, 64](0)
    comptime for blk in range(D_MODEL // 256):
        xl = xl.insert[offset=4 * blk](x.load[width=4](blk * 256 + xbase))
        xh = xh.insert[offset=4 * blk](x.load[width=4](blk * 256 + xbase + 32))
    var eoff = Int(ids[e]) * expert_bytes
    comptime for r in range(GU_ROWS):
        var grow = gate + eoff + (r0 + r) * Q4K_ROW
        var urow = up + eoff + (r0 + r) * Q4K_ROW
        var ga: Float32 = 0
        var ua: Float32 = 0
        comptime for blk in range(D_MODEL // 256):
            ga = q4k_block_gpu(grow + blk * Q4K_BYTES, lane, xl.slice[4, offset=4 * blk](), xh.slice[4, offset=4 * blk](), ga)
            ua = q4k_block_gpu(urow + blk * Q4K_BYTES, lane, xl.slice[4, offset=4 * blk](), xh.slice[4, offset=4 * blk](), ua)
        var g = warp_tree(ga)
        var u = warp_tree(ua)
        if lane == 0:
            h[e * D_FF + r0 + r] = silu_mul[1](g, u)[0]


def gate_up_kernel_split(
    gate: Pointer[UInt8, ImmutAnyOrigin], up: Pointer[UInt8, ImmutAnyOrigin],
    ids: Pointer[Int32, ImmutAnyOrigin], expert_bytes_: Int64, k_: Int64,
    x: Pointer[Float32, ImmutAnyOrigin], h: Pointer[Float32, MutAnyOrigin],
):
    """Even warps do a gate row, odd warps the matching up row (2x the warps in flight); they meet in shared memory."""
    var expert_bytes = Int(expert_bytes_)
    var k = Int(k_)
    var wib = Int(thread_idx.x) // 32
    var pair = Int(block_idx.x) * (WARPS_PER_BLOCK // 2) + wib // 2
    var lane = Int(thread_idx.x) % 32
    var sg = unsafe_stack_allocation[WARPS_PER_BLOCK, Float32, address_space=.SHARED]()
    var active = pair < k * D_FF
    if active:
        var e = pair // D_FF
        var r = pair % D_FF
        var mat = gate if wib % 2 == 0 else up
        var row = mat + Int(ids[e]) * expert_bytes + r * Q4K_ROW
        var xbase = 64 * (lane // 8) + 4 * (lane % 8)
        var a: Float32 = 0
        comptime for blk in range(D_MODEL // 256):
            a = q4k_block_gpu(row + blk * Q4K_BYTES, lane, x.load[width=4](blk * 256 + xbase),
                              x.load[width=4](blk * 256 + xbase + 32), a)
        var v = warp_tree(a)
        if lane == 0:
            sg[unsafe_offset=wib] = v
    barrier()
    if active and lane == 0 and wib % 2 == 0:
        h[pair] = silu_mul[1](sg[unsafe_offset=wib][0], sg[unsafe_offset=wib + 1][0])[0]
