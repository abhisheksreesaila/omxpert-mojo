# omxpert-mojo: routed-expert math that compiles for both the GPU and the CPU from ONE source.
#
# The trick: every routine works on a "lane vector" SIMD[float32, W].
#   GPU: W = 1, one warp = 32 threads, each thread is one lane (lane0 = its lane id).
#   CPU: W = 32, the 32 lanes are one SIMD vector (lane0 = 0), AVX2 lowers it to 4 ymm registers.
# Each lane runs the same fma chain in the same order, and the 32 lane sums are combined with the same
# butterfly tree (warp shuffles on the GPU, vector halving on the CPU). So the result is bit-identical
# whichever device ran the expert: placement can never change the model's output.
#
# Formats (ggml): Q4_K (gate/up), Q5_1 and Q8_0 (down). Activations stay float32; weights are dequantised
# in registers and never written back.

from std.math import fma, floor
from std.memory import bitcast
from max.gpu.primitives.warp import shuffle_idx
from std.utils import StaticTuple

comptime F32 = DType.float32
comptime MAGIC = UInt32(0x4B000000)  # bit pattern of 2^23: (MAGIC | q) as float - 2^23 == float(q) exactly, for q < 2^23


@inline(.always)
def u2f[W: Int](q: SIMD[DType.uint32, W]) -> SIMD[F32, W]:
    """Exact small-integer -> float without the quarter-rate I2F instruction: one OR and one FADD."""
    return bitcast[F32, W](q | MAGIC) - 8388608.0

# ---------------------------------------------------------------- small helpers


@inline(.always)
def f16_at(p: Pointer[UInt8, _], off: Int) -> Float32:
    var lo = UInt16(p[off])
    var hi = UInt16(p[off + 1])
    return bitcast[DType.float16, 1](lo | (hi << 8)).cast[F32]()


@inline(.always)
def u32_lanes[W: Int](p: Pointer[UInt8, _], lane0: Int) -> SIMD[DType.uint32, W]:
    """Lane l gets the little-endian uint32 at p + 4*l (lanes lane0 .. lane0+W-1)."""
    var out = SIMD[DType.uint32, W](0)
    comptime for i in range(W):
        var b = p + 4 * (lane0 + i)
        out[i] = UInt32(b[0]) | (UInt32(b[1]) << 8) | (UInt32(b[2]) << 16) | (UInt32(b[3]) << 24)
    return out


@inline(.always)
def lane_iota[W: Int](lane0: Int) -> SIMD[DType.int32, W]:
    var v = SIMD[DType.int32, W](0)
    comptime for i in range(W):
        v[i] = Int32(lane0 + i)
    return v


# ---------------------------------------------------------------- deterministic exp / SiLU
# Our own exp so the CPU and GPU agree bit for bit (library exp differs between targets).


@inline(.always)
def exp_det[W: Int](x_in: SIMD[F32, W]) -> SIMD[F32, W]:
    var x = x_in.clamp(-87.0, 88.0)
    var n = floor(fma(x, SIMD[F32, W](1.4426950408889634), SIMD[F32, W](0.5)))
    var r = fma(n, SIMD[F32, W](-0.693145751953125), x)  # ln2 hi
    r = fma(n, SIMD[F32, W](-1.428606765330187e-06), r)  # ln2 lo
    var p = SIMD[F32, W](1.9875691500e-4)
    p = fma(p, r, SIMD[F32, W](1.3981999507e-3))
    p = fma(p, r, SIMD[F32, W](8.3334519073e-3))
    p = fma(p, r, SIMD[F32, W](4.1665795894e-2))
    p = fma(p, r, SIMD[F32, W](1.6666665459e-1))
    p = fma(p, r, SIMD[F32, W](5.0000001201e-1))
    p = fma(p, r * r, r) + 1.0
    var e = (n.cast[DType.int32]() + 127) << 23
    return p * bitcast[F32, W](e)


@inline(.always)
def silu_mul[W: Int](g: SIMD[F32, W], u: SIMD[F32, W]) -> SIMD[F32, W]:
    return g / (exp_det[W](-g) + 1.0) * u


# ---------------------------------------------------------------- butterfly tree over 32 lanes (CPU side)


@inline(.always)
def tree32(v: SIMD[F32, 32]) -> Float32:
    var a = v.slice[16, offset=0]() + v.slice[16, offset=16]()
    var b = a.slice[8, offset=0]() + a.slice[8, offset=8]()
    var c = b.slice[4, offset=0]() + b.slice[4, offset=4]()
    var d = c.slice[2, offset=0]() + c.slice[2, offset=2]()
    return d[0] + d[1]


# ---------------------------------------------------------------- Q4_K row dot (gate / up)
# block: d f16 | dmin f16 | scales[12] (6-bit scale+min for 8 sub-blocks of 32) | qs[128]  = 144 bytes / 256 values
# Lane l owns qs bytes 4l..4l+3: chunk j = l/8 (64 values), sub-blocks 2j (low nibbles) and 2j+1 (high nibbles).
# xp is x permuted into lane order: xp[blk*256 + k*32 + l], k = 0..3 low nibble of byte k, 4..7 high nibble.

comptime Q4K_BYTES = 144


@inline(.always)
def byte_of(s0: UInt32, s1: UInt32, s2: UInt32, j: Int) -> UInt32:
    """Byte j (0..11) of the 12 scale bytes held in three little-endian words."""
    var w = s0 if j < 4 else (s1 if j < 8 else s2)
    return (w >> UInt32(8 * (j % 4))) & 0xFF


@inline(.always)
def scale_min_k4(s0: UInt32, s1: UInt32, s2: UInt32, j: Int) -> Tuple[UInt32, UInt32]:
    if j < 4:
        return (byte_of(s0, s1, s2, j) & 63, byte_of(s0, s1, s2, j + 4) & 63)
    return (
        (byte_of(s0, s1, s2, j + 4) & 0xF) | ((byte_of(s0, s1, s2, j - 4) >> 6) << 4),
        (byte_of(s0, s1, s2, j + 4) >> 4) | ((byte_of(s0, s1, s2, j) >> 6) << 4),
    )


@inline(.always)
def row_dot_q4k[W: Int, NBLK: Int](row: Pointer[UInt8, _], xp: Pointer[Float32, _], lane0: Int) -> SIMD[F32, W]:
    """xp: x itself when W == 1 (GPU), the lane-permuted copy when W == 32 (CPU); the values each lane sees are equal.
    Fully unrolled over NBLK blocks: every block's loads can be in flight at once. Arithmetic order per lane is
    fixed (block, then low nibbles k = 0..3, then high nibbles), identical on CPU and GPU."""
    var acc = SIMD[F32, W](0)
    comptime for blk in range(NBLK):
        var b = row + blk * Q4K_BYTES
        var hv = b.bitcast[UInt32]().load[width=4]()  # d|dmin, scales[0:12] in one 16-byte load
        var h0 = hv[0]
        var s0 = hv[1]
        var s1 = hv[2]
        var s2 = hv[3]
        var d = bitcast[DType.float16, 1](UInt16(h0 & 0xFFFF)).cast[F32]()
        var dmin = bitcast[DType.float16, 1](UInt16(h0 >> 16)).cast[F32]()
        var d1 = SIMD[F32, W](0)
        var m1 = SIMD[F32, W](0)
        var d2 = SIMD[F32, W](0)
        var m2 = SIMD[F32, W](0)
        comptime if W == 1:
            # GPU: lane t decodes sub-block t % 8 once; lanes fetch sub-blocks 2j, 2j+1 by shuffle (no divergence per lane)
            var mine = scale_min_k4(s0, s1, s2, lane0 % 8)
            var my_d = d * Float32(mine[0])
            var my_m = dmin * Float32(mine[1])
            var j = lane0 // 8
            d1[0] = shuffle_idx(my_d, UInt32(2 * j))
            m1[0] = shuffle_idx(my_m, UInt32(2 * j))
            d2[0] = shuffle_idx(my_d, UInt32(2 * j + 1))
            m2[0] = shuffle_idx(my_m, UInt32(2 * j + 1))
        else:
            var dd = SIMD[F32, 32](0)  # sub-block values in lanes 0..7, spread below by shuffles
            var mm = SIMD[F32, 32](0)
            comptime for t in range(8):
                var sm = scale_min_k4(s0, s1, s2, t)
                dd[t] = d * Float32(sm[0])
                mm[t] = dmin * Float32(sm[1])
            d1 = rebind[SIMD[F32, W]](dd.shuffle[0,0,0,0,0,0,0,0,2,2,2,2,2,2,2,2,4,4,4,4,4,4,4,4,6,6,6,6,6,6,6,6]())
            m1 = rebind[SIMD[F32, W]](mm.shuffle[0,0,0,0,0,0,0,0,2,2,2,2,2,2,2,2,4,4,4,4,4,4,4,4,6,6,6,6,6,6,6,6]())
            d2 = rebind[SIMD[F32, W]](dd.shuffle[1,1,1,1,1,1,1,1,3,3,3,3,3,3,3,3,5,5,5,5,5,5,5,5,7,7,7,7,7,7,7,7]())
            m2 = rebind[SIMD[F32, W]](mm.shuffle[1,1,1,1,1,1,1,1,3,3,3,3,3,3,3,3,5,5,5,5,5,5,5,5,7,7,7,7,7,7,7,7]())
        var q = (b + 16).bitcast[UInt32]().load[width=W](lane0)
        comptime if W == 1:
            # x natural layout: lane's 4 low-nibble values and 4 high-nibble values are contiguous -> 2 vector loads
            var base = blk * 256 + 64 * (lane0 // 8) + 4 * (lane0 % 8)
            var xl = xp.load[width=4](base)
            var xh = xp.load[width=4](base + 32)
            comptime for k in range(4):
                var ql = u2f[W]((q >> UInt32(8 * k)) & 0xF)
                acc = fma(fma(d1, ql, -m1), SIMD[F32, W](xl[k]), acc)
            comptime for k in range(4):
                var qh = u2f[W]((q >> UInt32(8 * k + 4)) & 0xF)
                acc = fma(fma(d2, qh, -m2), SIMD[F32, W](xh[k]), acc)
        else:
            var xb = xp + blk * 256 + lane0
            comptime for k in range(4):
                var ql = u2f[W]((q >> UInt32(8 * k)) & 0xF)
                acc = fma(fma(d1, ql, -m1), xb.load[width=W](k * 32), acc)
            comptime for k in range(4):
                var qh = u2f[W]((q >> UInt32(8 * k + 4)) & 0xF)
                acc = fma(fma(d2, qh, -m2), xb.load[width=W]((k + 4) * 32), acc)
    return acc


# ---------------------------------------------------------------- Q5_K row dot (gate / up of layer 2)
# block: d f16 | dmin f16 | scales[12] | qh[32] | qs[128] = 176 bytes / 256 values. Same lane mapping as Q4_K; the
# 5th bit of value (chunk j, position i) is bit 2j (low nibble) or 2j+1 (high nibble) of qh[i]. Lane l's positions
# 4(l%8)+k sit in qh word l%8.

comptime Q5K_BYTES = 176


@inline(.always)
def row_dot_q5k[W: Int, NBLK: Int](row: Pointer[UInt8, _], xp: Pointer[Float32, _], lane0: Int) -> SIMD[F32, W]:
    var acc = SIMD[F32, W](0)
    comptime for blk in range(NBLK):
        var b = row + blk * Q5K_BYTES
        var hv = b.bitcast[UInt32]().load[width=4]()
        var d = bitcast[DType.float16, 1](UInt16(hv[0] & 0xFFFF)).cast[F32]()
        var dmin = bitcast[DType.float16, 1](UInt16(hv[0] >> 16)).cast[F32]()
        var d1 = SIMD[F32, W](0)
        var m1 = SIMD[F32, W](0)
        var d2 = SIMD[F32, W](0)
        var m2 = SIMD[F32, W](0)
        var q = (b + 48).bitcast[UInt32]().load[width=W](lane0)
        var qhw = SIMD[DType.uint32, W](0)
        var sh = SIMD[DType.uint32, W](0)
        comptime if W == 1:
            var mine = scale_min_k4(hv[1], hv[2], hv[3], lane0 % 8)
            var my_d = d * Float32(mine[0])
            var my_m = dmin * Float32(mine[1])
            var j = lane0 // 8
            d1[0] = shuffle_idx(my_d, UInt32(2 * j))
            m1[0] = shuffle_idx(my_m, UInt32(2 * j))
            d2[0] = shuffle_idx(my_d, UInt32(2 * j + 1))
            m2[0] = shuffle_idx(my_m, UInt32(2 * j + 1))
            qhw[0] = (b + 16).bitcast[UInt32]()[lane0 % 8]
            sh[0] = UInt32(2 * j)
        else:
            var dd = SIMD[F32, 32](0)
            var mm = SIMD[F32, 32](0)
            comptime for t in range(8):
                var sm = scale_min_k4(hv[1], hv[2], hv[3], t)
                dd[t] = d * Float32(sm[0])
                mm[t] = dmin * Float32(sm[1])
            d1 = rebind[SIMD[F32, W]](dd.shuffle[0,0,0,0,0,0,0,0,2,2,2,2,2,2,2,2,4,4,4,4,4,4,4,4,6,6,6,6,6,6,6,6]())
            m1 = rebind[SIMD[F32, W]](mm.shuffle[0,0,0,0,0,0,0,0,2,2,2,2,2,2,2,2,4,4,4,4,4,4,4,4,6,6,6,6,6,6,6,6]())
            d2 = rebind[SIMD[F32, W]](dd.shuffle[1,1,1,1,1,1,1,1,3,3,3,3,3,3,3,3,5,5,5,5,5,5,5,5,7,7,7,7,7,7,7,7]())
            m2 = rebind[SIMD[F32, W]](mm.shuffle[1,1,1,1,1,1,1,1,3,3,3,3,3,3,3,3,5,5,5,5,5,5,5,5,7,7,7,7,7,7,7,7]())
            var q8 = SIMD[DType.uint32, 32](0)
            q8 = q8.insert[offset=0]((b + 16).bitcast[UInt32]().load[width=8]())
            qhw = rebind[SIMD[DType.uint32, W]](q8.shuffle[0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7]())
            comptime for i in range(W):
                sh[i] = UInt32(2 * (i // 8))
        comptime if W == 1:
            var base = blk * 256 + 64 * (lane0 // 8) + 4 * (lane0 % 8)
            var xl = xp.load[width=4](base)
            var xh = xp.load[width=4](base + 32)
            comptime for k in range(4):
                var ql = u2f[W](((q >> UInt32(8 * k)) & 0xF) | (((qhw >> (sh + UInt32(8 * k))) & 1) << 4))
                acc = fma(fma(d1, ql, -m1), SIMD[F32, W](xl[k]), acc)
            comptime for k in range(4):
                var qh = u2f[W](((q >> UInt32(8 * k + 4)) & 0xF) | (((qhw >> (sh + UInt32(8 * k + 1))) & 1) << 4))
                acc = fma(fma(d2, qh, -m2), SIMD[F32, W](xh[k]), acc)
        else:
            var xb = xp + blk * 256 + lane0
            comptime for k in range(4):
                var ql = u2f[W](((q >> UInt32(8 * k)) & 0xF) | (((qhw >> (sh + UInt32(8 * k))) & 1) << 4))
                acc = fma(fma(d1, ql, -m1), xb.load[width=W](k * 32), acc)
            comptime for k in range(4):
                var qh = u2f[W](((q >> UInt32(8 * k + 4)) & 0xF) | (((qhw >> (sh + UInt32(8 * k + 1))) & 1) << 4))
                acc = fma(fma(d2, qh, -m2), xb.load[width=W]((k + 4) * 32), acc)
    return acc


def permute_x_q4k(x: Pointer[Float32, _], xp: Pointer[mut=True, Float32, _], n: Int):
    """xp[blk*256 + k*32 + l] = x[blk*256 + 64*(l/8) + 4*(l%8) + (k%4) + 32*(k/4)]. A pure permutation."""
    for blk in range(n // 256):
        for k in range(8):
            for l in range(32):
                xp[blk * 256 + k * 32 + l] = x[blk * 256 + 64 * (l // 8) + 4 * (l % 8) + (k % 4) + 32 * (k // 4)]


# ---------------------------------------------------------------- Q5_1 / Q8_0 row dot (down)
# Q5_1 block: d f16 | m f16 | qh u32 | qs[16] = 24 bytes / 32 values; value l = d * (nibble | bit_l << 4) + m
# Q8_0 block: d f16 | qs int8[32]             = 34 bytes / 32 values; value l = d * qs[l]
# Lane l owns element l of every block, so x (here: h) is read in natural order.

comptime Q51_BYTES = 24
comptime Q80_BYTES = 34


@inline(.always)
def spread4(b: UInt32) -> UInt32:
    """Bits 0..3 of b -> bit 4 of bytes 0..3 (one multiply instead of four shift/mask pairs)."""
    return ((b * 0x00204081) & 0x01010101) << 4


@inline(.always)
def q51_block(b: Pointer[UInt8, _], p: Int, hl: SIMD[F32, 4], hh: SIMD[F32, 4], acc: Float32) -> Float32:
    """One lane's share (qs bytes 4p..4p+3 -> 8 values) of one Q5_1 block. THE arithmetic, used by CPU and GPU:
    s = sum q_k h_k (low k = 0..3, then high), hs = sum h_k (same order), return fma(m, hs, fma(d, s, acc))."""
    var hw = b.bitcast[UInt32]()
    var h0 = hw[0]
    var qh = hw[1]
    var qs = hw[2 + p]
    var d = bitcast[DType.float16, 1](UInt16(h0 & 0xFFFF)).cast[F32]()
    var m = bitcast[DType.float16, 1](UInt16(h0 >> 16)).cast[F32]()
    var lo = (qs & 0x0F0F0F0F) | spread4((qh >> UInt32(4 * p)) & 0xF)
    var hi = ((qs >> 4) & 0x0F0F0F0F) | spread4((qh >> UInt32(16 + 4 * p)) & 0xF)
    var sq: Float32 = 0
    var sh: Float32 = 0
    comptime for k in range(4):
        sq = fma(u2f[1]((lo >> UInt32(8 * k)) & 0xFF)[0], hl[k], sq)
        sh = sh + hl[k]
    comptime for k in range(4):
        sq = fma(u2f[1]((hi >> UInt32(8 * k)) & 0xFF)[0], hh[k], sq)
        sh = sh + hh[k]
    return fma(m, sh, fma(d, sq, acc))


@inline(.always)
def row_dot_q51[W: Int, NBLK: Int](row: Pointer[UInt8, _], h: Pointer[Float32, _], lane0: Int) -> SIMD[F32, W]:
    """Lane l owns qs bytes 4p..4p+3 (p = l % 4) of block 8*it + l/4, it = 0, 1, 2 (in that order); lanes past the
    last block add nothing. h read from memory (the CPU path; the GPU keeps h in registers, see gpu_moe.mojo)."""
    var acc = SIMD[F32, W](0)
    comptime ITERS = (NBLK + 7) // 8
    comptime if W == 32:
        # here h is the arranged copy from arrange_h_q51
        comptime for it in range(ITERS):
            acc = rebind[SIMD[F32, W]](q51_iter_vec[NBLK, it](row, h, rebind[SIMD[F32, 32]](acc)))
        return acc
    comptime for it in range(ITERS):
        comptime for i in range(W):
            var l = lane0 + i
            var blk = 8 * it + l // 4
            if blk < NBLK:
                var p = l % 4
                acc[i] = q51_block(row + blk * Q51_BYTES, p, h.load[width=4](blk * 32 + 4 * p),
                                   h.load[width=4](blk * 32 + 16 + 4 * p), acc[i])
    return acc


@inline(.always)
def q51_iter_vec[NBLK: Int, it: Int](row: Pointer[UInt8, _], hv: Pointer[Float32, _], acc: SIMD[F32, 32]) -> SIMD[F32, 32]:
    """CPU: q51_block for all 32 lanes at once (lane l: block 8*it + l/4, p = l % 4). Elementwise the same operations
    in the same order, so each lane's result equals the GPU lane's bit for bit."""
    # 8 blocks per step: one 16-byte qs load and one 8-byte header load per block, spread to its 4 lanes by shuffles
    var h0 = SIMD[DType.uint32, 32](0)  # per-block words in lanes 0..7, spread to 4 lanes each below
    var qh = SIMD[DType.uint32, 32](0)
    var q0 = SIMD[DType.uint32, 4](0)
    var q1 = SIMD[DType.uint32, 4](0)
    var q2 = SIMD[DType.uint32, 4](0)
    var q3 = SIMD[DType.uint32, 4](0)
    var q4 = SIMD[DType.uint32, 4](0)
    var q5 = SIMD[DType.uint32, 4](0)
    var q6 = SIMD[DType.uint32, 4](0)
    var q7 = SIMD[DType.uint32, 4](0)
    comptime for j in range(8):
        comptime blk = min(8 * it + j, NBLK - 1)  # clamp: lanes past the end read a valid block, result discarded
        var hw = (row + blk * Q51_BYTES).bitcast[UInt32]()
        h0[j] = hw[0]
        qh[j] = hw[1]
        var qv = (hw + 2).load[width=4](0)
        comptime if j == 0: q0 = qv
        elif j == 1: q1 = qv
        elif j == 2: q2 = qv
        elif j == 3: q3 = qv
        elif j == 4: q4 = qv
        elif j == 5: q5 = qv
        elif j == 6: q6 = qv
        else: q7 = qv
    var qs = q0.join(q1).join(q2.join(q3)).join(q4.join(q5).join(q6.join(q7)))
    var h0v = h0.shuffle[0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7]()
    var qhv = qh.shuffle[0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7]()
    var shl = SIMD[DType.uint32, 32](0, 4, 8, 12, 0, 4, 8, 12, 0, 4, 8, 12, 0, 4, 8, 12, 0, 4, 8, 12, 0, 4, 8, 12, 0, 4, 8, 12, 0, 4, 8, 12)
    var d = bitcast[DType.float16, 32]((h0v & 0xFFFF).cast[DType.uint16]()).cast[F32]()
    var m = bitcast[DType.float16, 32]((h0v >> 16).cast[DType.uint16]()).cast[F32]()
    var lo = (qs & 0x0F0F0F0F) | ((((qhv >> shl) & 0xF) * 0x00204081) & 0x01010101) << 4
    var hi = ((qs >> 4) & 0x0F0F0F0F) | ((((qhv >> (shl + 16)) & 0xF) * 0x00204081) & 0x01010101) << 4
    var sq = SIMD[F32, 32](0)
    var sh = SIMD[F32, 32](0)
    comptime for k in range(4):
        var hk = hv.load[width=32]((it * 8 + k) * 32)
        sq = fma(u2f[32]((lo >> UInt32(8 * k)) & 0xFF), hk, sq)
        sh = sh + hk
    comptime for k in range(4):
        var hk = hv.load[width=32]((it * 8 + 4 + k) * 32)
        sq = fma(u2f[32]((hi >> UInt32(8 * k)) & 0xFF), hk, sq)
        sh = sh + hk
    var new = fma(m, sh, fma(d, sq, acc))
    comptime if 8 * it + 8 <= NBLK:
        return new
    var keep = SIMD[DType.bool, 32](fill=False)
    comptime for l in range(32):
        keep[l] = 8 * it + l // 4 < NBLK
    return keep.select(new, acc)


def arrange_h_q51(h: Pointer[Float32, _], hv: Pointer[mut=True, Float32, _], nblk: Int):
    """CPU: lane-order copy of one expert's h for q51_iter_vec: hv[(it*8 + j)*32 + l] = the value lane l multiplies at
    step j (j < 4: element 4p+j, else 16+4p+(j-4); block 8*it + l/4, clamped). A pure copy, done once per expert."""
    for it in range((nblk + 7) // 8):
        for l in range(32):
            var blk = min(8 * it + l // 4, nblk - 1)
            var pp = l % 4
            for j in range(4):
                hv[(it * 8 + j) * 32 + l] = h[blk * 32 + 4 * pp + j]
                hv[(it * 8 + 4 + j) * 32 + l] = h[blk * 32 + 16 + 4 * pp + j]


@inline(.always)
def row_dot_q80[W: Int, NBLK: Int](row: Pointer[UInt8, _], h: Pointer[Float32, _], lane0: Int) -> SIMD[F32, W]:
    var acc = SIMD[F32, W](0)
    comptime for blk in range(NBLK):
        var b = row + blk * Q80_BYTES
        var d = SIMD[F32, W](f16_at(b, 0))
        var q = SIMD[F32, W](0)
        comptime for i in range(W):
            q[i] = bitcast[DType.int8, 1](b[2 + lane0 + i]).cast[F32]()
        acc = fma(d * q, (h + blk * 32 + lane0).load[width=W](), acc)
    return acc
