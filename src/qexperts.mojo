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

comptime F32 = DType.float32

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
def scale_min_k4(s: Pointer[UInt8, _], j: Int) -> Tuple[UInt8, UInt8]:
    if j < 4:
        return (s[j] & 63, s[j + 4] & 63)
    return ((s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4), (s[j + 4] >> 4) | ((s[j] >> 6) << 4))


@inline(.always)
def row_dot_q4k[W: Int](row: Pointer[UInt8, _], nblk: Int, xp: Pointer[Float32, _], lane0: Int) -> SIMD[F32, W]:
    var acc = SIMD[F32, W](0)
    for blk in range(nblk):
        var b = row + blk * Q4K_BYTES
        var d = f16_at(b, 0)
        var dmin = f16_at(b, 2)
        var sc = b + 4
        # per-lane scale / min for its low (2j) and high (2j+1) sub-block
        var d1 = SIMD[F32, W](0)
        var m1 = SIMD[F32, W](0)
        var d2 = SIMD[F32, W](0)
        var m2 = SIMD[F32, W](0)
        comptime for i in range(W):
            var j = (lane0 + i) // 8
            var lo = scale_min_k4(sc, 2 * j)
            var hi = scale_min_k4(sc, 2 * j + 1)
            d1[i] = d * Float32(lo[0])
            m1[i] = dmin * Float32(lo[1])
            d2[i] = d * Float32(hi[0])
            m2[i] = dmin * Float32(hi[1])
        var q = u32_lanes[W](b + 16, lane0)
        var xb = xp + blk * 256 + lane0
        comptime for k in range(4):
            var ql = ((q >> UInt32(8 * k)) & 0xF).cast[F32]()
            acc = fma(fma(d1, ql, -m1), xb.load[width=W](k * 32), acc)
        comptime for k in range(4):
            var qh = ((q >> UInt32(8 * k + 4)) & 0xF).cast[F32]()
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
def row_dot_q51[W: Int](row: Pointer[UInt8, _], nblk: Int, h: Pointer[Float32, _], lane0: Int) -> SIMD[F32, W]:
    var acc = SIMD[F32, W](0)
    var lanes = lane_iota[W](lane0)
    for blk in range(nblk):
        var b = row + blk * Q51_BYTES
        var d = SIMD[F32, W](f16_at(b, 0))
        var m = SIMD[F32, W](f16_at(b, 2))
        var qh = UInt32(b[4]) | (UInt32(b[5]) << 8) | (UInt32(b[6]) << 16) | (UInt32(b[7]) << 24)
        var nib = SIMD[DType.uint32, W](0)
        comptime for i in range(W):
            var l = lane0 + i
            var byte = UInt32(b[8 + (l % 16)])
            nib[i] = (byte >> UInt32(4 * (l // 16))) & 0xF
        var hb = ((SIMD[DType.uint32, W](qh) >> lanes.cast[DType.uint32]()) & 1) << 4
        var v = fma(d, (nib | hb).cast[F32](), m)
        acc = fma(v, (h + blk * 32 + lane0).load[width=W](), acc)
    return acc


@inline(.always)
def row_dot_q80[W: Int](row: Pointer[UInt8, _], nblk: Int, h: Pointer[Float32, _], lane0: Int) -> SIMD[F32, W]:
    var acc = SIMD[F32, W](0)
    for blk in range(nblk):
        var b = row + blk * Q80_BYTES
        var d = SIMD[F32, W](f16_at(b, 0))
        var q = SIMD[F32, W](0)
        comptime for i in range(W):
            q[i] = bitcast[DType.int8, 1](b[2 + lane0 + i]).cast[F32]()
        acc = fma(d * q, (h + blk * 32 + lane0).load[width=W](), acc)
    return acc
