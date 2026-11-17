# Batched (multi-token) expert math for the CPU: dequantise a block once, apply it to up to NT tokens.
# Each token's arithmetic is exactly the one-token path's (row_dot_q4k / q51_block): the dequantised weight w is
# computed by the same fma, then fma(w, x_t, acc_t) runs in the same order. So a batch gives the same bits as
# running its tokens one at a time, on the CPU or the GPU.
from std.math import fma
from std.memory import bitcast
from std.collections import Array
from qexperts import F32, Q4K_BYTES, Q51_BYTES, scale_min_k4, u2f, tree32, spread4


@inline(.always)
def q4k_rows_multi[NT: Int, NBLK: Int](row: Pointer[UInt8, _], xps: Pointer[Float32, _], offs: Pointer[Int, _],
                                     res: Pointer[mut=True, Float32, _]):
    """res[t] = row . x_t for t < NT, x_t = the lane-permuted activation at xps + offs[t] (permute_x_q4k)."""
    var acc = Array[SIMD[F32, 32], NT](fill=SIMD[F32, 32](0))
    comptime for blk in range(NBLK):
        var b = row + blk * Q4K_BYTES
        var hv = b.bitcast[UInt32]().load[width=4]()
        var d = bitcast[DType.float16, 1](UInt16(hv[0] & 0xFFFF)).cast[F32]()
        var dmin = bitcast[DType.float16, 1](UInt16(hv[0] >> 16)).cast[F32]()
        var dd = SIMD[F32, 32](0)
        var mm = SIMD[F32, 32](0)
        comptime for t in range(8):
            var sm = scale_min_k4(hv[1], hv[2], hv[3], t)
            dd[t] = d * Float32(sm[0])
            mm[t] = dmin * Float32(sm[1])
        var d1 = dd.shuffle[0,0,0,0,0,0,0,0,2,2,2,2,2,2,2,2,4,4,4,4,4,4,4,4,6,6,6,6,6,6,6,6]()
        var m1 = mm.shuffle[0,0,0,0,0,0,0,0,2,2,2,2,2,2,2,2,4,4,4,4,4,4,4,4,6,6,6,6,6,6,6,6]()
        var d2 = dd.shuffle[1,1,1,1,1,1,1,1,3,3,3,3,3,3,3,3,5,5,5,5,5,5,5,5,7,7,7,7,7,7,7,7]()
        var m2 = mm.shuffle[1,1,1,1,1,1,1,1,3,3,3,3,3,3,3,3,5,5,5,5,5,5,5,5,7,7,7,7,7,7,7,7]()
        var q = (b + 16).bitcast[UInt32]().load[width=32](0)
        comptime for k in range(4):
            var w = fma(d1, u2f[32]((q >> UInt32(8 * k)) & 0xF), -m1)
            comptime for t in range(NT):
                acc[t] = fma(w, (xps + offs[t] + blk * 256).load[width=32](k * 32), acc[t])
        comptime for k in range(4):
            var w = fma(d2, u2f[32]((q >> UInt32(8 * k + 4)) & 0xF), -m2)
            comptime for t in range(NT):
                acc[t] = fma(w, (xps + offs[t] + blk * 256).load[width=32]((k + 4) * 32), acc[t])
    comptime for t in range(NT):
        res[t] = tree32(acc[t])


@inline(.always)
def q51_rows_multi[NT: Int, NBLK: Int](row: Pointer[UInt8, _], hvs: Pointer[Float32, _], hstride: Int,
                                     res: Pointer[mut=True, Float32, _]):
    """res[t] = row . h_t, h_t arranged by arrange_h_q51 at hvs + t * hstride. Same per-lane order as q51_iter_vec."""
    var acc = Array[SIMD[F32, 32], NT](fill=SIMD[F32, 32](0))
    comptime ITERS = (NBLK + 7) // 8
    comptime for it in range(ITERS):
        var h0 = SIMD[DType.uint32, 32](0)
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
            comptime blk = min(8 * it + j, NBLK - 1)
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
        var qf = Array[SIMD[F32, 32], 8](fill=SIMD[F32, 32](0))
        comptime for k in range(4):
            qf[k] = u2f[32]((lo >> UInt32(8 * k)) & 0xFF)
            qf[4 + k] = u2f[32]((hi >> UInt32(8 * k)) & 0xFF)
        comptime for t in range(NT):
            var hv = hvs + t * hstride
            var sq = SIMD[F32, 32](0)
            var sh = SIMD[F32, 32](0)
            comptime for k in range(8):
                var hk = hv.load[width=32]((it * 8 + k) * 32)
                sq = fma(qf[k], hk, sq)
                sh = sh + hk
            var new = fma(m, sh, fma(d, sq, acc[t]))
            comptime if 8 * it + 8 <= NBLK:
                acc[t] = new
            else:
                var keep = SIMD[DType.bool, 32](fill=False)
                comptime for l in range(32):
                    keep[l] = 8 * it + l // 4 < NBLK
                acc[t] = keep.select(new, acc[t])
    comptime for t in range(NT):
        res[t] = tree32(acc[t])
