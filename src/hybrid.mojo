# Hybrid expert layer: routing never leaves the GPU, hot experts run on the GPU, cold experts on the CPU, and the two
# meet through a mailbox in pinned host memory (GPU kernels read and write it directly; no host sync per layer).
#
# Per token, enqueued back to back on one stream:
#   router_kernel   logits = Wr x                                  (GPU)
#   topk_kernel     top-10, sorted ids, softmax weights, hot/cold split; posts cold ids + x to the mailbox, then seq
#   gate_up_hot     hot experts only                               (GPU)      ||  CPU workers see seq, compute the
#   down_hot        hot experts only, per-expert sums s_e          (GPU)      ||  cold experts' s_e into the mailbox,
#   combine_kernel  waits for done == seq (only if any cold), y = sum_e w_e s_e in id order   then write done = seq
# The combine order and every s_e are bit-identical whichever device computed them (qexperts.mojo), so the hybrid
# output equals the all-GPU output bit for bit.
from std.math import fma
from max.gpu import block_idx, block_dim, thread_idx
from max.gpu.primitives.warp import shuffle_xor, shuffle_idx
from max.gpu.sync import barrier
from max.gpu.intrinsics import threadfence, Scope
from std.memory import unsafe_stack_allocation, bitcast
from qexperts import exp_det, Q51_BYTES, Q4K_BYTES, q51_block, row_dot_q80
from gpu_moe import warp_tree, q4k_block_gpu, silu_mul, D_MODEL, D_FF, Q4K_ROW, Q51_ROW, Q80_ROW, WARPS_PER_BLOCK, TPB

comptime N_EXPERTS = 512
comptime TOPK = 10

# mailbox layout (in Int32 / Float32 words)
comptime MB_SEQ = 0        # request sequence number, written by the GPU
comptime MB_DONE = 16      # done sequence number, written by the CPU
comptime MB_NCOLD = 32
comptime MB_ERR = 48
comptime MB_LAYER = 8      # layer + 1 when the poster is the in-stream path (0: the host recorded the layer)
comptime MB_RING = 16      # mailbox slots per GPU ring
comptime MB_SLOT = 64      # [TOPK] slot index of each cold expert
comptime MB_ID = 80        # [TOPK] expert id of each cold expert
comptime MB_X = 256        # [D_MODEL] the token's activation
comptime MB_S = MB_X + D_MODEL       # [TOPK][D_MODEL] per-expert sums from the CPU, at their slot
comptime MB_WORDS = MB_S + TOPK * D_MODEL


def router_kernel(wr: Pointer[Float32, ImmutAnyOrigin], x: Pointer[Float32, ImmutAnyOrigin], logits: Pointer[Float32, MutAnyOrigin]):
    """One warp per expert row of the F32 router: logits[e] = Wr[e] . x."""
    var e = Int(block_idx.x) * WARPS_PER_BLOCK + Int(thread_idx.x) // 32
    var lane = Int(thread_idx.x) % 32
    if e >= N_EXPERTS:
        return
    var row = wr + e * D_MODEL
    var acc = SIMD[DType.float32, 4](0)
    comptime for j in range(D_MODEL // 128):
        acc = fma(row.load[width=4](j * 128 + lane * 4), x.load[width=4](j * 128 + lane * 4), acc)
    var v = warp_tree(acc.reduce_add())
    if lane == 0:
        logits[e] = v


def topk_kernel(
    logits: Pointer[Float32, ImmutAnyOrigin], res: Pointer[Int32, ImmutAnyOrigin], x: Pointer[Float32, ImmutAnyOrigin],
    ids: Pointer[Int32, MutAnyOrigin], w: Pointer[Float32, MutAnyOrigin], hot: Pointer[Int32, MutAnyOrigin],
    mb: Pointer[Int32, MutAnyOrigin], seq: Int32,
):
    """One block of 512 threads. Top-10 by 10 rounds of block argmax (ties -> lower id), ids sorted ascending,
    w = softmax restricted to the top-10 (= llama.cpp's softmax + renormalise), then post the cold part to the mailbox."""
    var t = Int(thread_idx.x)
    var lane = t % 32
    var wid = t // 32
    var wv = unsafe_stack_allocation[16, Float32, address_space=.SHARED]()
    var wi = unsafe_stack_allocation[16, Int32, address_space=.SHARED]()
    var sel = unsafe_stack_allocation[TOPK, Int32, address_space=.SHARED]()
    var v = logits[t]
    for r in range(TOPK):
        var bv = v
        var bi = Int32(t)
        comptime for off in [16, 8, 4, 2, 1]:
            var ov = shuffle_xor(bv, UInt32(off))
            var oi = shuffle_xor(bi, UInt32(off))
            if ov > bv or (ov == bv and oi < bi):
                bv = ov
                bi = oi
        if lane == 0:
            wv[unsafe_offset=wid] = bv
            wi[unsafe_offset=wid] = bi
        barrier()
        if t == 0:
            var mv = wv[unsafe_offset=0][0]
            var mi = wi[unsafe_offset=0][0]
            for j in range(1, 16):
                var cv = wv[unsafe_offset=j][0]
                var ci = wi[unsafe_offset=j][0]
                if cv > mv or (cv == mv and ci < mi):
                    mv = cv
                    mi = ci
            sel[unsafe_offset=r] = mi
        barrier()
        if t == Int(sel[unsafe_offset=r][0]):
            v = Float32(-3.0e38)
        barrier()
    if t == 0:
        # sort ascending (insertion sort on 10)
        for i in range(1, TOPK):
            var key = sel[unsafe_offset=i][0]
            var j = i - 1
            while j >= 0 and sel[unsafe_offset=j][0] > key:
                sel[unsafe_offset=j + 1] = sel[unsafe_offset=j][0]
                j -= 1
            sel[unsafe_offset=j + 1] = key
        var lmax = Float32(-3.0e38)
        for i in range(TOPK):
            lmax = max(lmax, logits[Int(sel[unsafe_offset=i][0])])
        var tot: Float32 = 0
        for i in range(TOPK):
            tot += exp_det[1](SIMD[DType.float32, 1](logits[Int(sel[unsafe_offset=i][0])] - lmax))[0]
        var ncold = 0
        for i in range(TOPK):
            var id = sel[unsafe_offset=i][0]
            ids[i] = id
            w[i] = exp_det[1](SIMD[DType.float32, 1](logits[Int(id)] - lmax))[0] / tot
            var hh = res[Int(id)]
            hot[i] = hh
            if hh == 0:
                mb.unsafe_store[volatile=True](MB_SLOT + ncold, Int32(i))
                mb.unsafe_store[volatile=True](MB_ID + ncold, id)
                ncold += 1
        mb.unsafe_store[volatile=True](MB_NCOLD, Int32(ncold))
        hot[TOPK] = Int32(ncold)
    # the activation, for the CPU
    var mbf = mb.bitcast[Float32]()
    for i in range(t, D_MODEL, 512):
        mbf.unsafe_store[volatile=True](MB_X + i, x[i])
    threadfence[Scope.SYSTEM]()
    barrier()
    if t == 0:
        threadfence[Scope.SYSTEM]()
        mb.unsafe_store[volatile=True](MB_SEQ, seq)


def gate_up_hot(
    gate: Pointer[UInt8, ImmutAnyOrigin], up: Pointer[UInt8, ImmutAnyOrigin],
    ids: Pointer[Int32, ImmutAnyOrigin], hot: Pointer[Int32, ImmutAnyOrigin], expert_bytes_: Int64,
    x: Pointer[Float32, ImmutAnyOrigin], h: Pointer[Float32, MutAnyOrigin],
):
    """gate_up_kernel_reg for the hot slots only (one warp per (slot, row); cold slots return at once)."""
    var expert_bytes = Int(expert_bytes_)
    var warp = Int(block_idx.x) * WARPS_PER_BLOCK + Int(thread_idx.x) // 32
    var lane = Int(thread_idx.x) % 32
    if warp >= TOPK * D_FF:
        return
    var e = warp // D_FF
    if hot[e] == 0:
        return
    var r = warp % D_FF
    var xbase = 64 * (lane // 8) + 4 * (lane % 8)
    var eoff = Int(ids[e]) * expert_bytes + r * Q4K_ROW
    var ga: Float32 = 0
    var ua: Float32 = 0
    comptime for blk in range(D_MODEL // 256):
        var xl = x.load[width=4](blk * 256 + xbase)
        var xh = x.load[width=4](blk * 256 + xbase + 32)
        ga = q4k_block_gpu(gate + eoff + blk * Q4K_BYTES, lane, xl, xh, ga)
        ua = q4k_block_gpu(up + eoff + blk * Q4K_BYTES, lane, xl, xh, ua)
    var g = warp_tree(ga)
    var u = warp_tree(ua)
    if lane == 0:
        h[warp] = silu_mul[1](g, u)[0]


comptime DROWS = 8


def down_hot(
    down: Pointer[UInt8, ImmutAnyOrigin], down_q80: Int64,
    ids: Pointer[Int32, ImmutAnyOrigin], hot: Pointer[Int32, ImmutAnyOrigin], expert_bytes_: Int64,
    h: Pointer[Float32, ImmutAnyOrigin], s: Pointer[Float32, MutAnyOrigin],
):
    """Per-expert down sums s[e, i] for the hot slots (block = TOPK warps x DROWS rows, h kept in registers)."""
    var expert_bytes = Int(expert_bytes_)
    var e = Int(thread_idx.x) // 32
    var lane = Int(thread_idx.x) % 32
    if hot[e] == 0:
        return
    var i0 = Int(block_idx.x) * DROWS
    var base = down + Int(ids[e]) * expert_bytes
    var he = h + e * D_FF
    if down_q80 == 1:
        comptime for r in range(DROWS):
            var v = warp_tree(row_dot_q80[1, D_FF // 32](base + (i0 + r) * Q80_ROW, he, lane)[0])
            if lane == 0:
                s[e * D_MODEL + i0 + r] = v
        return
    var p = lane % 4
    var hl0 = he.load[width=4]((lane // 4) * 32 + 4 * p)
    var hh0 = he.load[width=4]((lane // 4) * 32 + 16 + 4 * p)
    var hl1 = he.load[width=4](256 + (lane // 4) * 32 + 4 * p)
    var hh1 = he.load[width=4](256 + (lane // 4) * 32 + 16 + 4 * p)
    var hl2 = SIMD[DType.float32, 4](0)
    var hh2 = SIMD[DType.float32, 4](0)
    if lane < 16:
        hl2 = he.load[width=4](512 + (lane // 4) * 32 + 4 * p)
        hh2 = he.load[width=4](512 + (lane // 4) * 32 + 16 + 4 * p)
    comptime for r in range(DROWS):
        var row = base + (i0 + r) * Q51_ROW
        var acc: Float32 = 0
        acc = q51_block(row + (lane // 4) * Q51_BYTES, p, hl0, hh0, acc)
        acc = q51_block(row + (8 + lane // 4) * Q51_BYTES, p, hl1, hh1, acc)
        if lane < 16:
            acc = q51_block(row + (16 + lane // 4) * Q51_BYTES, p, hl2, hh2, acc)
        var v = warp_tree(acc)
        if lane == 0:
            s[e * D_MODEL + i0 + r] = v


def combine_kernel(
    s: Pointer[Float32, ImmutAnyOrigin], mb: Pointer[Int32, MutAnyOrigin], w: Pointer[Float32, ImmutAnyOrigin],
    hot: Pointer[Int32, ImmutAnyOrigin], y: Pointer[Float32, MutAnyOrigin], seq: Int32,
):
    """Waits for the CPU only if this token has cold experts, then y[i] = sum_e fma(w_e, s_e[i]) in slot order."""
    var t = Int(thread_idx.x)
    var i = Int(block_idx.x) * Int(block_dim.x) + t
    if hot[TOPK] > 0:
        if t == 0:
            var spins = 0
            while mb.unsafe_load[volatile=True](MB_DONE)[0] != seq:
                spins += 1
                if spins > 400_000_000:
                    mb.unsafe_store[volatile=True](MB_ERR, Int32(-1))
                    break
        barrier()
    if i >= D_MODEL:
        return
    var mbf = mb.bitcast[Float32]()
    var o: Float32 = 0
    for e in range(TOPK):
        var v: Float32
        if hot[e] != 0:
            v = s[e * D_MODEL + i]
        else:
            v = mbf.unsafe_load[volatile=True](MB_S + e * D_MODEL + i)[0]
        o = fma(w[e], v, o)
    y[i] = o
