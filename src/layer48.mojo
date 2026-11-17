# M4: the expert layer for all 48 layers. Hot experts live in a packed per-layer VRAM pool in profile order
# ("profile-specialised layout": slot s of layer l holds the s-th pinned expert, so the hot path never touches a
# 512-entry tensor), cold experts are read by the CPU straight from the memory-mapped GGUF.
# Kernels per layer (one stream per GPU, no host sync): router -> topk (here: real routing from a trace) ->
# gate/up (hot slots) -> down (hot slots) -> combine (+ residual into the next layer's input).
# GPU 0 holds layers 0-23, GPU 1 layers 24-47; the token's activation crosses between them through pinned host
# memory (post_x / wait_x), the same mailbox idea as the CPU hand-off.
from std.math import fma
from max.gpu import block_idx, block_dim, thread_idx
from max.gpu.sync import barrier
from max.gpu.primitives.warp import shuffle_xor
from max.gpu.intrinsics import threadfence, Scope
from std.memory import unsafe_stack_allocation
from qexperts import exp_det, Q51_BYTES, Q4K_BYTES, Q5K_BYTES, q51_block, row_dot_q80, row_dot_q5k
from gpu_moe import warp_tree, q4k_block_gpu, silu_mul, D_MODEL, D_FF, Q80_ROW, Q51_ROW, WARPS_PER_BLOCK, TPB
from hybrid import N_EXPERTS, TOPK, MB_SEQ, MB_DONE, MB_NCOLD, MB_ERR, MB_SLOT, MB_ID, MB_X, MB_S, MB_WORDS, DROWS


def topk_trace_kernel(
    logits: Pointer[Float32, ImmutAnyOrigin], tids: Pointer[Int32, ImmutAnyOrigin], res: Pointer[Int32, ImmutAnyOrigin],
    x: Pointer[Float32, ImmutAnyOrigin], w: Pointer[Float32, MutAnyOrigin], slots: Pointer[Int32, MutAnyOrigin],
    mb: Pointer[Int32, MutAnyOrigin], seq: Int32,
):
    """Like topk_kernel, but the 10 experts come from a real routing trace (sorted); weights are the softmax of the
    router's logits over them. slots[i] = VRAM slot of expert i or -1 (cold); slots[TOPK] = number of cold experts."""
    var t = Int(thread_idx.x)
    if t == 0:
        var lmax = Float32(-3.0e38)
        for i in range(TOPK):
            lmax = max(lmax, logits[Int(tids[i])])
        var tot: Float32 = 0
        for i in range(TOPK):
            tot += exp_det[1](SIMD[DType.float32, 1](logits[Int(tids[i])] - lmax))[0]
        var ncold = 0
        for i in range(TOPK):
            var id = tids[i]
            w[i] = exp_det[1](SIMD[DType.float32, 1](logits[Int(id)] - lmax))[0] / tot
            var sl = res[Int(id)]
            slots[i] = sl
            if sl < 0:
                mb.unsafe_store[volatile=True](MB_SLOT + ncold, Int32(i))
                mb.unsafe_store[volatile=True](MB_ID + ncold, id)
                ncold += 1
        mb.unsafe_store[volatile=True](MB_NCOLD, Int32(ncold))
        slots[TOPK] = Int32(ncold)
    barrier()
    # the CPU needs x only if something is cold
    if slots[TOPK] > 0:
        var mbf = mb.bitcast[Float32]()
        for i in range(t, D_MODEL, Int(block_dim.x)):
            mbf.unsafe_store[volatile=True](MB_X + i, x[i])
    threadfence[Scope.SYSTEM]()
    barrier()
    if t == 0:
        threadfence[Scope.SYSTEM]()
        mb.unsafe_store[volatile=True](MB_SEQ, seq)


def gate_up_slots[GT: Int](
    gate: Pointer[UInt8, ImmutAnyOrigin], up: Pointer[UInt8, ImmutAnyOrigin],
    slots: Pointer[Int32, ImmutAnyOrigin], expert_bytes_: Int64,
    x: Pointer[Float32, ImmutAnyOrigin], h: Pointer[Float32, MutAnyOrigin],
):
    """GT = 0: Q4_K, GT = 1: Q5_K. One warp per (hot slot, row)."""
    var expert_bytes = Int(expert_bytes_)
    var warp = Int(block_idx.x) * WARPS_PER_BLOCK + Int(thread_idx.x) // 32
    var lane = Int(thread_idx.x) % 32
    if warp >= TOPK * D_FF:
        return
    var e = warp // D_FF
    var sl = Int(slots[e])
    if sl < 0:
        return
    var r = warp % D_FF
    comptime ROW = Q4K_BYTES * (D_MODEL // 256) if GT == 0 else Q5K_BYTES * (D_MODEL // 256)
    var off = sl * expert_bytes + r * ROW
    var g: Float32
    var u: Float32
    comptime if GT == 0:
        var xbase = 64 * (lane // 8) + 4 * (lane % 8)
        var ga: Float32 = 0
        var ua: Float32 = 0
        comptime for blk in range(D_MODEL // 256):
            var xl = x.load[width=4](blk * 256 + xbase)
            var xh = x.load[width=4](blk * 256 + xbase + 32)
            ga = q4k_block_gpu(gate + off + blk * Q4K_BYTES, lane, xl, xh, ga)
            ua = q4k_block_gpu(up + off + blk * Q4K_BYTES, lane, xl, xh, ua)
        g = warp_tree(ga)
        u = warp_tree(ua)
    else:
        g = warp_tree(row_dot_q5k[1, D_MODEL // 256](gate + off, x, lane)[0])
        u = warp_tree(row_dot_q5k[1, D_MODEL // 256](up + off, x, lane)[0])
    if lane == 0:
        h[warp] = silu_mul[1](g, u)[0]


def down_slots(
    down: Pointer[UInt8, ImmutAnyOrigin], down_q80: Int64,
    slots: Pointer[Int32, ImmutAnyOrigin], expert_bytes_: Int64,
    h: Pointer[Float32, ImmutAnyOrigin], s: Pointer[Float32, MutAnyOrigin],
):
    var expert_bytes = Int(expert_bytes_)
    var e = Int(thread_idx.x) // 32
    var lane = Int(thread_idx.x) % 32
    var sl = Int(slots[e])
    if sl < 0:
        return
    var i0 = Int(block_idx.x) * DROWS
    var base = down + sl * expert_bytes
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


def combine_res_kernel(
    s: Pointer[Float32, ImmutAnyOrigin], mb: Pointer[Int32, MutAnyOrigin], w: Pointer[Float32, ImmutAnyOrigin],
    slots: Pointer[Int32, ImmutAnyOrigin], xin: Pointer[Float32, ImmutAnyOrigin], xout: Pointer[Float32, MutAnyOrigin],
    seq: Int32,
):
    """Waits for the CPU only if any expert is cold; y = sum_e fma(w_e, s_e) in slot order; next input = x + y."""
    var t = Int(thread_idx.x)
    var i = Int(block_idx.x) * Int(block_dim.x) + t
    if slots[TOPK] > 0:
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
        if slots[e] >= 0:
            v = s[e * D_MODEL + i]
        else:
            v = mbf.unsafe_load[volatile=True](MB_S + e * D_MODEL + i)[0]
        o = fma(w[e], v, o)
    xout[i] = xin[i] + o


def post_x_kernel(x: Pointer[Float32, ImmutAnyOrigin], hb: Pointer[Int32, MutAnyOrigin], seq: Int32):
    """GPU A -> host: the activation (words 64..) then the flag (word 0)."""
    var t = Int(thread_idx.x)
    var hf = hb.bitcast[Float32]()
    for i in range(t, D_MODEL, Int(block_dim.x)):
        hf.unsafe_store[volatile=True](64 + i, x[i])
    threadfence[Scope.SYSTEM]()
    barrier()
    if t == 0:
        threadfence[Scope.SYSTEM]()
        hb.unsafe_store[volatile=True](0, seq)


def wait_x_kernel(hb: Pointer[Int32, MutAnyOrigin], seq: Int32, x: Pointer[Float32, MutAnyOrigin]):
    """Host -> GPU B: spin until the flag says seq, then take the activation."""
    var t = Int(thread_idx.x)
    if t == 0:
        var spins = 0
        while hb.unsafe_load[volatile=True](0)[0] != seq:
            spins += 1
            if spins > 400_000_000:
                hb.unsafe_store[volatile=True](1, Int32(-1))
                break
    barrier()
    var hf = hb.bitcast[Float32]()
    for i in range(t, D_MODEL, Int(block_dim.x)):
        x[i] = hf.unsafe_load[volatile=True](64 + i)[0]


def topk_slots_kernel(
    logits: Pointer[Float32, ImmutAnyOrigin], res: Pointer[Int32, ImmutAnyOrigin], x: Pointer[Float32, ImmutAnyOrigin],
    w: Pointer[Float32, MutAnyOrigin], slots: Pointer[Int32, MutAnyOrigin], mb: Pointer[Int32, MutAnyOrigin], seq: Int32,
):
    """Real routing (M5): top-10 of the router logits by block argmax (ties -> lower id), sorted ids, weights =
    softmax over the 10 (llama.cpp: softmax, top-k, renormalise; this model has no extra weight scale).
    slots[i] = VRAM slot or -1, slots[TOPK] = number of cold experts; cold ids + x posted to the mailbox."""
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
            w[i] = exp_det[1](SIMD[DType.float32, 1](logits[Int(id)] - lmax))[0] / tot
            var sl = res[Int(id)]
            slots[i] = sl
            if sl < 0:
                mb.unsafe_store[volatile=True](MB_SLOT + ncold, Int32(i))
                mb.unsafe_store[volatile=True](MB_ID + ncold, id)
                ncold += 1
        mb.unsafe_store[volatile=True](MB_NCOLD, Int32(ncold))
        slots[TOPK] = Int32(ncold)
    barrier()
    if slots[TOPK] > 0:
        var mbf = mb.bitcast[Float32]()
        for i in range(t, D_MODEL, Int(block_dim.x)):
            mbf.unsafe_store[volatile=True](MB_X + i, x[i])
    threadfence[Scope.SYSTEM]()
    barrier()
    if t == 0:
        threadfence[Scope.SYSTEM]()
        mb.unsafe_store[volatile=True](MB_SEQ, seq)


# ================================================================ batched path (prompt chunks, speculative verify)
# Token t's arrays live at [t * stride]. The CPU gets one request per batch: a list of (expert, token, slot) cold
# picks in the batch area; it reads each cold expert once and writes results to bres[t][slot][:].
comptime BW = 16   # words per token in the per-token small arrays (ids 0..9, slots 0..10)
comptime STAGE_BASE = 1 << 20   # slot values >= this point into the staging pool (cold experts uploaded for a big batch)


def router_batch_kernel(wr: Pointer[Float32, ImmutAnyOrigin], x: Pointer[Float32, ImmutAnyOrigin], logits: Pointer[Float32, MutAnyOrigin]):
    """Blocks [t * 64, t * 64 + 64) score token t: one warp per expert, as router_kernel."""
    var t = Int(block_idx.x) // (N_EXPERTS // WARPS_PER_BLOCK)
    var e = (Int(block_idx.x) % (N_EXPERTS // WARPS_PER_BLOCK)) * WARPS_PER_BLOCK + Int(thread_idx.x) // 32
    var lane = Int(thread_idx.x) % 32
    var row = wr + e * D_MODEL
    var xt = x + t * D_MODEL
    var acc = SIMD[DType.float32, 4](0)
    comptime for j in range(D_MODEL // 128):
        acc = fma(row.load[width=4](j * 128 + lane * 4), xt.load[width=4](j * 128 + lane * 4), acc)
    var v = warp_tree(acc.reduce_add())
    if lane == 0:
        logits[t * N_EXPERTS + e] = v


def topk_batch_kernel(logits: Pointer[Float32, ImmutAnyOrigin], res: Pointer[Int32, ImmutAnyOrigin],
                      w: Pointer[Float32, MutAnyOrigin], ids: Pointer[Int32, MutAnyOrigin], slots: Pointer[Int32, MutAnyOrigin]):
    """Block t = token t (512 threads): same top-10 and weights as topk_slots_kernel, no mailbox."""
    var tok = Int(block_idx.x)
    var t = Int(thread_idx.x)
    var lane = t % 32
    var wid = t // 32
    var wv = unsafe_stack_allocation[16, Float32, address_space=.SHARED]()
    var wi = unsafe_stack_allocation[16, Int32, address_space=.SHARED]()
    var sel = unsafe_stack_allocation[TOPK, Int32, address_space=.SHARED]()
    var lg = logits + tok * N_EXPERTS
    var v = lg[t]
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
        for i in range(1, TOPK):
            var key = sel[unsafe_offset=i][0]
            var j = i - 1
            while j >= 0 and sel[unsafe_offset=j][0] > key:
                sel[unsafe_offset=j + 1] = sel[unsafe_offset=j][0]
                j -= 1
            sel[unsafe_offset=j + 1] = key
        var lmax = Float32(-3.0e38)
        for i in range(TOPK):
            lmax = max(lmax, lg[Int(sel[unsafe_offset=i][0])])
        var tot: Float32 = 0
        for i in range(TOPK):
            tot += exp_det[1](SIMD[DType.float32, 1](lg[Int(sel[unsafe_offset=i][0])] - lmax))[0]
        var ncold = 0
        for i in range(TOPK):
            var id = sel[unsafe_offset=i][0]
            w[tok * BW + i] = exp_det[1](SIMD[DType.float32, 1](lg[Int(id)] - lmax))[0] / tot
            ids[tok * BW + i] = id
            var sl = res[Int(id)]
            slots[tok * BW + i] = sl
            if sl < 0:
                ncold += 1
        slots[tok * BW + TOPK] = Int32(ncold)


def collect_cold_kernel(ids: Pointer[Int32, ImmutAnyOrigin], slots: Pointer[Int32, ImmutAnyOrigin], ntok: Int64,
                        barea: Pointer[Int32, MutAnyOrigin], mb: Pointer[Int32, MutAnyOrigin], seq: Int32,
                        total: Pointer[Int32, MutAnyOrigin], post: Int64):
    """One thread: list every cold pick (expert, token, slot) into the pinned batch area, then post the request.
    barea[0] = count, then triples. Posts NCOLD = -1 (batch request) or 0 (nothing cold) and the sequence number."""
    if thread_idx.x != 0:
        return
    var n = 0
    for t in range(Int(ntok)):
        if slots[t * BW + TOPK] > 0:
            for i in range(TOPK):
                if slots[t * BW + i] < 0:
                    barea.unsafe_store[volatile=True](1 + 3 * n, ids[t * BW + i])
                    barea.unsafe_store[volatile=True](2 + 3 * n, Int32(t))
                    barea.unsafe_store[volatile=True](3 + 3 * n, Int32(i))
                    n += 1
    barea.unsafe_store[volatile=True](0, Int32(n))
    total[0] = Int32(n)
    threadfence[Scope.SYSTEM]()
    if post == 1:
        mb.unsafe_store[volatile=True](MB_NCOLD, Int32(-1) if n > 0 else Int32(0))
        threadfence[Scope.SYSTEM]()
        mb.unsafe_store[volatile=True](MB_SEQ, seq)


def gate_up_batch[GT: Int](
    gate: Pointer[UInt8, ImmutAnyOrigin], up: Pointer[UInt8, ImmutAnyOrigin],
    slots: Pointer[Int32, ImmutAnyOrigin], expert_bytes_: Int64,
    x: Pointer[Float32, ImmutAnyOrigin], h: Pointer[Float32, MutAnyOrigin],
    sgate: Pointer[UInt8, ImmutAnyOrigin], sup: Pointer[UInt8, ImmutAnyOrigin],
):
    """One warp per (token, slot, row) for hot slots; h[t][e][r]. Same per-lane math as gate_up_slots."""
    var expert_bytes = Int(expert_bytes_)
    var warp = Int(block_idx.x) * WARPS_PER_BLOCK + Int(thread_idx.x) // 32
    var lane = Int(thread_idx.x) % 32
    var tok = warp // (TOPK * D_FF)
    var rem = warp % (TOPK * D_FF)
    var e = rem // D_FF
    var r = rem % D_FF
    var sl = Int(slots[tok * BW + e])
    if sl < 0:
        return
    var gbase = gate
    var ubase = up
    if sl >= STAGE_BASE:
        sl -= STAGE_BASE
        gbase = sgate
        ubase = sup
    var xt = x + tok * D_MODEL
    comptime ROW = Q4K_BYTES * (D_MODEL // 256) if GT == 0 else Q5K_BYTES * (D_MODEL // 256)
    var off = sl * expert_bytes + r * ROW
    var g: Float32
    var u: Float32
    comptime if GT == 0:
        var xbase = 64 * (lane // 8) + 4 * (lane % 8)
        var ga: Float32 = 0
        var ua: Float32 = 0
        comptime for blk in range(D_MODEL // 256):
            var xl = xt.load[width=4](blk * 256 + xbase)
            var xh = xt.load[width=4](blk * 256 + xbase + 32)
            ga = q4k_block_gpu(gbase + off + blk * Q4K_BYTES, lane, xl, xh, ga)
            ua = q4k_block_gpu(ubase + off + blk * Q4K_BYTES, lane, xl, xh, ua)
        g = warp_tree(ga)
        u = warp_tree(ua)
    else:
        g = warp_tree(row_dot_q5k[1, D_MODEL // 256](gbase + off, xt, lane)[0])
        u = warp_tree(row_dot_q5k[1, D_MODEL // 256](ubase + off, xt, lane)[0])
    if lane == 0:
        h[tok * TOPK * D_FF + e * D_FF + r] = silu_mul[1](g, u)[0]


def down_batch(
    down: Pointer[UInt8, ImmutAnyOrigin], down_q80: Int64,
    slots: Pointer[Int32, ImmutAnyOrigin], expert_bytes_: Int64,
    h: Pointer[Float32, ImmutAnyOrigin], s: Pointer[Float32, MutAnyOrigin],
    sdown: Pointer[UInt8, ImmutAnyOrigin],
):
    """Block = (token, DROWS rows), TOPK warps (one per slot); s[t][e][i] for hot slots. Same math as down_slots."""
    var expert_bytes = Int(expert_bytes_)
    var nb = D_MODEL // DROWS
    var tok = Int(block_idx.x) // nb
    var i0 = (Int(block_idx.x) % nb) * DROWS
    var e = Int(thread_idx.x) // 32
    var lane = Int(thread_idx.x) % 32
    var sl = Int(slots[tok * BW + e])
    if sl < 0:
        return
    var base = down + sl * expert_bytes
    if sl >= STAGE_BASE:
        base = sdown + (sl - STAGE_BASE) * expert_bytes
    var he = h + tok * TOPK * D_FF + e * D_FF
    var so = s + tok * TOPK * D_MODEL + e * D_MODEL
    if down_q80 == 1:
        comptime for r in range(DROWS):
            var v = warp_tree(row_dot_q80[1, D_FF // 32](base + (i0 + r) * Q80_ROW, he, lane)[0])
            if lane == 0:
                so[i0 + r] = v
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
            so[i0 + r] = v


def wait_done_kernel(mb: Pointer[Int32, MutAnyOrigin], total: Pointer[Int32, ImmutAnyOrigin], seq: Int32):
    """One thread: if the batch had cold picks, wait for the CPU's done flag."""
    if thread_idx.x != 0 or total[0] == 0:
        return
    var spins = 0
    while mb.unsafe_load[volatile=True](MB_DONE)[0] != seq:
        spins += 1
        if spins > 400_000_000:
            mb.unsafe_store[volatile=True](MB_ERR, Int32(-1))
            return


def combine_batch_kernel(s: Pointer[Float32, ImmutAnyOrigin], bres: Pointer[Float32, MutAnyOrigin],
                         w: Pointer[Float32, ImmutAnyOrigin], slots: Pointer[Int32, ImmutAnyOrigin], y: Pointer[Float32, MutAnyOrigin]):
    """Grid = tokens x (D_MODEL / 256). y[t][i] = sum_e fma(w[t][e], s[t][e][i]) in slot order; cold sums from the CPU
    (bres, pinned host memory) at the same [t][e][i] position."""
    var nb = D_MODEL // 256
    var tok = Int(block_idx.x) // nb
    var i = (Int(block_idx.x) % nb) * 256 + Int(thread_idx.x)
    var o: Float32 = 0
    for e in range(TOPK):
        var v: Float32
        if slots[tok * BW + e] >= 0:
            v = s[tok * TOPK * D_MODEL + e * D_MODEL + i]
        else:
            v = bres.unsafe_load[volatile=True](tok * TOPK * D_MODEL + e * D_MODEL + i)[0]
        o = fma(w[tok * BW + e], v, o)
    y[tok * D_MODEL + i] = o



def remap_stage_kernel(ids: Pointer[Int32, ImmutAnyOrigin], slots: Pointer[Int32, MutAnyOrigin], stage_of: Pointer[Int32, ImmutAnyOrigin],
                       ntok: Int64):
    """Cold picks whose expert was uploaded for this batch now point at its staging slot (STAGE_BASE + index)."""
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i >= Int(ntok) * TOPK:
        return
    var t = i // TOPK
    var e = i % TOPK
    if slots[t * BW + e] < 0:
        var st = stage_of[Int(ids[t * BW + e])]
        if st >= 0:
            slots[t * BW + e] = Int32(STAGE_BASE) + st
