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
