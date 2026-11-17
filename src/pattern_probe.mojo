# Is the down kernel limited by its load pattern? Same loads as row_dot_q51 (10 experts x 2560 rows x 480 B), xor only.
from std.time import perf_counter_ns
from max.gpu import block_idx, block_dim, thread_idx
from max.gpu.host import DeviceContext


def pattern_kernel(base: Pointer[UInt8, ImmutAnyOrigin], h: Pointer[Float32, ImmutAnyOrigin], res: Pointer[UInt32, MutAnyOrigin], rows_per_warp: Int64, use_h: Int64):
    var warp = Int(block_idx.x) * (Int(block_dim.x) // 32) + Int(thread_idx.x) // 32
    var lane = Int(thread_idx.x) % 32
    var acc = UInt32(0)
    var hacc: Float32 = 0
    for r in range(Int(rows_per_warp)):
        var row = base + (warp * Int(rows_per_warp) + r) * 480
        comptime for it in range(3):
            var blk = 8 * it + lane // 4
            if blk < 20:
                var hw = (row + blk * 24).bitcast[UInt32]()
                acc ^= hw[0] ^ hw[1] ^ hw[2 + lane % 4]
                if use_h == 1:
                    var hl = h.load[width=4](blk * 32 + 4 * (lane % 4))
                    var hh = h.load[width=4](blk * 32 + 16 + 4 * (lane % 4))
                    hacc += hl.reduce_add() + hh.reduce_add()
    if acc == 12345 or hacc == 1.25:
        res[0] = acc


def main() raises:
    with DeviceContext() as ctx:
        var n = 10 * 2560 * 480
        var buf = ctx.enqueue_create_buffer[DType.uint8](n * 8)
        buf.enqueue_fill(1)
        var hb = ctx.enqueue_create_buffer[DType.float32](640)
        hb.enqueue_fill(0.5)
        var o = ctx.enqueue_create_buffer[DType.uint32](1)
        for use_h in [0, 1]:
            for rpw in [1, 2, 4]:
                for tpb in [128, 256, 512]:
                    var warps = 25600 // rpw
                    var best = 1 << 62
                    for _ in range(3):
                        var t0 = perf_counter_ns()
                        for it in range(200):
                            ctx.enqueue_function[pattern_kernel](buf.unsafe_ptr() + (it % 8) * n, hb, o, Int64(rpw), Int64(use_h), grid_dim=warps * 32 // tpb, block_dim=tpb)
                        ctx.synchronize()
                        best = min(best, perf_counter_ns() - t0)
                    var us = Float64(best) / 200.0 / 1000.0
                    print("use_h", use_h, "rows/warp", rpw, "tpb", tpb, ":", us, "us", Float64(n) / (us * 1000.0), "GB/s")
