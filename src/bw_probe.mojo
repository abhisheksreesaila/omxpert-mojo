# Practical VRAM read ceiling for MoE-sized reads: sum N bytes with 16-byte loads, time back-to-back launches.
from std.time import perf_counter_ns
from std.math import ceildiv
from max.gpu import block_idx, block_dim, thread_idx, grid_dim
from max.gpu.host import DeviceContext


def read_kernel(p: Pointer[UInt32, ImmutAnyOrigin], n4: Int64, res: Pointer[UInt32, MutAnyOrigin]):
    var t = Int(block_idx.x * block_dim.x + thread_idx.x)
    var stride = Int(grid_dim.x * block_dim.x)
    var acc = SIMD[DType.uint32, 4](0)
    var i = t
    while i < Int(n4):
        acc ^= p.load[width=4](4 * i)
        i += stride
    if acc.reduce_add() == 12345:
        res[0] = acc[0]


def main() raises:
    with DeviceContext() as ctx:
        var total = 1 << 30
        var buf = ctx.enqueue_create_buffer[DType.uint32](total // 4)
        buf.enqueue_fill(1)
        var o = ctx.enqueue_create_buffer[DType.uint32](1)
        for mb in [12, 18, 31, 256]:
            var n = mb * 1024 * 1024
            for blocks in [328, 656, 1312, 2624]:
                var best = 1 << 62
                for _ in range(3):
                    var t0 = perf_counter_ns()
                    for it in range(200):
                        var off = ((it * 7919) % 30) * 32 * 1024 * 1024 // 4   # different region each launch
                        ctx.enqueue_function[read_kernel](buf.unsafe_ptr() + off, Int64(n // 16), o, grid_dim=blocks, block_dim=256)
                    ctx.synchronize()
                    best = min(best, perf_counter_ns() - t0)
                var us = Float64(best) / 200.0 / 1000.0
                print(mb, "MB blocks", blocks, ":", us, "us", Float64(n) / (us * 1000.0), "GB/s")
