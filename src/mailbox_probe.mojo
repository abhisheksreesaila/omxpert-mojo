# M3 building block: GPU <-> CPU round trip through pinned host memory, with no launches or syncs in between.
# One persistent GPU thread posts request i (volatile store + system fence) and spins until the CPU answers i.
from std.time import perf_counter_ns
from std.atomic import Atomic
from max.gpu.host import DeviceContext
from max.gpu.intrinsics import threadfence, Scope


def ping_kernel(mb: Pointer[Int32, MutAnyOrigin], n: Int64):
    for i in range(1, Int(n) + 1):
        mb.unsafe_store[volatile=True](0, Int32(i))
        threadfence[Scope.SYSTEM]()
        var spins = 0
        while mb.unsafe_load[volatile=True](16)[0] != Int32(i):
            spins += 1
            if spins > 200_000_000:
                mb.unsafe_store[volatile=True](32, Int32(-1))   # timeout marker
                return


def main() raises:
    with DeviceContext() as ctx:
        var mb = ctx.enqueue_create_host_buffer[DType.int32](64)
        ctx.synchronize()
        var p = mb.unsafe_ptr()
        for i in range(64):
            p[i] = 0
        var n = 20000
        ctx.enqueue_function[ping_kernel](p, Int64(n), grid_dim=1, block_dim=1)
        var t0 = perf_counter_ns()
        var t_first = 0
        for i in range(1, n + 1):
            while p.unsafe_load[volatile=True](0)[0] != Int32(i):
                if p.unsafe_load[volatile=True](32)[0] == -1:
                    raise Error("GPU timed out")
            if i == 1:
                t_first = perf_counter_ns()
            p.unsafe_store[volatile=True](16, Int32(i))
        var t1 = perf_counter_ns()
        ctx.synchronize()
        print("{\"mailbox_roundtrip_us\": ", Float64(t1 - t_first) / Float64(n - 1) / 1000.0, "}")
