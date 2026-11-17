# GPU0 spins on a pinned flag that GPU1 sets, concurrently (the M4 layer-47 -> layer-0 hand-off in isolation).
from std.time import perf_counter_ns, sleep
from max.gpu import thread_idx
from max.gpu.host import DeviceContext
from max.gpu.intrinsics import threadfence, Scope


def poke(p: Pointer[Int32, MutAnyOrigin], v: Int32):
    if thread_idx.x == 0:
        threadfence[Scope.SYSTEM]()
        p.unsafe_store[volatile=True](0, v)
        threadfence[Scope.SYSTEM]()


def spin(p: Pointer[Int32, MutAnyOrigin], v: Int32, out_: Pointer[Int32, MutAnyOrigin]):
    if thread_idx.x == 0:
        var n = 0
        while p.unsafe_load[volatile=True](0)[0] != v:
            n += 1
            if n > 50_000_000:
                out_.unsafe_store[volatile=True](0, Int32(-1))
                return
        out_.unsafe_store[volatile=True](0, Int32(n % 1000000))


def main() raises:
    var c0 = DeviceContext(device_id=0)
    var c1 = DeviceContext(device_id=1)
    for owner in [0, 1]:
        var hb = c0.enqueue_create_host_buffer[DType.int32](16) if owner == 0 else c1.enqueue_create_host_buffer[DType.int32](16)
        var res = c0.enqueue_create_host_buffer[DType.int32](16)
        c0.synchronize()
        c1.synchronize()
        hb.unsafe_ptr()[0] = 0
        res.unsafe_ptr()[0] = 99
        c0.enqueue_function[spin](hb.unsafe_ptr().as_unsafe_any_origin(), Int32(5), res.unsafe_ptr().as_unsafe_any_origin(), grid_dim=1, block_dim=32)
        sleep(0.2)
        c1.enqueue_function[poke](hb.unsafe_ptr().as_unsafe_any_origin(), Int32(5), grid_dim=1, block_dim=32)
        c1.synchronize()
        var t0 = perf_counter_ns()
        c0.synchronize()
        print("flag owned by ctx", owner, ": GPU0 saw GPU1's write:", res.unsafe_ptr()[0] != -1, " (", Float64(perf_counter_ns() - t0) / 1e6, "ms after poke)")
