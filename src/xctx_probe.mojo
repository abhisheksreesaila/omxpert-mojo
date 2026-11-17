# Can a kernel on GPU 1 write pinned host memory allocated through GPU 0's context (and vice versa)?
from max.gpu import thread_idx
from max.gpu.host import DeviceContext
from max.gpu.intrinsics import threadfence, Scope


def poke(p: Pointer[Int32, MutAnyOrigin], v: Int32):
    if thread_idx.x == 0:
        p.unsafe_store[volatile=True](0, v)
        threadfence[Scope.SYSTEM]()


def main() raises:
    var c0 = DeviceContext(device_id=0)
    var c1 = DeviceContext(device_id=1)
    var h0 = c0.enqueue_create_host_buffer[DType.int32](16)
    var h1 = c1.enqueue_create_host_buffer[DType.int32](16)
    c0.synchronize()
    c1.synchronize()
    h0.unsafe_ptr()[0] = 0
    h1.unsafe_ptr()[0] = 0
    c1.enqueue_function[poke](h0.unsafe_ptr().as_unsafe_any_origin(), Int32(11), grid_dim=1, block_dim=32)   # GPU1 -> buffer of ctx0
    c1.synchronize()
    c0.enqueue_function[poke](h1.unsafe_ptr().as_unsafe_any_origin(), Int32(22), grid_dim=1, block_dim=32)   # GPU0 -> buffer of ctx1
    c0.synchronize()
    print("GPU1 wrote ctx0 buffer:", h0.unsafe_ptr()[0], "(want 11);  GPU0 wrote ctx1 buffer:", h1.unsafe_ptr()[0], "(want 22)")
