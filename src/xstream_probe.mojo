# M8 probe: can Mojo launch its kernels on a CUDA stream created by someone else (llama.cpp's cudart stream), in the
# same CUDA context, and are those launches recorded by CUDA graph capture? Built as a shared library; the C driver
# tools/probe/xstream.c creates the stream, captures, replays and checks.
from std.ffi import external_call
from std.runtime import initialize_runtime
from std.memory import alloc
from max.gpu import thread_idx
from max.gpu.host import DeviceContext


def bump(p: Pointer[Int32, MutAnyOrigin], v: Int32):
    if thread_idx.x == 0:
        p[0] = p[0] + v


@export("probe_ctx")
def probe_ctx(dev: Int32) abi("C") -> Int64:
    """The CUcontext Mojo's DeviceContext uses on this device."""
    initialize_runtime()
    try:
        var ctx = DeviceContext(device_id=Int(dev))
        var res = Int64(0)
        _ = external_call["AsyncRT_DeviceContext_cuda_context", Pointer[UInt8, MutUntrackedOrigin]](Pointer(to=res), ctx._handle)
        return res
    except e:
        print("probe_ctx:", e)
        return -1


@export("probe_launch")
def probe_launch(dev: Int32, stream: Int64, p: Int64, v: Int32) abi("C") -> Int32:
    """Enqueue bump(p, v) on the external stream."""
    try:
        var ctx = DeviceContext(device_id=Int(dev))
        var s = ctx.create_external_stream(Pointer[NoneType, MutAnyOrigin](unsafe_from_address=Int(stream)))
        var f = ctx.compile_function[bump]()
        s.enqueue_function(f, Pointer[Int32, MutAnyOrigin](unsafe_from_address=Int(p)), v, grid_dim=1, block_dim=32)
        return 0
    except e:
        print("probe_launch:", e)
        return -1
