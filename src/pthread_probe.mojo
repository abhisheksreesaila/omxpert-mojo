# Persistent spinning workers made with pthread_create from Mojo; dispatch latency vs parallelize.
from std.ffi import external_call
from std.atomic import Atomic
from std.memory import alloc
from std.time import perf_counter_ns

comptime NW = 12


def worker_entry(arg: Pointer[Int64, MutAnyOrigin]) abi("C") -> Pointer[Int64, MutAnyOrigin]:
    # arg[0] = generation (posted by main), arg[1] = done counter, arg[2] = stop, arg[3 + id] = my id slot
    var seen = Int64(0)
    while True:
        var g = Atomic.load(arg)
        if g != seen:
            seen = g
            _ = Atomic.fetch_add(arg + 1, Int64(1))
        if Atomic.load(arg + 2) != 0:
            break
    return arg


def main() raises:
    var ctl = alloc[Int64](8)
    for i in range(8):
        ctl[i] = 0
    var tids = alloc[UInt64](NW)
    for i in range(NW):
        var rc = external_call["pthread_create", Int32](tids + i, Int64(0), worker_entry, ctl)
        if rc != 0:
            raise Error("pthread_create failed")
    var n = 20000
    var t0 = perf_counter_ns()
    for it in range(1, n + 1):
        _ = Atomic.fetch_add(ctl, Int64(1))                 # wake everyone
        while Atomic.load(ctl + 1) < Int64(it * NW):        # wait until all NW workers answered
            pass
    var dt = perf_counter_ns() - t0
    ctl[2] = 1
    for i in range(NW):
        _ = external_call["pthread_join", Int32](tids[i], Int64(0))
    print("{\"pool_dispatch_us\": ", Float64(dt) / Float64(n) / 1000.0, ", \"workers\": ", NW, "}")
