# The CPU as a co-processor: NW persistent, spinning worker threads (pthreads, made from Mojo through the C FFI).
# Worker 0 watches the GPUs' mailbox rings itself; when a layer posts cold experts, all workers split the rows,
# meet at atomic barriers (~1 us, vs ~240 us for a parallelize() dispatch), and write each expert's sums straight into
# the mailbox slot the GPU combine reads. Same per-row math as cpu_moe (qexperts.mojo), so the bits are unchanged.
from std.ffi import external_call
from std.atomic import Atomic
from std.memory import alloc
from std.math import fma
from std.sys.intrinsics import prefetch
from std.time import perf_counter_ns
from qexperts import row_dot_q4k, row_dot_q5k, row_dot_q51, row_dot_q80, tree32, silu_mul, permute_x_q4k, arrange_h_q51
from cpu_moe import D_MODEL, D_FF, Q4K_ROW, Q5K_ROW, Q51_ROW, Q80_ROW, HV_PER_EXPERT
from hybrid import TOPK, MB_SEQ, MB_DONE, MB_NCOLD, MB_SLOT, MB_ID, MB_X, MB_S, MB_WORDS

comptime NW = 12
comptime NLMAX = 64
comptime RING = 16

# control words (Int64)
comptime C_GEN = 0       # job generation: worker 0 bumps it to release the others
comptime C_BAR = 8       # barrier counter
comptime C_STOP = 16
comptime C_DONE = 24     # steps completed (for the host's enqueue window)
comptime C_IDS = 32      # worker id dispenser
comptime C_BUSY_NS = 40  # time spent on cold experts (ns)
comptime C_COLD = 48     # cold experts served


struct PoolCtl:
    @__allow_legacy_any_origin_fields
    var ctl: Pointer[Int64, MutAnyOrigin]
    var nl: Int
    @__allow_legacy_any_origin_fields
    var gpu: Pointer[Int, MutAnyOrigin]
    @__allow_legacy_any_origin_fields
    var gate: Pointer[Pointer[UInt8, MutAnyOrigin], MutAnyOrigin]
    @__allow_legacy_any_origin_fields
    var up: Pointer[Pointer[UInt8, MutAnyOrigin], MutAnyOrigin]
    @__allow_legacy_any_origin_fields
    var down: Pointer[Pointer[UInt8, MutAnyOrigin], MutAnyOrigin]
    @__allow_legacy_any_origin_fields
    var geb: Pointer[Int, MutAnyOrigin]
    @__allow_legacy_any_origin_fields
    var deb: Pointer[Int, MutAnyOrigin]
    @__allow_legacy_any_origin_fields
    var gq5k: Pointer[Bool, MutAnyOrigin]
    @__allow_legacy_any_origin_fields
    var dq80: Pointer[Bool, MutAnyOrigin]
    @__allow_legacy_any_origin_fields
    var ring0: Pointer[Int32, MutAnyOrigin]
    @__allow_legacy_any_origin_fields
    var ring1: Pointer[Int32, MutAnyOrigin]
    var total_steps: Int
    @__allow_legacy_any_origin_fields
    var steplayer: Pointer[Int, MutAnyOrigin]   # host records the layer of each step (index step & 4095); -1 = use step % nl
    # job (written by worker 0 before bumping C_GEN)
    var j_layer: Int
    var j_k: Int
    @__allow_legacy_any_origin_fields
    var j_mb: Pointer[Int32, MutAnyOrigin]
    @__allow_legacy_any_origin_fields
    var j_off_gu: Pointer[Int, MutAnyOrigin]
    @__allow_legacy_any_origin_fields
    var j_off_d: Pointer[Int, MutAnyOrigin]
    @__allow_legacy_any_origin_fields
    var j_slot: Pointer[Int, MutAnyOrigin]
    @__allow_legacy_any_origin_fields
    var xp: Pointer[Float32, MutAnyOrigin]
    @__allow_legacy_any_origin_fields
    var h: Pointer[Float32, MutAnyOrigin]
    @__allow_legacy_any_origin_fields
    var hv: Pointer[Float32, MutAnyOrigin]

    def __init__(out self):
        self.ctl = alloc[Int64](64).as_unsafe_any_origin()
        for i in range(64):
            self.ctl[i] = 0
        self.nl = 0
        self.gpu = alloc[Int](NLMAX).as_unsafe_any_origin()
        self.gate = alloc[Pointer[UInt8, MutAnyOrigin]](NLMAX).as_unsafe_any_origin()
        self.up = alloc[Pointer[UInt8, MutAnyOrigin]](NLMAX).as_unsafe_any_origin()
        self.down = alloc[Pointer[UInt8, MutAnyOrigin]](NLMAX).as_unsafe_any_origin()
        self.geb = alloc[Int](NLMAX).as_unsafe_any_origin()
        self.deb = alloc[Int](NLMAX).as_unsafe_any_origin()
        self.gq5k = alloc[Bool](NLMAX).as_unsafe_any_origin()
        self.dq80 = alloc[Bool](NLMAX).as_unsafe_any_origin()
        self.ring0 = alloc[Int32](1).as_unsafe_any_origin()
        self.ring1 = alloc[Int32](1).as_unsafe_any_origin()
        self.total_steps = 0
        self.steplayer = alloc[Int](4096).as_unsafe_any_origin()
        for i in range(4096):
            self.steplayer[i] = -1
        self.j_layer = 0
        self.j_k = 0
        self.j_mb = alloc[Int32](1).as_unsafe_any_origin()
        self.j_off_gu = alloc[Int](TOPK).as_unsafe_any_origin()
        self.j_off_d = alloc[Int](TOPK).as_unsafe_any_origin()
        self.j_slot = alloc[Int](TOPK).as_unsafe_any_origin()
        self.xp = alloc[Float32](D_MODEL).as_unsafe_any_origin()
        self.h = alloc[Float32](TOPK * D_FF).as_unsafe_any_origin()
        self.hv = alloc[Float32](TOPK * HV_PER_EXPERT).as_unsafe_any_origin()


@inline(.always)
def pool_barrier(ctl: Pointer[Int64, MutAnyOrigin], mut round: Int64):
    round += 1
    _ = Atomic.fetch_add(ctl + C_BAR, Int64(1))
    while Atomic.load(ctl + C_BAR) < round * NW:
        pass


def do_job(p: Pointer[PoolCtl, MutAnyOrigin], w: Int, mut round: Int64):
    """All NW workers: gate/up rows, barrier, arrange h, barrier, down rows straight into the mailbox slots, barrier."""
    var l = p[].j_layer
    var k = p[].j_k
    var gate = p[].gate[l]
    var up = p[].up[l]
    var down = p[].down[l]
    var q5k = p[].gq5k[l]
    var q80 = p[].dq80[l]
    var og = p[].j_off_gu
    var od = p[].j_off_d
    var xp = p[].xp
    var h = p[].h
    var hv = p[].hv
    var na = k * D_FF
    for t in range(w * na // NW, (w + 1) * na // NW):
        var e = t // D_FF
        var r = t % D_FF
        var g: Float32
        var u: Float32
        if q5k:
            g = tree32(row_dot_q5k[32, D_MODEL // 256](gate + og[e] + r * Q5K_ROW, xp, 0))
            u = tree32(row_dot_q5k[32, D_MODEL // 256](up + og[e] + r * Q5K_ROW, xp, 0))
        else:
            g = tree32(row_dot_q4k[32, D_MODEL // 256](gate + og[e] + r * Q4K_ROW, xp, 0))
            u = tree32(row_dot_q4k[32, D_MODEL // 256](up + og[e] + r * Q4K_ROW, xp, 0))
        h[t] = silu_mul[1](g, u)[0]
    pool_barrier(p[].ctl, round)
    if not q80:
        var e = w
        while e < k:
            arrange_h_q51(h + e * D_FF, hv + e * HV_PER_EXPERT, D_FF // 32)
            e += NW
    pool_barrier(p[].ctl, round)
    var sout = p[].j_mb.bitcast[Float32]() + MB_S
    var i0 = w * D_MODEL // NW
    var i1 = (w + 1) * D_MODEL // NW
    for e in range(k):
        var dst = sout + p[].j_slot[e] * D_MODEL
        for i in range(i0, i1):
            var v: Float32
            if q80:
                v = tree32(row_dot_q80[32, D_FF // 32](down + od[e] + i * Q80_ROW, h + e * D_FF, 0))
            else:
                v = tree32(row_dot_q51[32, D_FF // 32](down + od[e] + i * Q51_ROW, hv + e * HV_PER_EXPERT, 0))
            dst[i] = v
    pool_barrier(p[].ctl, round)


def worker_entry(arg: Pointer[PoolCtl, MutAnyOrigin]) abi("C") -> Pointer[PoolCtl, MutAnyOrigin]:
    var p = arg
    var ctl = p[].ctl
    var w = Int(Atomic.fetch_add(ctl + C_IDS, Int64(1)))
    var round = Int64(0)
    if w == 0:
        # coordinator: walk the steps in order, serve each layer's cold experts with the whole team
        for step in range(p[].total_steps):
            var seq = Int32(step + 1)
            # the request may come from either GPU: watch both rings for this sequence number
            var mb0 = p[].ring0 + (Int(seq) % RING) * MB_WORDS
            var mb1 = p[].ring1 + (Int(seq) % RING) * MB_WORDS
            var mb = mb0
            while True:
                if mb0.unsafe_load[volatile=True](MB_SEQ)[0] == seq:
                    mb = mb0
                    break
                if mb1.unsafe_load[volatile=True](MB_SEQ)[0] == seq:
                    mb = mb1
                    break
                if Atomic.load(ctl + C_STOP) != 0:
                    return arg
            var l = p[].steplayer[step & 4095]
            if l < 0:
                l = step % p[].nl
            var k = Int(mb.unsafe_load[volatile=True](MB_NCOLD)[0])
            if k > 0:
                var t0 = perf_counter_ns()
                p[].j_layer = l
                p[].j_k = k
                p[].j_mb = mb
                for j in range(k):
                    var id = Int(mb.unsafe_load[volatile=True](MB_ID + j)[0])
                    p[].j_off_gu[j] = id * p[].geb[l]
                    p[].j_off_d[j] = id * p[].deb[l]
                    p[].j_slot[j] = Int(mb.unsafe_load[volatile=True](MB_SLOT + j)[0])
                permute_x_q4k(mb.bitcast[Float32]() + MB_X, p[].xp, D_MODEL)
                _ = Atomic.fetch_add(ctl + C_GEN, Int64(1))     # release the team
                do_job(p, 0, round)
                _ = Atomic.fetch_add(ctl + C_COLD, Int64(k))
                _ = Atomic.fetch_add(ctl + C_BUSY_NS, Int64(perf_counter_ns() - t0))
            mb.unsafe_store[volatile=True](MB_DONE, seq)
            _ = Atomic.fetch_add(ctl + C_DONE, Int64(1))
        Atomic.store(ctl + C_STOP, Int64(1))
        return arg
    var seen = Int64(0)
    while True:
        var g = Atomic.load(ctl + C_GEN)
        if g != seen:
            seen = g
            do_job(p, w, round)
        elif Atomic.load(ctl + C_STOP) != 0:
            break
    return arg


def start_pool(p: Pointer[PoolCtl, MutAnyOrigin], tids: Pointer[UInt64, MutAnyOrigin]) raises:
    for i in range(NW):
        var rc = external_call["pthread_create", Int32](tids + i, Int64(0), worker_entry, p)
        if rc != 0:
            raise Error("pthread_create failed")


def join_pool(p: Pointer[PoolCtl, MutAnyOrigin], tids: Pointer[UInt64, MutAnyOrigin]):
    Atomic.store(p[].ctl + C_STOP, Int64(1))
    for i in range(NW):
        _ = external_call["pthread_join", Int32](tids[i], Int64(0))
