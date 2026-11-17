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
from qmulti import q4k_rows_multi, q51_rows_multi
from qexperts import row_dot_q4k, row_dot_q5k, row_dot_q51, row_dot_q80, tree32, silu_mul, permute_x_q4k, arrange_h_q51
from cpu_moe import D_MODEL, D_FF, Q4K_ROW, Q5K_ROW, Q51_ROW, Q80_ROW, HV_PER_EXPERT
from hybrid import TOPK, MB_SEQ, MB_DONE, MB_NCOLD, MB_SLOT, MB_ID, MB_X, MB_S, MB_WORDS, MB_LAYER, MB_RING

comptime NW = 12
comptime NLMAX = 64
comptime RING = MB_RING
comptime BMAXT_DEFAULT = 512    # tokens per batch call (OMX_BMAXT overrides, see omx_lib)

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

    # batch job (prompt chunk / speculative verify)
    var j_mode: Int                     # 0 = one token, 1 = batch
    @__allow_legacy_any_origin_fields
    var barea: Pointer[Int32, MutAnyOrigin]   # pinned: count, then (expert, token, slot) triples (GPU writes)
    @__allow_legacy_any_origin_fields
    var bres: Pointer[Float32, MutAnyOrigin]  # pinned: [token][slot][D_MODEL] results (GPU reads)
    @__allow_legacy_any_origin_fields
    var bx: Pointer[Float32, MutAnyOrigin]    # host activations of the batch [token][D_MODEL]
    var b_ngroups: Int
    var b_npicks: Int
    @__allow_legacy_any_origin_fields
    var g_id: Pointer[Int, MutAnyOrigin]      # per group (unique cold expert): id, first pick, pick count
    @__allow_legacy_any_origin_fields
    var g_start: Pointer[Int, MutAnyOrigin]
    @__allow_legacy_any_origin_fields
    var g_cnt: Pointer[Int, MutAnyOrigin]
    @__allow_legacy_any_origin_fields
    var p_xoff: Pointer[Int, MutAnyOrigin]    # per pick (grouped): offset of its token's permuted x, token, slot
    @__allow_legacy_any_origin_fields
    var p_tok: Pointer[Int, MutAnyOrigin]
    @__allow_legacy_any_origin_fields
    var p_slot: Pointer[Int, MutAnyOrigin]
    @__allow_legacy_any_origin_fields
    var b_toks: Pointer[Int, MutAnyOrigin]    # tokens that need a permuted x, and how many
    var b_ntoks: Int
    @__allow_legacy_any_origin_fields
    var xp_all: Pointer[Float32, MutAnyOrigin]
    @__allow_legacy_any_origin_fields
    var h_all: Pointer[Float32, MutAnyOrigin]
    @__allow_legacy_any_origin_fields
    var hv_all: Pointer[Float32, MutAnyOrigin]

    var bmaxt: Int

    def __init__(out self, bmaxt: Int = BMAXT_DEFAULT):
        self.bmaxt = bmaxt
        var BMAXT = bmaxt
        var BMAXP = bmaxt * TOPK
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
        self.j_mode = 0
        self.barea = alloc[Int32](1).as_unsafe_any_origin()
        self.bres = alloc[Float32](1).as_unsafe_any_origin()
        self.bx = alloc[Float32](1).as_unsafe_any_origin()
        self.b_ngroups = 0
        self.b_npicks = 0
        self.g_id = alloc[Int](BMAXP).as_unsafe_any_origin()
        self.g_start = alloc[Int](BMAXP).as_unsafe_any_origin()
        self.g_cnt = alloc[Int](BMAXP).as_unsafe_any_origin()
        self.p_xoff = alloc[Int](BMAXP).as_unsafe_any_origin()
        self.p_tok = alloc[Int](BMAXP).as_unsafe_any_origin()
        self.p_slot = alloc[Int](BMAXP).as_unsafe_any_origin()
        self.b_toks = alloc[Int](BMAXT).as_unsafe_any_origin()
        self.b_ntoks = 0
        self.xp_all = alloc[Float32](BMAXT * D_MODEL).as_unsafe_any_origin()
        self.h_all = alloc[Float32](BMAXP * D_FF).as_unsafe_any_origin()
        self.hv_all = alloc[Float32](BMAXP * HV_PER_EXPERT).as_unsafe_any_origin()


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


@inline(.always)
def q4k_tile(row: Pointer[UInt8, _], xp: Pointer[Float32, _], offs: Pointer[Int, _], n: Int, res: Pointer[mut=True, Float32, _]):
    """Up to n tokens of one row, in tiles of 4 (the tile size never changes a token's arithmetic)."""
    var i = 0
    while i + 4 <= n:
        q4k_rows_multi[4, D_MODEL // 256](row, xp, offs + i, res + i)
        i += 4
    var rem = n - i
    if rem == 3:
        q4k_rows_multi[3, D_MODEL // 256](row, xp, offs + i, res + i)
    elif rem == 2:
        q4k_rows_multi[2, D_MODEL // 256](row, xp, offs + i, res + i)
    elif rem == 1:
        q4k_rows_multi[1, D_MODEL // 256](row, xp, offs + i, res + i)


@inline(.always)
def q51_tile(row: Pointer[UInt8, _], hv: Pointer[Float32, _], n: Int, res: Pointer[mut=True, Float32, _]):
    var i = 0
    while i + 4 <= n:
        q51_rows_multi[4, D_FF // 32](row, hv + i * HV_PER_EXPERT, HV_PER_EXPERT, res + i)
        i += 4
    var rem = n - i
    if rem == 3:
        q51_rows_multi[3, D_FF // 32](row, hv + i * HV_PER_EXPERT, HV_PER_EXPERT, res + i)
    elif rem == 2:
        q51_rows_multi[2, D_FF // 32](row, hv + i * HV_PER_EXPERT, HV_PER_EXPERT, res + i)
    elif rem == 1:
        q51_rows_multi[1, D_FF // 32](row, hv + i * HV_PER_EXPERT, HV_PER_EXPERT, res + i)


def setup_batch(p: Pointer[PoolCtl, MutAnyOrigin], l: Int):
    """Worker 0: group the batch's cold picks by expert (counting sort), list the tokens that need a permuted x."""
    var ba = p[].barea
    var n = Int(ba.unsafe_load[volatile=True](0)[0])
    var cnt = alloc[Int](512)
    var pos = alloc[Int](512)
    for e in range(512):
        cnt[e] = 0
    for i in range(n):
        cnt[Int(ba.unsafe_load[volatile=True](1 + 3 * i)[0])] += 1
    var ng = 0
    var acc = 0
    for e in range(512):
        pos[e] = acc
        if cnt[e] > 0:
            p[].g_id[ng] = e
            p[].g_start[ng] = acc
            p[].g_cnt[ng] = cnt[e]
            ng += 1
        acc += cnt[e]
    var need = alloc[Int](p[].bmaxt)
    for t in range(p[].bmaxt):
        need[t] = -1
    var nt = 0
    for i in range(n):
        var e = Int(ba.unsafe_load[volatile=True](1 + 3 * i)[0])
        var t = Int(ba.unsafe_load[volatile=True](2 + 3 * i)[0])
        var sl = Int(ba.unsafe_load[volatile=True](3 + 3 * i)[0])
        var q = pos[e]
        pos[e] += 1
        p[].p_tok[q] = t
        p[].p_slot[q] = sl
        if need[t] < 0:
            need[t] = nt
            p[].b_toks[nt] = t
            nt += 1
        p[].p_xoff[q] = need[t] * D_MODEL
    p[].b_ngroups = ng
    p[].b_npicks = n
    p[].b_ntoks = nt
    p[].j_layer = l
    cnt.free()
    pos.free()
    need.free()


def do_batch_job(p: Pointer[PoolCtl, MutAnyOrigin], w: Int, mut round: Int64):
    """All NW workers: permute x, gate/up rows (each cold expert read once for all its tokens), arrange h, down rows
    into bres[token][slot][:]. Per-token arithmetic is the one-token path's, so results are bit-identical."""
    var l = p[].j_layer
    var gate = p[].gate[l]
    var up = p[].up[l]
    var down = p[].down[l]
    var q5k = p[].gq5k[l]
    var q80 = p[].dq80[l]
    var geb = p[].geb[l]
    var deb = p[].deb[l]
    var ng = p[].b_ngroups
    var np = p[].b_npicks
    var nt = p[].b_ntoks
    # 0: permuted activations of the tokens involved
    for i in range(w * nt // NW, (w + 1) * nt // NW):
        permute_x_q4k(p[].bx + p[].b_toks[i] * D_MODEL, p[].xp_all + i * D_MODEL, D_MODEL)
    pool_barrier(p[].ctl, round)
    # A: gate/up, units = (group, row)
    var g_buf = alloc[Float32](p[].bmaxt)
    var u_buf = alloc[Float32](p[].bmaxt)
    var na = ng * D_FF
    for unit in range(w * na // NW, (w + 1) * na // NW):
        var j = unit // D_FF
        var r = unit % D_FF
        var e = p[].g_id[j]
        var p0 = p[].g_start[j]
        var c = p[].g_cnt[j]
        if q5k:
            for k in range(c):
                var xo = p[].xp_all + p[].p_xoff[p0 + k]
                var g = tree32(row_dot_q5k[32, D_MODEL // 256](gate + e * geb + r * Q5K_ROW, xo, 0))
                var u = tree32(row_dot_q5k[32, D_MODEL // 256](up + e * geb + r * Q5K_ROW, xo, 0))
                p[].h_all[(p0 + k) * D_FF + r] = silu_mul[1](g, u)[0]
        else:
            q4k_tile(gate + e * geb + r * Q4K_ROW, p[].xp_all, p[].p_xoff + p0, c, g_buf)
            q4k_tile(up + e * geb + r * Q4K_ROW, p[].xp_all, p[].p_xoff + p0, c, u_buf)
            for k in range(c):
                p[].h_all[(p0 + k) * D_FF + r] = silu_mul[1](g_buf[k], u_buf[k])[0]
    pool_barrier(p[].ctl, round)
    # arrange h per pick for the Q5_1 down rows
    if not q80:
        for q in range(w * np // NW, (w + 1) * np // NW):
            arrange_h_q51(p[].h_all + q * D_FF, p[].hv_all + q * HV_PER_EXPERT, D_FF // 32)
    pool_barrier(p[].ctl, round)
    # B: down rows, units = (group, row) -> bres[token][slot][row]
    var nb = ng * D_MODEL
    var bres = p[].bres
    for unit in range(w * nb // NW, (w + 1) * nb // NW):
        var j = unit // D_MODEL
        var i = unit % D_MODEL
        var e = p[].g_id[j]
        var p0 = p[].g_start[j]
        var c = p[].g_cnt[j]
        if q80:
            for k in range(c):
                g_buf[k] = tree32(row_dot_q80[32, D_FF // 32](down + e * deb + i * Q80_ROW, p[].h_all + (p0 + k) * D_FF, 0))
        else:
            q51_tile(down + e * deb + i * Q51_ROW, p[].hv_all + p0 * HV_PER_EXPERT, c, g_buf)
        for k in range(c):
            bres[(p[].p_tok[p0 + k] * TOPK + p[].p_slot[p0 + k]) * D_MODEL + i] = g_buf[k]
    g_buf.free()
    u_buf.free()
    pool_barrier(p[].ctl, round)


@inline(.always)
def idle_wait(spins: Int, mut t_idle: Int) -> Int:
    """Spin while work is likely (within 20 ms of the last job), then sleep 50 us per poll so an idle server doesn't keep
    12 cores at full power. 20 ms, not less: with 92% of picks hot, workers often see no cold expert for several layers
    in a row during an answer, and a sleeping worker costs ~60-100 us on wake. Only the first layer after a pause pays."""
    comptime IDLE = 1 << 40
    if spins >= IDLE:
        _ = external_call["usleep", Int32](UInt32(50))
        return spins
    var n = spins + 1
    if n == 1:
        t_idle = Int(perf_counter_ns())
    elif n % 1024 == 0 and Int(perf_counter_ns()) - t_idle > 20_000_000:
        return IDLE
    return n


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
            var spins = 0
            var t_idle = 0
            while True:
                if mb0.unsafe_load[volatile=True](MB_SEQ)[0] == seq:
                    mb = mb0
                    break
                if mb1.unsafe_load[volatile=True](MB_SEQ)[0] == seq:
                    mb = mb1
                    break
                if Atomic.load(ctl + C_STOP) != 0:
                    return arg
                spins = idle_wait(spins, t_idle)
            var l = p[].steplayer[step & 4095]
            var lw = Int(mb.unsafe_load[volatile=True](MB_LAYER)[0])
            if lw > 0:     # posted from the in-stream path: the GPU wrote the layer (no host step record)
                l = lw - 1
                mb.unsafe_store[volatile=True](MB_LAYER, Int32(0))
            if l < 0:
                l = step % p[].nl
            var k = Int(mb.unsafe_load[volatile=True](MB_NCOLD)[0])
            if k < 0:
                var t0b = perf_counter_ns()
                setup_batch(p, l)
                p[].j_mode = 1
                _ = Atomic.fetch_add(ctl + C_GEN, Int64(1))
                do_batch_job(p, 0, round)
                _ = Atomic.fetch_add(ctl + C_COLD, Int64(p[].b_npicks))
                _ = Atomic.fetch_add(ctl + C_BUSY_NS, Int64(perf_counter_ns() - t0b))
            elif k > 0:
                p[].j_mode = 0
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
    var spins = 0
    var t_idle = 0
    while True:
        var g = Atomic.load(ctl + C_GEN)
        if g != seen:
            seen = g
            spins = 0
            if p[].j_mode == 1:
                do_batch_job(p, w, round)
            else:
                do_job(p, w, round)
        elif Atomic.load(ctl + C_STOP) != 0:
            break
        else:
            spins = idle_wait(spins, t_idle)
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
