# M4 proof: the routed MoE of all 48 layers of Qwen3.8-Flash-Next, one token at a time, with real routing from the
# owner's held-out chat trace and the owner's profile deciding the hot set (results/pins.bin, llama.cpp's budget).
# Hot experts: packed per-layer VRAM pools on two GPUs. Cold experts: the CPU, straight from the mmapped GGUF.
# Prints the MoE time per token (router + top-k + experts + combine, 48 layers) for comparison with llama.cpp's.
# args: n_warm n_tokens
from std.sys import argv
from std.ffi import external_call
from std.time import perf_counter_ns
from std.random import random_float64, seed
from std.math import ceildiv
from std.memory import alloc, unsafe_memcpy
from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from hybrid import router_kernel, N_EXPERTS, TOPK, MB_SEQ, MB_DONE, MB_NCOLD, MB_ERR, MB_SLOT, MB_ID, MB_X, MB_S, MB_WORDS, DROWS
from layer48 import topk_trace_kernel, gate_up_slots, down_slots, combine_res_kernel, post_x_kernel, wait_x_kernel
from gpu_moe import D_MODEL, D_FF, WARPS_PER_BLOCK, TPB
from cpu_moe import moe_cpu, HV_PER_EXPERT
from cpu_pool import PoolCtl, start_pool, join_pool, NW, C_DONE, C_COLD, C_BUSY_NS
from std.atomic import Atomic
from max.algorithm import parallelize

comptime NL = 48
comptime RING = 16
comptime WINDOW = 12


struct LayerInfo(Copyable, Movable):
    var gpu: Int
    var nhot: Int
    var gfile: Int
    var goff: Int
    var gq5k: Bool
    var geb: Int
    var ufile: Int
    var uoff: Int
    var dfile: Int
    var doff: Int
    var dq80: Bool
    var deb: Int
    var rfile: Int
    var roff: Int

    def __init__(out self):
        self.gpu = 0
        self.nhot = 0
        self.gfile = 0
        self.goff = 0
        self.gq5k = False
        self.geb = 0
        self.ufile = 0
        self.uoff = 0
        self.dfile = 0
        self.doff = 0
        self.dq80 = False
        self.deb = 0
        self.rfile = 0
        self.roff = 0


def mmap_file(path: String) raises -> Pointer[UInt8, MutUntrackedOrigin]:
    var cpath = path + "\0"
    var fd = external_call["open64", Int32](cpath.unsafe_ptr(), Int32(0), Int32(0))
    if fd < 0:
        raise Error("open failed: " + path)
    var size = external_call["lseek64", Int64](fd, Int64(0), Int32(2))
    # PROT_READ = 1, MAP_SHARED = 1
    var p = external_call["mmap64", Pointer[UInt8, MutUntrackedOrigin]](Int64(0), size, Int32(1), Int32(1), fd, Int64(0))
    return p


def page_out(p: Pointer[UInt8, MutUntrackedOrigin], n: Int):
    """Give a hot expert's pages back to the OS (it lives in VRAM now), so the page cache keeps the cold experts.
    Same idea as llama.cpp EDX_PAGEOUT. Range rounded inward to whole 4 KiB pages."""
    var a = (Int(p) + 4095) // 4096 * 4096
    var b = (Int(p) + n) // 4096 * 4096
    if b > a:
        _ = external_call["madvise", Int32](a, b - a, Int32(21))   # MADV_PAGEOUT


def read_file(path: String) raises -> List[UInt8]:
    var f = open(path, "r")
    var b = f.read_bytes()
    f.close()
    return b^


def main() raises:
    var a = argv()
    var n_warm = Int(a[1])
    var n_tok = Int(a[2])
    var total = n_warm + n_tok
    var debug = len(a) > 3

    # ---- model table
    var mf = open("results/model.txt", "r")
    var text = mf.read()
    mf.close()
    var lines = text.splitlines()
    var files = lines[0].split(" ")
    var maps = List[Pointer[UInt8, MutUntrackedOrigin]]()
    for f in files:
        maps.append(mmap_file(String(f)))
    var L = List[LayerInfo]()
    for l in range(NL):
        var t = lines[l + 1].split(" ")
        var li = LayerInfo()
        li.gpu = Int(t[1])
        li.nhot = Int(t[2])
        li.gfile = Int(t[3])
        li.goff = Int(t[4])
        li.gq5k = t[5] == "Q5_K"
        li.geb = Int(t[6])
        li.ufile = Int(t[7])
        li.uoff = Int(t[8])
        li.dfile = Int(t[11])
        li.doff = Int(t[12])
        li.dq80 = t[13] == "Q8_0"
        li.deb = Int(t[14])
        li.rfile = Int(t[15])
        li.roff = Int(t[16])
        L.append(li^)
    var pins_b = read_file("results/pins.bin")
    var pins = pins_b.unsafe_ptr().bitcast[Int32]()
    var tr_b = read_file("results/trace_chat.bin")
    var trace = tr_b.unsafe_ptr().bitcast[Int32]()
    if len(tr_b) // 4 < total * NL * TOPK:
        raise Error("trace too short")

    var ctx0 = DeviceContext(device_id=0)
    var ctx1 = DeviceContext(device_id=1)
    var ctxs = [ctx0, ctx1]

    # ---- hot pools (profile order), routers, residency tables
    var gpools = List[DeviceBuffer[DType.uint8]]()
    var upools = List[DeviceBuffer[DType.uint8]]()
    var dpools = List[DeviceBuffer[DType.uint8]]()
    var routers = List[DeviceBuffer[DType.uint8]]()
    var restabs = List[DeviceBuffer[DType.int32]]()
    var t_load = perf_counter_ns()
    var hot_bytes = 0
    for l in range(NL):
        ref li = L[l]
        var ctx = ctxs[li.gpu]
        var n = max(li.nhot, 1)
        var hs_g = ctx.enqueue_create_host_buffer[DType.uint8](n * li.geb)
        var hs_u = ctx.enqueue_create_host_buffer[DType.uint8](n * li.geb)
        var hs_d = ctx.enqueue_create_host_buffer[DType.uint8](n * li.deb)
        var hs_r = ctx.enqueue_create_host_buffer[DType.uint8](N_EXPERTS * D_MODEL * 4)
        var hs_t = ctx.enqueue_create_host_buffer[DType.int32](N_EXPERTS)
        ctx.synchronize()
        for e in range(N_EXPERTS):
            hs_t.unsafe_ptr()[e] = pins[l * N_EXPERTS + e]
        var pg = hs_g.unsafe_ptr().as_unsafe_any_origin()
        var pu = hs_u.unsafe_ptr().as_unsafe_any_origin()
        var pd = hs_d.unsafe_ptr().as_unsafe_any_origin()
        var sg = maps[li.gfile] + li.goff
        var su = maps[li.ufile] + li.uoff
        var sd = maps[li.dfile] + li.doff
        var eg = li.geb
        var ed = li.deb
        var lbase = l * N_EXPERTS

        def copy_expert(e: Int) {imm}:
            var sl = Int(pins[lbase + e])
            if sl >= 0:
                unsafe_memcpy(dest=pg + sl * eg, src=sg + e * eg, count=eg)
                unsafe_memcpy(dest=pu + sl * eg, src=su + e * eg, count=eg)
                unsafe_memcpy(dest=pd + sl * ed, src=sd + e * ed, count=ed)

        parallelize(copy_expert, N_EXPERTS)
        for e in range(N_EXPERTS):
            if Int(pins[lbase + e]) >= 0:
                page_out(sg + e * eg, eg)
                page_out(su + e * eg, eg)
                page_out(sd + e * ed, ed)
        unsafe_memcpy(dest=hs_r.unsafe_ptr(), src=maps[li.rfile] + li.roff, count=N_EXPERTS * D_MODEL * 4)
        var dg = ctx.enqueue_create_buffer[DType.uint8](n * li.geb)
        var du = ctx.enqueue_create_buffer[DType.uint8](n * li.geb)
        var dd = ctx.enqueue_create_buffer[DType.uint8](n * li.deb)
        var dr = ctx.enqueue_create_buffer[DType.uint8](N_EXPERTS * D_MODEL * 4)
        var dt = ctx.enqueue_create_buffer[DType.int32](N_EXPERTS)
        ctx.enqueue_copy(dst_buf=dg, src_buf=hs_g)
        ctx.enqueue_copy(dst_buf=du, src_buf=hs_u)
        ctx.enqueue_copy(dst_buf=dd, src_buf=hs_d)
        ctx.enqueue_copy(dst_buf=dr, src_buf=hs_r)
        ctx.enqueue_copy(dst_buf=dt, src_buf=hs_t)
        ctx.synchronize()
        hot_bytes += li.nhot * (2 * li.geb + li.deb)
        gpools.append(dg^)
        upools.append(du^)
        dpools.append(dd^)
        routers.append(dr^)
        restabs.append(dt^)
    var load_s = Float64(perf_counter_ns() - t_load) / 1e9
    print("loaded", Float64(hot_bytes) / 1e9, "GB of hot experts in", load_s, "s")

    # ---- per-GPU working buffers, mailbox rings, trace copies
    var logits = List[DeviceBuffer[DType.float32]]()
    var wbuf = List[DeviceBuffer[DType.float32]]()
    var slots = List[DeviceBuffer[DType.int32]]()
    var hbuf = List[DeviceBuffer[DType.float32]]()
    var sbuf = List[DeviceBuffer[DType.float32]]()
    var xa = List[DeviceBuffer[DType.float32]]()
    var xb = List[DeviceBuffer[DType.float32]]()
    var trd = List[DeviceBuffer[DType.int32]]()
    var rings = List[HostBuffer[DType.int32]]()
    for g in range(2):
        var ctx = ctxs[g]
        logits.append(ctx.enqueue_create_buffer[DType.float32](N_EXPERTS))
        wbuf.append(ctx.enqueue_create_buffer[DType.float32](TOPK))
        slots.append(ctx.enqueue_create_buffer[DType.int32](TOPK + 1))
        hbuf.append(ctx.enqueue_create_buffer[DType.float32](TOPK * D_FF))
        sbuf.append(ctx.enqueue_create_buffer[DType.float32](TOPK * D_MODEL))
        xa.append(ctx.enqueue_create_buffer[DType.float32](D_MODEL))
        xb.append(ctx.enqueue_create_buffer[DType.float32](D_MODEL))
        var ht = ctx.enqueue_create_host_buffer[DType.int32](total * NL * TOPK)
        var r = ctx.enqueue_create_host_buffer[DType.int32](RING * MB_WORDS)
        ctx.synchronize()
        unsafe_memcpy(dest=ht.unsafe_ptr(), src=trace, count=total * NL * TOPK)
        for i in range(RING * MB_WORDS):
            r.unsafe_ptr()[i] = 0
        var dtr = ctx.enqueue_create_buffer[DType.int32](total * NL * TOPK)
        ctx.enqueue_copy(dst_buf=dtr, src_buf=ht)
        trd.append(dtr^)
        rings.append(r^)
    # activation hand-off buffers (GPU0 -> GPU1 and back), in pinned host memory
    var h01 = ctx1.enqueue_create_host_buffer[DType.int32](64 + D_MODEL)
    var h10 = ctx0.enqueue_create_host_buffer[DType.int32](64 + D_MODEL)
    var hx0 = ctx0.enqueue_create_host_buffer[DType.float32](D_MODEL)
    ctx0.synchronize()
    ctx1.synchronize()
    for i in range(64 + D_MODEL):
        h01.unsafe_ptr()[i] = 0
        h10.unsafe_ptr()[i] = 0
    seed(5)
    for i in range(D_MODEL):
        hx0.unsafe_ptr()[i] = Float32(random_float64(-0.5, 0.5))
    ctx0.enqueue_copy(dst_buf=xa[0], src_buf=hx0)
    ctx0.synchronize()

    # CPU scratch
    var xp = alloc[Float32](D_MODEL)
    var hc = alloc[Float32](TOPK * D_FF)
    var hvc = alloc[Float32](TOPK * HV_PER_EXPERT)
    var sc = alloc[Float32](TOPK * D_MODEL)
    var yc = alloc[Float32](D_MODEL)
    var wd = alloc[Float32](TOPK)
    var off_gu = alloc[Int](TOPK)
    var off_d = alloc[Int](TOPK)
    var cur = alloc[Int](2)   # which ping-pong buffer holds the current x on each GPU (0 = xa)
    cur[0] = 0
    cur[1] = 0
    # raw device pointers for the enqueue closure
    comptime PU8 = Pointer[UInt8, MutAnyOrigin]
    comptime PF32 = Pointer[Float32, MutAnyOrigin]
    comptime PI32 = Pointer[Int32, MutAnyOrigin]
    var gp = alloc[PU8](NL)
    var upp = alloc[PU8](NL)
    var dp = alloc[PU8](NL)
    var rp = alloc[PF32](NL)
    var tp = alloc[PI32](NL)
    for l in range(NL):
        gp[l] = gpools[l].unsafe_ptr().as_unsafe_any_origin()
        upp[l] = upools[l].unsafe_ptr().as_unsafe_any_origin()
        dp[l] = dpools[l].unsafe_ptr().as_unsafe_any_origin()
        rp[l] = routers[l].unsafe_ptr().bitcast[Float32]().as_unsafe_any_origin()
        tp[l] = restabs[l].unsafe_ptr().as_unsafe_any_origin()
    var lg = alloc[PF32](2)
    var wp = alloc[PF32](2)
    var slp = alloc[PI32](2)
    var hp = alloc[PF32](2)
    var sp = alloc[PF32](2)
    var xap = alloc[PF32](2)
    var xbp = alloc[PF32](2)
    var trp = alloc[PI32](2)
    var rgp = alloc[PI32](2)
    for g in range(2):
        lg[g] = logits[g].unsafe_ptr().as_unsafe_any_origin()
        wp[g] = wbuf[g].unsafe_ptr().as_unsafe_any_origin()
        slp[g] = slots[g].unsafe_ptr().as_unsafe_any_origin()
        hp[g] = hbuf[g].unsafe_ptr().as_unsafe_any_origin()
        sp[g] = sbuf[g].unsafe_ptr().as_unsafe_any_origin()
        xap[g] = xa[g].unsafe_ptr().as_unsafe_any_origin()
        xbp[g] = xb[g].unsafe_ptr().as_unsafe_any_origin()
        trp[g] = trd[g].unsafe_ptr().as_unsafe_any_origin()
        rgp[g] = rings[g].unsafe_ptr().as_unsafe_any_origin()
    var h01p = h01.unsafe_ptr().as_unsafe_any_origin()
    var h10p = h10.unsafe_ptr().as_unsafe_any_origin()
    var gq5 = alloc[Bool](NL)
    var dq8 = alloc[Bool](NL)
    var geb = alloc[Int](NL)
    var deb = alloc[Int](NL)
    var gpu = alloc[Int](NL)
    for l in range(NL):
        gq5[l] = L[l].gq5k
        dq8[l] = L[l].dq80
        geb[l] = L[l].geb
        deb[l] = L[l].deb
        gpu[l] = L[l].gpu

    @parameter
    def enqueue(step: Int) raises:
        var t = step // NL
        var l = step % NL
        var g = gpu[l]
        var ctx = ctx0 if g == 0 else ctx1
        var seq = Int32(step + 1)
        var xin = xap[g] if cur[g] == 0 else xbp[g]
        var xout = xbp[g] if cur[g] == 0 else xap[g]
        if l == 24:
            ctx.enqueue_function[wait_x_kernel](h01p, Int32(t + 1), xin, grid_dim=1, block_dim=256)
        if l == 0 and t > 0:
            ctx.enqueue_function[wait_x_kernel](h10p, Int32(t), xin, grid_dim=1, block_dim=256)
        var mb = rgp[g] + (Int(seq) % RING) * MB_WORDS
        ctx.enqueue_function[router_kernel](rp[l], xin, lg[g], grid_dim=N_EXPERTS // WARPS_PER_BLOCK, block_dim=TPB)
        if debug and step == 0:
            ctx.synchronize()
            print("router ok")
        ctx.enqueue_function[topk_trace_kernel](lg[g], trp[g] + step * TOPK, tp[l], xin, wp[g], slp[g], mb, seq,
            grid_dim=1, block_dim=256)
        if debug and step == 0:
            ctx.synchronize()
            print("topk ok, seq posted:", mb[MB_SEQ], "ncold", mb[MB_NCOLD])
        if gq5[l]:
            ctx.enqueue_function[gate_up_slots[1]](gp[l], upp[l], slp[g], Int64(geb[l]), xin, hp[g],
                grid_dim=ceildiv(TOPK * D_FF, WARPS_PER_BLOCK), block_dim=TPB)
        else:
            ctx.enqueue_function[gate_up_slots[0]](gp[l], upp[l], slp[g], Int64(geb[l]), xin, hp[g],
                grid_dim=ceildiv(TOPK * D_FF, WARPS_PER_BLOCK), block_dim=TPB)
        ctx.enqueue_function[down_slots](dp[l], Int64(1 if dq8[l] else 0), slp[g], Int64(deb[l]), hp[g], sp[g],
            grid_dim=D_MODEL // DROWS, block_dim=TOPK * 32)
        if debug and step == 0:
            ctx.synchronize()
            print("gate/up + down ok")
        ctx.enqueue_function[combine_res_kernel](sp[g], mb, wp[g], slp[g], xin, xout, seq,
            grid_dim=ceildiv(D_MODEL, 256), block_dim=256)
        cur[g] = 1 - cur[g]
        if l == 23:
            ctx.enqueue_function[post_x_kernel](xout, h01p, Int32(t + 1), grid_dim=1, block_dim=256)
        if l == 47:
            ctx.enqueue_function[post_x_kernel](xout, h10p, Int32(t + 1), grid_dim=1, block_dim=256)

    # Load every kernel variant on both GPUs BEFORE any kernel spins: a first launch loads its module, and module
    # loading waits for the device to idle, which deadlocks against a combine kernel waiting for the CPU.
    for g in range(2):
        var ctx = ctx0 if g == 0 else ctx1
        var dslots = ctx.enqueue_create_host_buffer[DType.int32](TOPK + 1)
        var dmb = ctx.enqueue_create_host_buffer[DType.int32](MB_WORDS)
        var dflag = ctx.enqueue_create_host_buffer[DType.int32](64 + D_MODEL)
        ctx.synchronize()
        for i in range(TOPK):
            dslots.unsafe_ptr()[i] = -1
        dslots.unsafe_ptr()[TOPK] = 0
        dflag.unsafe_ptr()[0] = -7
        var sl = dslots.unsafe_ptr().as_unsafe_any_origin()
        var mbd = dmb.unsafe_ptr().as_unsafe_any_origin()
        var fl = dflag.unsafe_ptr().as_unsafe_any_origin()
        var l0 = 0 if g == 0 else 24
        ctx.enqueue_function[router_kernel](rp[l0], xap[g], lg[g], grid_dim=N_EXPERTS // WARPS_PER_BLOCK, block_dim=TPB)
        ctx.enqueue_function[topk_trace_kernel](lg[g], trp[g], tp[l0], xap[g], wp[g], hp[g].bitcast[Int32](), mbd, Int32(-3),
            grid_dim=1, block_dim=256)
        ctx.enqueue_function[gate_up_slots[0]](gp[l0], upp[l0], sl, Int64(geb[l0]), xap[g], hp[g],
            grid_dim=ceildiv(TOPK * D_FF, WARPS_PER_BLOCK), block_dim=TPB)
        ctx.enqueue_function[gate_up_slots[1]](gp[l0], upp[l0], sl, Int64(geb[l0]), xap[g], hp[g],
            grid_dim=ceildiv(TOPK * D_FF, WARPS_PER_BLOCK), block_dim=TPB)
        ctx.enqueue_function[down_slots](dp[l0], Int64(0), sl, Int64(deb[l0]), hp[g], sp[g],
            grid_dim=D_MODEL // DROWS, block_dim=TOPK * 32)
        ctx.enqueue_function[combine_res_kernel](sp[g], mbd, wp[g], sl, xap[g], xbp[g], Int32(-3),
            grid_dim=ceildiv(D_MODEL, 256), block_dim=256)
        ctx.enqueue_function[post_x_kernel](xap[g], fl, Int32(-7), grid_dim=1, block_dim=256)
        ctx.enqueue_function[wait_x_kernel](fl, Int32(-7), xbp[g], grid_dim=1, block_dim=256)
        ctx.synchronize()
    # the warm-up overwrote x on GPU 0: start again from the random activation
    ctx0.enqueue_copy(dst_buf=xa[0], src_buf=hx0)
    ctx0.synchronize()

    var nsteps = total * NL
    # the CPU co-processor: persistent spinning workers that watch the mailbox rings themselves
    var pool = alloc[PoolCtl](1)
    pool.unsafe_write(PoolCtl())
    var pp = pool.as_unsafe_any_origin()
    pp[].nl = NL
    pp[].total_steps = nsteps
    for l in range(NL):
        pp[].gpu[l] = gpu[l]
        pp[].gate[l] = (maps[L[l].gfile] + L[l].goff).as_unsafe_any_origin()
        pp[].up[l] = (maps[L[l].ufile] + L[l].uoff).as_unsafe_any_origin()
        pp[].down[l] = (maps[L[l].dfile] + L[l].doff).as_unsafe_any_origin()
        pp[].geb[l] = geb[l]
        pp[].deb[l] = deb[l]
        pp[].gq5k[l] = gq5[l]
        pp[].dq80[l] = dq8[l]
    pp[].ring0 = rgp[0]
    pp[].ring1 = rgp[1]
    var tids = alloc[UInt64](NW).as_unsafe_any_origin()
    start_pool(pp, tids)
    var ctl = pp[].ctl
    var t_meas = perf_counter_ns()
    var cold0 = Int64(0)
    var busy0 = Int64(0)
    for s in range(min(WINDOW, nsteps)):
        enqueue(s)
    if n_warm == 0:
        t_meas = perf_counter_ns()
    for s in range(nsteps):
        while Atomic.load(ctl + C_DONE) < Int64(s + 1):   # step s served (or had nothing cold)
            pass
        if s + 1 == n_warm * NL:
            t_meas = perf_counter_ns()
            cold0 = Atomic.load(ctl + C_COLD)
            busy0 = Atomic.load(ctl + C_BUSY_NS)
        if s + WINDOW < nsteps:
            enqueue(s + WINDOW)
    ctx0.synchronize()
    ctx1.synchronize()
    join_pool(pp, tids)
    var cold_total = Int(Atomic.load(ctl + C_COLD) - cold0)
    var cpu_ns = Int(Atomic.load(ctl + C_BUSY_NS) - busy0)
    var dt = perf_counter_ns() - t_meas
    print("{\"tokens\": ", n_tok, ", \"moe_ms_per_token\": ", Float64(dt) / Float64(n_tok) / 1e6,
          ", \"cold_per_token\": ", Float64(cold_total) / Float64(n_tok), ", \"cpu_ms_per_token\": ",
          Float64(cpu_ns) / Float64(n_tok) / 1e6, ", \"hot_gb\": ", Float64(hot_bytes) / 1e9, ", \"load_s\": ", load_s, "}")
