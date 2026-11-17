# libomxmoe.so: the Mojo expert layer as a C library, so llama.cpp can call it for the routed MoE while it keeps
# attention, linear attention, the n-gram table and the shared expert.
#   void* omx_init(const char* model_txt, const char* pins_bin)     load hot experts to VRAM, start the CPU pool
#   int   omx_moe(void* h, int layer, const float* x, float* y)     one token through one layer's routed experts
#   void  omx_stats(void* h, double* out3)                           calls, cold experts served, CPU ms
# x/y are host float[2560]. Hot experts: packed VRAM pools from the owner's profile. Cold: CPU pool via the mailbox.
from std.ffi import external_call
from std.time import perf_counter_ns
from std.math import ceildiv
from std.memory import alloc, unsafe_memcpy
from std.atomic import Atomic
from std.runtime import initialize_runtime
from max.algorithm import parallelize
from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from hybrid import router_kernel, N_EXPERTS, TOPK, MB_WORDS, DROWS, MB_NCOLD, MB_SEQ
from layer48 import topk_slots_kernel, gate_up_slots, down_slots, combine_res_kernel, router_batch_kernel, topk_batch_kernel, collect_cold_kernel, gate_up_batch, down_batch, wait_done_kernel, combine_batch_kernel, BW, remap_stage_kernel
from gpu_moe import D_MODEL, D_FF, WARPS_PER_BLOCK, TPB
from cpu_pool import PoolCtl, start_pool, join_pool, NW, RING, C_COLD, C_BUSY_NS, BMAXT_DEFAULT
from std.os import getenv

comptime NL = 48
comptime PU8 = Pointer[UInt8, MutAnyOrigin]
comptime PF32 = Pointer[Float32, MutAnyOrigin]
comptime PI32 = Pointer[Int32, MutAnyOrigin]


def mmap_file(path: String) raises -> Pointer[UInt8, MutUntrackedOrigin]:
    var cpath = path + "\0"
    var fd = external_call["open64", Int32](cpath.unsafe_ptr(), Int32(0), Int32(0))
    if fd < 0:
        raise Error("open failed: " + path)
    var size = external_call["lseek64", Int64](fd, Int64(0), Int32(2))
    return external_call["mmap64", Pointer[UInt8, MutUntrackedOrigin]](Int64(0), size, Int32(1), Int32(1), fd, Int64(0))


def page_out(p: Pointer[UInt8, MutUntrackedOrigin], n: Int):
    var a = (Int(p) + 4095) // 4096 * 4096
    var b = (Int(p) + n) // 4096 * 4096
    if b > a:
        _ = external_call["madvise", Int32](a, b - a, Int32(21))


struct Engine:
    var ctx0: DeviceContext
    var ctx1: DeviceContext
    var bufs8: List[DeviceBuffer[DType.uint8]]
    var bufs32: List[DeviceBuffer[DType.int32]]
    var bufsf: List[DeviceBuffer[DType.float32]]
    var hosti: List[HostBuffer[DType.int32]]
    var hostf: List[HostBuffer[DType.float32]]
    var host8: List[HostBuffer[DType.uint8]]
    @__allow_legacy_any_origin_fields
    var gp: Pointer[PU8, MutAnyOrigin]
    @__allow_legacy_any_origin_fields
    var upp: Pointer[PU8, MutAnyOrigin]
    @__allow_legacy_any_origin_fields
    var dp: Pointer[PU8, MutAnyOrigin]
    @__allow_legacy_any_origin_fields
    var rp: Pointer[PF32, MutAnyOrigin]
    @__allow_legacy_any_origin_fields
    var tp: Pointer[PI32, MutAnyOrigin]
    @__allow_legacy_any_origin_fields
    var info: Pointer[Int, MutAnyOrigin]     # per layer: gpu, geb, deb, gq5k, dq80
    @__allow_legacy_any_origin_fields
    var g: Pointer[PF32, MutAnyOrigin]       # per GPU: logits, w, h, s, x, y, zero, hx(host), hy(host)  (9 each)
    @__allow_legacy_any_origin_fields
    var sl: Pointer[PI32, MutAnyOrigin]      # per GPU: slots, ring
    @__allow_legacy_any_origin_fields
    var pool: Pointer[PoolCtl, MutAnyOrigin]
    @__allow_legacy_any_origin_fields
    var tids: Pointer[UInt64, MutAnyOrigin]
    @__allow_legacy_any_origin_fields
    var b: Pointer[PF32, MutAnyOrigin]      # per GPU (10 each): X, logits, w, ids, slots, h, s, y, total | host: hx, hy
    @__allow_legacy_any_origin_fields
    var bh: Pointer[PF32, MutAnyOrigin]
    var bmaxt: Int
    var nstage: Int
    @__allow_legacy_any_origin_fields
    var st: Pointer[PU8, MutAnyOrigin]     # per GPU: stage gate, up, down (device) | host pinned: gate, up, down (shared)
    @__allow_legacy_any_origin_fields
    var sth: Pointer[PU8, MutAnyOrigin]
    @__allow_legacy_any_origin_fields
    var stmap: Pointer[PI32, MutAnyOrigin]  # per GPU: stage_of device [512] | host pinned stage_of [512], host pinned count
    var step: Int
    var calls: Int
    var call_ns: Int

    def __init__(out self, model_txt: String, pins_bin: String) raises:
        self.ctx0 = DeviceContext(device_id=0)
        self.ctx1 = DeviceContext(device_id=1)
        self.bufs8 = List[DeviceBuffer[DType.uint8]]()
        self.bufs32 = List[DeviceBuffer[DType.int32]]()
        self.bufsf = List[DeviceBuffer[DType.float32]]()
        self.hosti = List[HostBuffer[DType.int32]]()
        self.hostf = List[HostBuffer[DType.float32]]()
        self.host8 = List[HostBuffer[DType.uint8]]()
        self.gp = alloc[PU8](NL).as_unsafe_any_origin()
        self.upp = alloc[PU8](NL).as_unsafe_any_origin()
        self.dp = alloc[PU8](NL).as_unsafe_any_origin()
        self.rp = alloc[PF32](NL).as_unsafe_any_origin()
        self.tp = alloc[PI32](NL).as_unsafe_any_origin()
        self.info = alloc[Int](NL * 5).as_unsafe_any_origin()
        self.g = alloc[PF32](2 * 9).as_unsafe_any_origin()
        self.sl = alloc[PI32](2 * 2).as_unsafe_any_origin()
        var bm = getenv("OMX_BMAXT")
        self.bmaxt = Int(bm) if bm.byte_length() > 0 else BMAXT_DEFAULT
        self.pool = alloc[PoolCtl](1).as_unsafe_any_origin()
        self.pool.unsafe_write(PoolCtl(self.bmaxt))
        self.tids = alloc[UInt64](NW).as_unsafe_any_origin()
        self.b = alloc[PF32](2 * 10).as_unsafe_any_origin()
        self.st = alloc[PU8](2 * 3).as_unsafe_any_origin()
        self.sth = alloc[PU8](3).as_unsafe_any_origin()
        self.stmap = alloc[PI32](2 + 2).as_unsafe_any_origin()
        var ns = getenv("OMX_NSTAGE")
        self.nstage = Int(ns) if ns.byte_length() > 0 else 96
        self.bh = alloc[PF32](2 * 2).as_unsafe_any_origin()
        self.step = 0
        self.calls = 0
        self.call_ns = 0
        var t_init = perf_counter_ns()

        var mf = open(model_txt, "r")
        var text = mf.read()
        mf.close()
        var lines = text.splitlines()
        var files = lines[0].split(" ")
        var maps = List[Pointer[UInt8, MutUntrackedOrigin]]()
        for f in files:
            maps.append(mmap_file(String(f)))
        var pf = open(pins_bin, "r")
        var pins_b = pf.read_bytes()
        pf.close()
        var pins = pins_b.unsafe_ptr().bitcast[Int32]()
        var p = self.pool
        p[].nl = NL
        p[].total_steps = 1 << 62
        for l in range(NL):
            var t = lines[l + 1].split(" ")
            var gpu = Int(t[1])
            var nhot = max(Int(t[2]), 1)
            var gsrc = maps[Int(t[3])] + Int(t[4])
            var gq5 = t[5] == "Q5_K"
            var geb = Int(t[6])
            var usrc = maps[Int(t[7])] + Int(t[8])
            var dsrc = maps[Int(t[11])] + Int(t[12])
            var dq8 = t[13] == "Q8_0"
            var deb = Int(t[14])
            var rsrc = maps[Int(t[15])] + Int(t[16])
            self.info[l * 5 + 0] = gpu
            self.info[l * 5 + 1] = geb
            self.info[l * 5 + 2] = deb
            self.info[l * 5 + 3] = 1 if gq5 else 0
            self.info[l * 5 + 4] = 1 if dq8 else 0
            p[].gpu[l] = gpu
            p[].gate[l] = gsrc.as_unsafe_any_origin()
            p[].up[l] = usrc.as_unsafe_any_origin()
            p[].down[l] = dsrc.as_unsafe_any_origin()
            p[].geb[l] = geb
            p[].deb[l] = deb
            p[].gq5k[l] = gq5
            p[].dq80[l] = dq8
            var ctx = self.ctx0 if gpu == 0 else self.ctx1
            var hs_g = ctx.enqueue_create_host_buffer[DType.uint8](nhot * geb)
            var hs_u = ctx.enqueue_create_host_buffer[DType.uint8](nhot * geb)
            var hs_d = ctx.enqueue_create_host_buffer[DType.uint8](nhot * deb)
            var hs_r = ctx.enqueue_create_host_buffer[DType.uint8](N_EXPERTS * D_MODEL * 4)
            var hs_t = ctx.enqueue_create_host_buffer[DType.int32](N_EXPERTS)
            ctx.synchronize()
            var lbase = l * N_EXPERTS
            for e in range(N_EXPERTS):
                hs_t.unsafe_ptr()[e] = pins[lbase + e]
            var pg = hs_g.unsafe_ptr().as_unsafe_any_origin()
            var pu = hs_u.unsafe_ptr().as_unsafe_any_origin()
            var pd = hs_d.unsafe_ptr().as_unsafe_any_origin()

            def copy_expert(e: Int) {imm}:
                var s = Int(pins[lbase + e])
                if s >= 0:
                    unsafe_memcpy(dest=pg + s * geb, src=gsrc + e * geb, count=geb)
                    unsafe_memcpy(dest=pu + s * geb, src=usrc + e * geb, count=geb)
                    unsafe_memcpy(dest=pd + s * deb, src=dsrc + e * deb, count=deb)

            parallelize(copy_expert, N_EXPERTS)
            unsafe_memcpy(dest=hs_r.unsafe_ptr(), src=rsrc, count=N_EXPERTS * D_MODEL * 4)
            var dg = ctx.enqueue_create_buffer[DType.uint8](nhot * geb)
            var du = ctx.enqueue_create_buffer[DType.uint8](nhot * geb)
            var dd = ctx.enqueue_create_buffer[DType.uint8](nhot * deb)
            var dr = ctx.enqueue_create_buffer[DType.uint8](N_EXPERTS * D_MODEL * 4)
            var dt = ctx.enqueue_create_buffer[DType.int32](N_EXPERTS)
            ctx.enqueue_copy(dst_buf=dg, src_buf=hs_g)
            ctx.enqueue_copy(dst_buf=du, src_buf=hs_u)
            ctx.enqueue_copy(dst_buf=dd, src_buf=hs_d)
            ctx.enqueue_copy(dst_buf=dr, src_buf=hs_r)
            ctx.enqueue_copy(dst_buf=dt, src_buf=hs_t)
            ctx.synchronize()
            for e in range(N_EXPERTS):
                if Int(pins[lbase + e]) >= 0:
                    page_out(gsrc + e * geb, geb)
                    page_out(usrc + e * geb, geb)
                    page_out(dsrc + e * deb, deb)
            self.gp[l] = dg.unsafe_ptr().as_unsafe_any_origin()
            self.upp[l] = du.unsafe_ptr().as_unsafe_any_origin()
            self.dp[l] = dd.unsafe_ptr().as_unsafe_any_origin()
            self.rp[l] = dr.unsafe_ptr().bitcast[Float32]().as_unsafe_any_origin()
            self.tp[l] = dt.unsafe_ptr().as_unsafe_any_origin()
            self.bufs8.append(dg^)
            self.bufs8.append(du^)
            self.bufs8.append(dd^)
            self.bufs8.append(dr^)
            self.bufs32.append(dt^)
        for gi in range(2):
            var ctx = self.ctx0 if gi == 0 else self.ctx1
            var sizes = [N_EXPERTS, TOPK, TOPK * D_FF, TOPK * D_MODEL, D_MODEL, D_MODEL, D_MODEL]
            for j in range(7):
                var b = ctx.enqueue_create_buffer[DType.float32](sizes[j])
                if j == 6:
                    b.enqueue_fill(0.0)
                self.g[gi * 9 + j] = b.unsafe_ptr().as_unsafe_any_origin()
                self.bufsf.append(b^)
            for j in range(2):
                var hb = ctx.enqueue_create_host_buffer[DType.float32](D_MODEL)
                self.g[gi * 9 + 7 + j] = hb.unsafe_ptr().as_unsafe_any_origin()
                self.hostf.append(hb^)
            var sb = ctx.enqueue_create_buffer[DType.int32](TOPK + 1)
            self.sl[gi * 2 + 0] = sb.unsafe_ptr().as_unsafe_any_origin()
            self.bufs32.append(sb^)
            var rg = ctx.enqueue_create_host_buffer[DType.int32](RING * MB_WORDS)
            ctx.synchronize()
            for i in range(RING * MB_WORDS):
                rg.unsafe_ptr()[i] = 0
            self.sl[gi * 2 + 1] = rg.unsafe_ptr().as_unsafe_any_origin()
            self.hosti.append(rg^)
        p[].ring0 = self.sl[1]
        p[].ring1 = self.sl[3]
        # batch path buffers (prompt chunks, speculative verify): up to BMAXT tokens per call
        for gi in range(2):
            var ctx = self.ctx0 if gi == 0 else self.ctx1
            var BMAXT = self.bmaxt
            var bsizes = [BMAXT * D_MODEL, BMAXT * N_EXPERTS, BMAXT * BW, BMAXT * BW, BMAXT * BW, BMAXT * TOPK * D_FF, BMAXT * TOPK * D_MODEL, BMAXT * D_MODEL, 16]
            for j in range(9):
                var bb = ctx.enqueue_create_buffer[DType.float32](bsizes[j])
                self.b[gi * 10 + j] = bb.unsafe_ptr().as_unsafe_any_origin()
                self.bufsf.append(bb^)
            for j in range(2):
                var hb = ctx.enqueue_create_host_buffer[DType.float32](self.bmaxt * D_MODEL)
                self.bh[gi * 2 + j] = hb.unsafe_ptr().as_unsafe_any_origin()
                self.hostf.append(hb^)
        # staging pools for cold experts uploaded for a big batch (max expert sizes: Q5_K gate/up, Q8_0 down)
        comptime GEB_MAX = 1126400
        comptime DEB_MAX = 1740800
        for gi in range(2):
            var ctx = self.ctx0 if gi == 0 else self.ctx1
            var sizes8 = [self.nstage * GEB_MAX, self.nstage * GEB_MAX, self.nstage * DEB_MAX]
            for j in range(3):
                var sb8 = ctx.enqueue_create_buffer[DType.uint8](max(sizes8[j], 1))
                self.st[gi * 3 + j] = sb8.unsafe_ptr().as_unsafe_any_origin()
                self.bufs8.append(sb8^)
            var smap = ctx.enqueue_create_buffer[DType.int32](N_EXPERTS)
            self.stmap[gi] = smap.unsafe_ptr().as_unsafe_any_origin()
            self.bufs32.append(smap^)
        var sizesh = [self.nstage * GEB_MAX, self.nstage * GEB_MAX, self.nstage * DEB_MAX]
        for j in range(3):
            var hb8 = self.ctx0.enqueue_create_host_buffer[DType.uint8](max(sizesh[j], 1))
            self.sth[j] = hb8.unsafe_ptr().as_unsafe_any_origin()
            self.hosti.append(self.ctx0.enqueue_create_host_buffer[DType.int32](1))   # keep list types simple
            self.host8.append(hb8^)
        var hmap = self.ctx0.enqueue_create_host_buffer[DType.int32](N_EXPERTS + 16)
        self.stmap[2] = hmap.unsafe_ptr().as_unsafe_any_origin()
        self.stmap[3] = hmap.unsafe_ptr().as_unsafe_any_origin() + N_EXPERTS
        self.hosti.append(hmap^)
        var ba = self.ctx0.enqueue_create_host_buffer[DType.int32](1 + 3 * self.bmaxt * TOPK)
        var br = self.ctx0.enqueue_create_host_buffer[DType.float32](self.bmaxt * TOPK * D_MODEL)
        self.ctx0.synchronize()
        p[].barea = ba.unsafe_ptr().as_unsafe_any_origin()
        p[].bres = br.unsafe_ptr().as_unsafe_any_origin()
        self.hosti.append(ba^)
        self.hostf.append(br^)
        # load every kernel variant before anything can spin (module loads wait for an idle device)
        for gi in range(2):
            var ctx = self.ctx0 if gi == 0 else self.ctx1
            var dsl = ctx.enqueue_create_host_buffer[DType.int32](TOPK + 1)
            var dmb = ctx.enqueue_create_host_buffer[DType.int32](MB_WORDS)
            ctx.synchronize()
            for i in range(TOPK):
                dsl.unsafe_ptr()[i] = -1
            dsl.unsafe_ptr()[TOPK] = 0
            var dsp = dsl.unsafe_ptr().as_unsafe_any_origin()
            var dmp = dmb.unsafe_ptr().as_unsafe_any_origin()
            var l0 = 0 if gi == 0 else 24
            var G = gi * 9
            ctx.enqueue_function[router_kernel](self.rp[l0], self.g[G + 4], self.g[G + 0], grid_dim=N_EXPERTS // WARPS_PER_BLOCK, block_dim=TPB)
            ctx.enqueue_function[topk_slots_kernel](self.g[G + 0], self.tp[l0], self.g[G + 4], self.g[G + 1], self.g[G + 2].bitcast[Int32](), dmp, Int32(-3),
                grid_dim=1, block_dim=N_EXPERTS)
            ctx.enqueue_function[gate_up_slots[0]](self.gp[l0], self.upp[l0], dsp, Int64(self.info[l0 * 5 + 1]), self.g[G + 4], self.g[G + 2],
                grid_dim=ceildiv(TOPK * D_FF, WARPS_PER_BLOCK), block_dim=TPB)
            ctx.enqueue_function[gate_up_slots[1]](self.gp[l0], self.upp[l0], dsp, Int64(self.info[l0 * 5 + 1]), self.g[G + 4], self.g[G + 2],
                grid_dim=ceildiv(TOPK * D_FF, WARPS_PER_BLOCK), block_dim=TPB)
            ctx.enqueue_function[down_slots](self.dp[l0], Int64(0), dsp, Int64(self.info[l0 * 5 + 2]), self.g[G + 2], self.g[G + 3],
                grid_dim=D_MODEL // DROWS, block_dim=TOPK * 32)
            ctx.enqueue_function[combine_res_kernel](self.g[G + 3], dmp, self.g[G + 1], dsp, self.g[G + 6], self.g[G + 5], Int32(-3),
                grid_dim=ceildiv(D_MODEL, 256), block_dim=256)
            var B = gi * 10
            var bsl = self.b[B + 4].bitcast[Int32]()
            ctx.enqueue_function[router_batch_kernel](self.rp[l0], self.b[B + 0], self.b[B + 1], grid_dim=N_EXPERTS // WARPS_PER_BLOCK, block_dim=TPB)
            ctx.enqueue_function[topk_batch_kernel](self.b[B + 1], self.tp[l0], self.b[B + 2], self.b[B + 3].bitcast[Int32](), bsl, grid_dim=1, block_dim=N_EXPERTS)
            var dba = ctx.enqueue_create_host_buffer[DType.int32](1 + 3 * TOPK)
            ctx.synchronize()
            ctx.enqueue_function[collect_cold_kernel](self.b[B + 3].bitcast[Int32](), dsp, Int64(1), dba.unsafe_ptr().as_unsafe_any_origin(), dmp, Int32(-3), self.b[B + 8].bitcast[Int32](), Int64(1), grid_dim=1, block_dim=32)
            ctx.enqueue_function[gate_up_batch[0]](self.gp[l0], self.upp[l0], dsp, Int64(self.info[l0 * 5 + 1]), self.b[B + 0], self.b[B + 5], self.st[gi * 3 + 0], self.st[gi * 3 + 1],
                grid_dim=ceildiv(TOPK * D_FF, WARPS_PER_BLOCK), block_dim=TPB)
            ctx.enqueue_function[gate_up_batch[1]](self.gp[l0], self.upp[l0], dsp, Int64(self.info[l0 * 5 + 1]), self.b[B + 0], self.b[B + 5], self.st[gi * 3 + 0], self.st[gi * 3 + 1],
                grid_dim=ceildiv(TOPK * D_FF, WARPS_PER_BLOCK), block_dim=TPB)
            ctx.enqueue_function[down_batch](self.dp[l0], Int64(0), dsp, Int64(self.info[l0 * 5 + 2]), self.b[B + 5], self.b[B + 6], self.st[gi * 3 + 2],
                grid_dim=D_MODEL // DROWS, block_dim=TOPK * 32)
            ctx.enqueue_function[remap_stage_kernel](self.b[B + 3].bitcast[Int32](), dsp, self.stmap[gi], Int64(0), grid_dim=1, block_dim=32)
            ctx.enqueue_function[wait_done_kernel](dmp, self.b[B + 8].bitcast[Int32](), Int32(-3), grid_dim=1, block_dim=32)
            ctx.enqueue_function[combine_batch_kernel](self.b[B + 6], p[].bres, self.b[B + 2], dsp, self.b[B + 7],
                grid_dim=D_MODEL // 256, block_dim=256)
            ctx.synchronize()
        start_pool(self.pool, self.tids)
        print("omx: Mojo expert layer ready,", Float64(perf_counter_ns() - t_init) / 1e9, "s to load hot experts")

    def moe(mut self, layer: Int, x: Pointer[Float32, MutAnyOrigin], y: Pointer[Float32, MutAnyOrigin]) raises:
        var t0 = perf_counter_ns()
        var gi = self.info[layer * 5 + 0]
        var ctx = self.ctx0 if gi == 0 else self.ctx1
        var G = gi * 9
        var seq = Int32(self.step + 1)
        self.pool[].steplayer[self.step & 4095] = layer
        self.step += 1
        self.calls += 1
        unsafe_memcpy(dest=self.g[G + 7], src=x, count=D_MODEL)
        ctx.enqueue_copy(self.g[G + 4], self.g[G + 7], D_MODEL)
        var mb = self.sl[gi * 2 + 1] + (Int(seq) % RING) * MB_WORDS
        var slots = self.sl[gi * 2 + 0]
        ctx.enqueue_function[router_kernel](self.rp[layer], self.g[G + 4], self.g[G + 0], grid_dim=N_EXPERTS // WARPS_PER_BLOCK, block_dim=TPB)
        ctx.enqueue_function[topk_slots_kernel](self.g[G + 0], self.tp[layer], self.g[G + 4], self.g[G + 1], slots, mb, seq,
            grid_dim=1, block_dim=N_EXPERTS)
        if self.info[layer * 5 + 3] == 1:
            ctx.enqueue_function[gate_up_slots[1]](self.gp[layer], self.upp[layer], slots, Int64(self.info[layer * 5 + 1]), self.g[G + 4], self.g[G + 2],
                grid_dim=ceildiv(TOPK * D_FF, WARPS_PER_BLOCK), block_dim=TPB)
        else:
            ctx.enqueue_function[gate_up_slots[0]](self.gp[layer], self.upp[layer], slots, Int64(self.info[layer * 5 + 1]), self.g[G + 4], self.g[G + 2],
                grid_dim=ceildiv(TOPK * D_FF, WARPS_PER_BLOCK), block_dim=TPB)
        ctx.enqueue_function[down_slots](self.dp[layer], Int64(self.info[layer * 5 + 4]), slots, Int64(self.info[layer * 5 + 2]), self.g[G + 2], self.g[G + 3],
            grid_dim=D_MODEL // DROWS, block_dim=TOPK * 32)
        ctx.enqueue_function[combine_res_kernel](self.g[G + 3], mb, self.g[G + 1], slots, self.g[G + 6], self.g[G + 5], seq,
            grid_dim=ceildiv(D_MODEL, 256), block_dim=256)
        ctx.enqueue_copy(self.g[G + 8], self.g[G + 5], D_MODEL)
        ctx.synchronize()
        unsafe_memcpy(dest=y, src=self.g[G + 8], count=D_MODEL)
        self.call_ns += perf_counter_ns() - t0
        self.report()

    def report(mut self):
        if self.calls % (NL * 64) == 0 and self.calls > 0:
            print("omx:", self.calls // NL, "layer-tokens/48,", Float64(self.call_ns) / Float64(NL * 64) / 1e3, "us per call avg,",
                  "cold picks served", Atomic.load(self.pool[].ctl + C_COLD))
            self.call_ns = 0

    def moe_batch(mut self, layer: Int, x: Pointer[Float32, MutAnyOrigin], y: Pointer[Float32, MutAnyOrigin], ntok: Int) raises:
        """ntok tokens through one layer: each hot expert's rows run for all tokens that picked it, each cold expert is
        read once by the CPU for all its tokens. Same bits as ntok calls of moe()."""
        var done = 0
        while done < ntok:
            var T = min(self.bmaxt, ntok - done)
            self.moe_chunk(layer, x + done * D_MODEL, y + done * D_MODEL, T)
            done += T

    def moe_chunk(mut self, layer: Int, x: Pointer[Float32, MutAnyOrigin], y: Pointer[Float32, MutAnyOrigin], T: Int) raises:
        var t0 = perf_counter_ns()
        var gi = self.info[layer * 5 + 0]
        var ctx = self.ctx0 if gi == 0 else self.ctx1
        var B = gi * 10
        var seq = Int32(self.step + 1)
        var p = self.pool
        p[].steplayer[self.step & 4095] = layer
        p[].bx = x
        self.step += 1
        self.calls += T
        var mb = self.sl[gi * 2 + 1] + (Int(seq) % RING) * MB_WORDS
        var slots = self.b[B + 4].bitcast[Int32]()
        var ids = self.b[B + 3].bitcast[Int32]()
        var total = self.b[B + 8].bitcast[Int32]()
        var geb = self.info[layer * 5 + 1]
        var deb = self.info[layer * 5 + 2]
        var upload = T >= 64 and self.nstage > 0
        unsafe_memcpy(dest=self.bh[gi * 2 + 0], src=x, count=T * D_MODEL)
        ctx.enqueue_copy(self.b[B + 0], self.bh[gi * 2 + 0], T * D_MODEL)
        ctx.enqueue_function[router_batch_kernel](self.rp[layer], self.b[B + 0], self.b[B + 1], grid_dim=T * (N_EXPERTS // WARPS_PER_BLOCK), block_dim=TPB)
        ctx.enqueue_function[topk_batch_kernel](self.b[B + 1], self.tp[layer], self.b[B + 2], ids, slots,
            grid_dim=T, block_dim=N_EXPERTS)
        ctx.enqueue_function[collect_cold_kernel](ids, slots, Int64(T), p[].barea, mb, seq, total, Int64(0 if upload else 1),
            grid_dim=1, block_dim=32)
        if upload:
            # dual path: the experts with the most tokens go to the GPU over PCIe, the rest to the CPU pool
            ctx.synchronize()
            var ba = p[].barea
            var n = Int(ba[0])
            var cnt = alloc[Int](N_EXPERTS)
            for e in range(N_EXPERTS):
                cnt[e] = 0
            for i in range(n):
                cnt[Int(ba[1 + 3 * i])] += 1
            var smap = self.stmap[2]
            for e in range(N_EXPERTS):
                smap[e] = -1
            var chosen = List[Int]()
            var kmin = 3
            while len(chosen) < self.nstage:      # greedy: most tokens first
                var best = -1
                var bc = kmin - 1
                for e in range(N_EXPERTS):
                    if smap[e] < 0 and cnt[e] > bc:
                        bc = cnt[e]
                        best = e
                if best < 0:
                    break
                smap[best] = Int32(len(chosen))
                chosen.append(best)
            # the CPU's share: keep only picks of experts not uploaded
            var m = 0
            for i in range(n):
                var e = Int(ba[1 + 3 * i])
                if smap[e] < 0:
                    ba[1 + 3 * m] = ba[1 + 3 * i]
                    ba[2 + 3 * m] = ba[2 + 3 * i]
                    ba[3 + 3 * m] = ba[3 + 3 * i]
                    m += 1
            ba[0] = Int32(m)
            self.stmap[3][0] = Int32(m)
            mb.unsafe_store[volatile=True](MB_NCOLD, Int32(-1) if m > 0 else Int32(0))
            mb.unsafe_store[volatile=True](MB_SEQ, seq)       # the CPU pool starts now
            var nc = len(chosen)
            if nc > 0:
                var gsrc = p[].gate[layer]
                var usrc = p[].up[layer]
                var dsrc = p[].down[layer]
                var hg = self.sth[0]
                var hu = self.sth[1]
                var hd = self.sth[2]
                var ch = alloc[Int](nc)
                for i in range(nc):
                    ch[i] = chosen[i]

                def stage_copy(i: Int) {imm}:
                    var e = ch[i]
                    unsafe_memcpy(dest=hg + i * geb, src=gsrc + e * geb, count=geb)
                    unsafe_memcpy(dest=hu + i * geb, src=usrc + e * geb, count=geb)
                    unsafe_memcpy(dest=hd + i * deb, src=dsrc + e * deb, count=deb)

                parallelize(stage_copy, nc)
                ch.free()
                ctx.enqueue_copy(self.st[gi * 3 + 0], self.sth[0], nc * geb)
                ctx.enqueue_copy(self.st[gi * 3 + 1], self.sth[1], nc * geb)
                ctx.enqueue_copy(self.st[gi * 3 + 2], self.sth[2], nc * deb)
            ctx.enqueue_copy(self.stmap[gi], smap, N_EXPERTS)
            ctx.enqueue_copy(total, self.stmap[3], 1)
            ctx.enqueue_function[remap_stage_kernel](ids, slots, self.stmap[gi], Int64(T), grid_dim=ceildiv(T * TOPK, 256), block_dim=256)
            cnt.free()
        if self.info[layer * 5 + 3] == 1:
            ctx.enqueue_function[gate_up_batch[1]](self.gp[layer], self.upp[layer], slots, Int64(geb), self.b[B + 0], self.b[B + 5], self.st[gi * 3 + 0], self.st[gi * 3 + 1],
                grid_dim=ceildiv(T * TOPK * D_FF, WARPS_PER_BLOCK), block_dim=TPB)
        else:
            ctx.enqueue_function[gate_up_batch[0]](self.gp[layer], self.upp[layer], slots, Int64(geb), self.b[B + 0], self.b[B + 5], self.st[gi * 3 + 0], self.st[gi * 3 + 1],
                grid_dim=ceildiv(T * TOPK * D_FF, WARPS_PER_BLOCK), block_dim=TPB)
        ctx.enqueue_function[down_batch](self.dp[layer], Int64(self.info[layer * 5 + 4]), slots, Int64(deb), self.b[B + 5], self.b[B + 6], self.st[gi * 3 + 2],
            grid_dim=T * (D_MODEL // DROWS), block_dim=TOPK * 32)
        ctx.enqueue_function[wait_done_kernel](mb, total, seq, grid_dim=1, block_dim=32)
        ctx.enqueue_function[combine_batch_kernel](self.b[B + 6], p[].bres, self.b[B + 2], slots, self.b[B + 7],
            grid_dim=T * (D_MODEL // 256), block_dim=256)
        ctx.enqueue_copy(self.bh[gi * 2 + 1], self.b[B + 7], T * D_MODEL)
        ctx.synchronize()
        unsafe_memcpy(dest=y, src=self.bh[gi * 2 + 1], count=T * D_MODEL)
        self.call_ns += perf_counter_ns() - t0


@export("omx_init")
def omx_init(model_txt: Pointer[UInt8, MutUntrackedOrigin], pins_bin: Pointer[UInt8, MutUntrackedOrigin]) abi("C") -> Int64:
    """Returns an opaque handle (0 on failure)."""
    initialize_runtime()
    try:
        var e = alloc[Engine](1).as_unsafe_any_origin()
        e.unsafe_write(Engine(String(unsafe_from_utf8_ptr=model_txt), String(unsafe_from_utf8_ptr=pins_bin)))
        return Int64(Int(e))
    except err:
        print("omx_init failed:", err)
        return 0


def _engine(h: Int64) -> Pointer[Engine, MutAnyOrigin]:
    return Pointer[Engine, MutAnyOrigin](unsafe_from_address=Int(h))


@export("omx_moe")
def omx_moe(h: Int64, layer: Int32, x: Pointer[Float32, MutAnyOrigin], y: Pointer[Float32, MutAnyOrigin]) abi("C") -> Int32:
    try:
        _engine(h)[].moe(Int(layer), x, y)
        return 0
    except err:
        print("omx_moe failed:", err)
        return -1


@export("omx_stats")
def omx_stats(h: Int64, out3: Pointer[Float64, MutAnyOrigin]) abi("C"):
    var e = _engine(h)
    out3[0] = Float64(e[].calls)
    out3[1] = Float64(Atomic.load(e[].pool[].ctl + C_COLD))
    out3[2] = Float64(Atomic.load(e[].pool[].ctl + C_BUSY_NS)) / 1e6


@export("omx_moe_batch")
def omx_moe_batch(h: Int64, layer: Int32, x: Pointer[Float32, MutAnyOrigin], y: Pointer[Float32, MutAnyOrigin], ntok: Int32) abi("C") -> Int32:
    try:
        _engine(h)[].moe_batch(Int(layer), x, y, Int(ntok))
        return 0
    except err:
        print("omx_moe_batch failed:", err)
        return -1
