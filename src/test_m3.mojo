# M3 proof: the hybrid layer (router + top-k on the GPU, hot experts on the GPU, cold experts on the CPU through the
# pinned-memory mailbox, no host sync per layer) gives output BIT-IDENTICAL to the same layer run all-GPU.
# args: gate_file gate_off up_file up_off down_file down_off down_is_q80 eb_gu eb_d router_file router_off n_tokens cold_pct
from std.sys import argv
from std.time import perf_counter_ns
from std.random import random_float64, seed
from std.math import ceildiv
from std.memory import alloc
from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from hybrid import (router_kernel, topk_kernel, gate_up_hot, down_hot, combine_kernel, DROWS, N_EXPERTS, TOPK,
                    MB_SEQ, MB_DONE, MB_NCOLD, MB_ERR, MB_SLOT, MB_ID, MB_X, MB_S, MB_WORDS)
from gpu_moe import D_MODEL, D_FF, WARPS_PER_BLOCK, TPB
from cpu_moe import moe_cpu, HV_PER_EXPERT

comptime WINDOW = 8   # tokens enqueued ahead of the CPU service loop
comptime RING = 16    # mailbox slots (> WINDOW): the GPU never reuses a slot the CPU has not read yet


def fill_host(host: HostBuffer[DType.uint8], path: String, off: Int, n: Int) raises:
    var f = open(path, "r")
    _ = f.seek(off)
    var done = 0
    var p = host.unsafe_ptr()
    while done < n:
        var b = f.read_bytes(min(1 << 26, n - done))
        for i in range(len(b)):
            p[done + i] = b[i]
        done += len(b)
    f.close()


def main() raises:
    var a = argv()
    var eb_gu = Int(a[8])
    var eb_d = Int(a[9])
    var q80 = Int(a[7])
    var n_tok = Int(a[12])
    var cold_pct = Int(a[13])
    with DeviceContext() as ctx:
        var hg = ctx.enqueue_create_host_buffer[DType.uint8](N_EXPERTS * eb_gu)
        var hu = ctx.enqueue_create_host_buffer[DType.uint8](N_EXPERTS * eb_gu)
        var hd = ctx.enqueue_create_host_buffer[DType.uint8](N_EXPERTS * eb_d)
        var hr = ctx.enqueue_create_host_buffer[DType.uint8](N_EXPERTS * D_MODEL * 4)
        ctx.synchronize()
        fill_host(hg, String(a[1]), Int(a[2]), N_EXPERTS * eb_gu)
        fill_host(hu, String(a[3]), Int(a[4]), N_EXPERTS * eb_gu)
        fill_host(hd, String(a[5]), Int(a[6]), N_EXPERTS * eb_d)
        fill_host(hr, String(a[10]), Int(a[11]), N_EXPERTS * D_MODEL * 4)
        var gate = ctx.enqueue_create_buffer[DType.uint8](N_EXPERTS * eb_gu)
        var up = ctx.enqueue_create_buffer[DType.uint8](N_EXPERTS * eb_gu)
        var down = ctx.enqueue_create_buffer[DType.uint8](N_EXPERTS * eb_d)
        var wr = ctx.enqueue_create_buffer[DType.uint8](N_EXPERTS * D_MODEL * 4)
        ctx.enqueue_copy(dst_buf=gate, src_buf=hg)
        ctx.enqueue_copy(dst_buf=up, src_buf=hu)
        ctx.enqueue_copy(dst_buf=down, src_buf=hd)
        ctx.enqueue_copy(dst_buf=wr, src_buf=hr)

        # residency tables: all hot, and ~cold_pct % cold (fixed hash, so the same experts every token)
        var hres = ctx.enqueue_create_host_buffer[DType.int32](2 * N_EXPERTS)
        var hx = ctx.enqueue_create_host_buffer[DType.float32](n_tok * D_MODEL)
        ctx.synchronize()
        for e in range(N_EXPERTS):
            hres.unsafe_ptr()[e] = 1
            hres.unsafe_ptr()[N_EXPERTS + e] = 0 if (e * 2654435761 % 1000003) % 100 < cold_pct else 1
        seed(7)
        for i in range(n_tok * D_MODEL):
            hx.unsafe_ptr()[i] = Float32(random_float64(-0.5, 0.5))
        var res = ctx.enqueue_create_buffer[DType.int32](2 * N_EXPERTS)
        var x = ctx.enqueue_create_buffer[DType.float32](n_tok * D_MODEL)
        ctx.enqueue_copy(dst_buf=res, src_buf=hres)
        ctx.enqueue_copy(dst_buf=x, src_buf=hx)
        var logits = ctx.enqueue_create_buffer[DType.float32](N_EXPERTS)
        var ids = ctx.enqueue_create_buffer[DType.int32](TOPK)
        var w = ctx.enqueue_create_buffer[DType.float32](TOPK)
        var hot = ctx.enqueue_create_buffer[DType.int32](TOPK + 1)
        var h = ctx.enqueue_create_buffer[DType.float32](TOPK * D_FF)
        var s = ctx.enqueue_create_buffer[DType.float32](TOPK * D_MODEL)
        var y = ctx.enqueue_create_buffer[DType.float32](2 * n_tok * D_MODEL)
        var mbh = ctx.enqueue_create_host_buffer[DType.int32](RING * MB_WORDS)
        ctx.synchronize()
        var mb0 = mbh.unsafe_ptr()
        for i in range(RING * MB_WORDS):
            mb0[i] = 0
        # CPU scratch
        var xp = alloc[Float32](D_MODEL)
        var hc = alloc[Float32](TOPK * D_FF)
        var hvc = alloc[Float32](TOPK * HV_PER_EXPERT)
        var sc = alloc[Float32](TOPK * D_MODEL)
        var yc = alloc[Float32](D_MODEL)
        var wdummy = alloc[Float32](TOPK)
        var off_gu = alloc[Int](TOPK)
        var off_d = alloc[Int](TOPK)
        var cold_hist = alloc[Int](TOPK + 1)

        @parameter
        def enqueue(t: Int, seq: Int32, rsel: Int, ysel: Int) raises:
            var xt = x.unsafe_ptr() + t * D_MODEL
            var mb = mb0 + (Int(seq) % RING) * MB_WORDS
            ctx.enqueue_function[router_kernel](wr.unsafe_ptr().bitcast[Float32](), xt, logits,
                grid_dim=N_EXPERTS // WARPS_PER_BLOCK, block_dim=TPB)
            ctx.enqueue_function[topk_kernel](logits, res.unsafe_ptr() + rsel * N_EXPERTS, xt, ids, w, hot, mb, seq,
                grid_dim=1, block_dim=N_EXPERTS)
            ctx.enqueue_function[gate_up_hot](gate, up, ids, hot, Int64(eb_gu), xt, h,
                grid_dim=ceildiv(TOPK * D_FF, WARPS_PER_BLOCK), block_dim=TPB)
            ctx.enqueue_function[down_hot](down, Int64(q80), ids, hot, Int64(eb_d), h, s,
                grid_dim=D_MODEL // DROWS, block_dim=TOPK * 32)
            ctx.enqueue_function[combine_kernel](s, mb, w, hot, y.unsafe_ptr() + (ysel * n_tok + t) * D_MODEL, seq,
                grid_dim=ceildiv(D_MODEL, 256), block_dim=256)

        @parameter
        def run(seq0: Int, rsel: Int, ysel: Int) raises -> Int:
            for c in range(TOPK + 1):
                cold_hist[c] = 0
            var t0 = perf_counter_ns()
            for t in range(min(WINDOW, n_tok)):
                enqueue(t, Int32(seq0 + t + 1), rsel, ysel)
            for t in range(n_tok):
                var seq = Int32(seq0 + t + 1)
                var mb = mb0 + (Int(seq) % RING) * MB_WORDS
                var spins = 0
                while mb.unsafe_load[volatile=True](MB_SEQ)[0] != seq:
                    spins += 1
                    if spins > 2_000_000_000:
                        raise Error("CPU waited too long for token " + String(seq))
                var ncold = Int(mb.unsafe_load[volatile=True](MB_NCOLD)[0])
                cold_hist[ncold] += 1
                if ncold > 0:
                    for j in range(ncold):
                        var id = Int(mb.unsafe_load[volatile=True](MB_ID + j)[0])
                        off_gu[j] = id * eb_gu
                        off_d[j] = id * eb_d
                    moe_cpu(hg.unsafe_ptr(), hu.unsafe_ptr(), hd.unsafe_ptr(), q80 == 1, off_gu, off_d, ncold,
                            wdummy, mb.bitcast[Float32]() + MB_X, xp, hc, hvc, sc, yc)
                    var mbf = mb.bitcast[Float32]()
                    for j in range(ncold):
                        var slot = Int(mb.unsafe_load[volatile=True](MB_SLOT + j)[0])
                        for i in range(D_MODEL):
                            mbf[MB_S + slot * D_MODEL + i] = sc[j * D_MODEL + i]
                mb.unsafe_store[volatile=True](MB_DONE, seq)
                if t + WINDOW < n_tok:
                    enqueue(t + WINDOW, Int32(seq0 + t + WINDOW + 1), rsel, ysel)
            ctx.synchronize()
            return perf_counter_ns() - t0

        _ = run(0, 0, 0)                          # warm-up (also the all-GPU reference)
        var t_gpu = run(n_tok, 0, 0)              # all-GPU
        var t_hyb = run(2 * n_tok, 1, 1)          # hybrid
        var hy = ctx.enqueue_create_host_buffer[DType.float32](2 * n_tok * D_MODEL)
        ctx.enqueue_copy(dst_buf=hy, src_buf=y)
        ctx.synchronize()
        var same = 0
        for t in range(n_tok):
            var ok = True
            for i in range(D_MODEL):
                if hy.unsafe_ptr()[t * D_MODEL + i].to_bits() != hy.unsafe_ptr()[(n_tok + t) * D_MODEL + i].to_bits():
                    ok = False
            if ok:
                same += 1
        var hist = String("")
        for c in range(TOPK + 1):
            hist += String(cold_hist[c]) + (" " if c < TOPK else "")
        print("{\"tokens\": ", n_tok, ", \"bit_identical\": ", same, ", \"all_gpu_us\": ", Float64(t_gpu) / Float64(n_tok) / 1000.0,
              ", \"hybrid_us\": ", Float64(t_hyb) / Float64(n_tok) / 1000.0, ", \"cold_pct\": ", cold_pct,
              ", \"cold_hist\": \"", hist, "\"}")
