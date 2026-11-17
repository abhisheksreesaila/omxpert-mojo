# M2 proof: the whole layer (512 experts, 1.57 GB) in RAM and in VRAM; 200 random tokens (x, sorted top-10 ids,
# weights) run on BOTH devices from the same qexperts.mojo source; every output float must match bit for bit.
# Also times the CPU over those 200 random routings (the layer is 25x the L3, so this is RAM speed, not cache).
# args: gate_file gate_off up_file up_off down_file down_off down_is_q80 eb_gu eb_d n_cases
from std.sys import argv
from std.time import perf_counter_ns
from std.random import random_ui64, random_float64, seed
from std.math import ceildiv
from std.memory import alloc
from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from gpu_moe import gate_up_kernel_reg, down_kernel_par, GU_ROWS, ROWS_PER_BLOCK, WARPS_PER_BLOCK, TPB, D_MODEL, D_FF
from std.runtime import parallelism_level
from cpu_moe import moe_cpu, moe_cpu_fused, HV_PER_EXPERT

comptime K = 10
comptime N_EXPERTS = 512


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
    var n_cases = Int(a[10])
    with DeviceContext() as ctx:
        var hg = ctx.enqueue_create_host_buffer[DType.uint8](N_EXPERTS * eb_gu)
        var hu = ctx.enqueue_create_host_buffer[DType.uint8](N_EXPERTS * eb_gu)
        var hd = ctx.enqueue_create_host_buffer[DType.uint8](N_EXPERTS * eb_d)
        ctx.synchronize()
        fill_host(hg, String(a[1]), Int(a[2]), N_EXPERTS * eb_gu)
        fill_host(hu, String(a[3]), Int(a[4]), N_EXPERTS * eb_gu)
        fill_host(hd, String(a[5]), Int(a[6]), N_EXPERTS * eb_d)
        var gate = ctx.enqueue_create_buffer[DType.uint8](N_EXPERTS * eb_gu)
        var up = ctx.enqueue_create_buffer[DType.uint8](N_EXPERTS * eb_gu)
        var down = ctx.enqueue_create_buffer[DType.uint8](N_EXPERTS * eb_d)
        ctx.enqueue_copy(dst_buf=gate, src_buf=hg)
        ctx.enqueue_copy(dst_buf=up, src_buf=hu)
        ctx.enqueue_copy(dst_buf=down, src_buf=hd)

        # cases
        seed(42)
        var hx = ctx.enqueue_create_host_buffer[DType.float32](n_cases * D_MODEL)
        var hids = ctx.enqueue_create_host_buffer[DType.int32](n_cases * K)
        var hw = ctx.enqueue_create_host_buffer[DType.float32](n_cases * K)
        var hy = ctx.enqueue_create_host_buffer[DType.float32](n_cases * D_MODEL)
        ctx.synchronize()
        for c in range(n_cases):
            for i in range(D_MODEL):
                hx.unsafe_ptr()[c * D_MODEL + i] = Float32(random_float64(-3.0, 3.0))
            var picked = List[Int]()
            while len(picked) < K:
                var e = Int(random_ui64(0, N_EXPERTS - 1))
                if e not in picked:
                    picked.append(e)
            sort(picked)
            var tot: Float32 = 0
            for e in range(K):
                hids.unsafe_ptr()[c * K + e] = Int32(picked[e])
                var wv = Float32(random_float64(0.01, 1.0))
                hw.unsafe_ptr()[c * K + e] = wv
                tot += wv
            for e in range(K):
                hw.unsafe_ptr()[c * K + e] /= tot
        var x = ctx.enqueue_create_buffer[DType.float32](n_cases * D_MODEL)
        var ids = ctx.enqueue_create_buffer[DType.int32](n_cases * K)
        var w = ctx.enqueue_create_buffer[DType.float32](n_cases * K)
        var h = ctx.enqueue_create_buffer[DType.float32](K * D_FF)
        var y = ctx.enqueue_create_buffer[DType.float32](n_cases * D_MODEL)
        ctx.enqueue_copy(dst_buf=x, src_buf=hx)
        ctx.enqueue_copy(dst_buf=ids, src_buf=hids)
        ctx.enqueue_copy(dst_buf=w, src_buf=hw)

        # GPU: all cases
        for c in range(n_cases):
            var ip = ids.unsafe_ptr() + c * K
            ctx.enqueue_function[gate_up_kernel_reg](gate, up, ip, Int64(eb_gu), Int64(K), x.unsafe_ptr() + c * D_MODEL, h,
                grid_dim=ceildiv(K * D_FF // GU_ROWS, WARPS_PER_BLOCK), block_dim=TPB)
            ctx.enqueue_function[down_kernel_par](down, Int64(q80), ip, w.unsafe_ptr() + c * K, Int64(eb_d), Int64(K), h,
                y.unsafe_ptr() + c * D_MODEL, grid_dim=D_MODEL // ROWS_PER_BLOCK, block_dim=K * 32)
        ctx.enqueue_copy(dst_buf=hy, src_buf=y)
        ctx.synchronize()

        # CPU: all cases, timed
        var xp = alloc[Float32](D_MODEL)
        var hc = alloc[Float32](K * D_FF)
        var hvc = alloc[Float32](K * HV_PER_EXPERT)
        var s = alloc[Float32](K * D_MODEL)
        var yc = alloc[Float32](n_cases * D_MODEL)
        var off_gu = alloc[Int](K)
        var off_d = alloc[Int](K)
        var bar = alloc[Int64](2)
        var t0 = perf_counter_ns()
        for c in range(n_cases):
            for e in range(K):
                off_gu[e] = Int(hids.unsafe_ptr()[c * K + e]) * eb_gu
                off_d[e] = Int(hids.unsafe_ptr()[c * K + e]) * eb_d
            moe_cpu(hg.unsafe_ptr(), hu.unsafe_ptr(), hd.unsafe_ptr(), q80 == 1, off_gu, off_d, K,
                    hw.unsafe_ptr() + c * K, hx.unsafe_ptr() + c * D_MODEL, xp, hc, hvc, s, yc + c * D_MODEL)
        var dt = perf_counter_ns() - t0
        var same_cases = 0
        var diff_floats = 0
        for c in range(n_cases):
            var ok = True
            for i in range(D_MODEL):
                var g = hy.unsafe_ptr()[c * D_MODEL + i]
                var cc = yc[c * D_MODEL + i]
                if g.to_bits() != cc.to_bits():
                    ok = False
                    diff_floats += 1
            if ok:
                same_cases += 1
        var tst = List[Int]()
        for st in [1, 2]:
            var t1 = perf_counter_ns()
            for c in range(n_cases):
                for e in range(K):
                    off_gu[e] = Int(hids.unsafe_ptr()[c * K + e]) * eb_gu
                    off_d[e] = Int(hids.unsafe_ptr()[c * K + e]) * eb_d
                moe_cpu(hg.unsafe_ptr(), hu.unsafe_ptr(), hd.unsafe_ptr(), q80 == 1, off_gu, off_d, K,
                        hw.unsafe_ptr() + c * K, hx.unsafe_ptr() + c * D_MODEL, xp, hc, hvc, s, yc + c * D_MODEL, st)
            tst.append(perf_counter_ns() - t1)
        print("parallelism_level", parallelism_level())
        print("stage A (gate/up 18.4 MB) us:", Float64(tst[0]) / Float64(n_cases) / 1000.0, " GB/s:", Float64(n_cases * K * 2 * eb_gu) / Float64(tst[0]))
        print("stage B (down 12.3 MB) us:", Float64(tst[1]) / Float64(n_cases) / 1000.0, " GB/s:", Float64(n_cases * K * eb_d) / Float64(tst[1]))

        var nbytes = n_cases * K * (2 * eb_gu + eb_d)
        print("{\"cases\": ", n_cases, ", \"bit_identical_cases\": ", same_cases, ", \"diff_floats\": ", diff_floats,
              ", \"cpu_us_per_token_layer\": ", Float64(dt) / Float64(n_cases) / 1000.0, ", \"cpu_gbs\": ",
              Float64(nbytes) / Float64(dt), "}")
