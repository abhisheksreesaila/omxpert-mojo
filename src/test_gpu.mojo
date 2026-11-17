# M1 proof: run a reference case of one layer on the GPU (whole layer resident in VRAM), report the error against the
# float64 reference, then time random top-10 routings over all 512 experts (so nothing hides in the 6 MB L2).
# args: gate_file gate_off up_file up_off down_file down_off down_is_q80 eb_gu eb_d case_file out_file
from std.sys import argv
from std.time import perf_counter_ns
from std.random import random_ui64, seed
from std.math import ceildiv
from max.gpu.host import DeviceContext, DeviceBuffer
from gpu_moe import permute_x_kernel, gate_up_kernel, gate_up_kernel_reg, gate_up_kernel_split, GU_ROWS, down_kernel, down_kernel_par, ROWS_PER_BLOCK, D_MODEL, D_FF, TPB, WARPS_PER_BLOCK

comptime K = 10
comptime N_EXPERTS = 512
comptime ROUTES = 256


def load_tensor(ctx: DeviceContext, path: String, off: Int, n: Int) raises -> DeviceBuffer[DType.uint8]:
    var host = ctx.enqueue_create_host_buffer[DType.uint8](n)
    ctx.synchronize()
    var f = open(path, "r")
    _ = f.seek(off)
    var done = 0
    var chunk = 1 << 26
    while done < n:
        var b = f.read_bytes(min(chunk, n - done))
        var p = host.unsafe_ptr()
        for i in range(len(b)):
            p[done + i] = b[i]
        done += len(b)
    f.close()
    var dev = ctx.enqueue_create_buffer[DType.uint8](n)
    ctx.enqueue_copy(dst_buf=dev, src_buf=host)
    ctx.synchronize()
    return dev^


def main() raises:
    var a = argv()
    var eb_gu = Int(a[8])
    var eb_d = Int(a[9])
    var q80 = Int(a[7])
    with DeviceContext() as ctx:
        var gate = load_tensor(ctx, String(a[1]), Int(a[2]), N_EXPERTS * eb_gu)
        var up = load_tensor(ctx, String(a[3]), Int(a[4]), N_EXPERTS * eb_gu)
        var down = load_tensor(ctx, String(a[5]), Int(a[6]), N_EXPERTS * eb_d)

        # reference case: x, ids, w, ref
        var f = open(String(a[10]), "r")
        var cs = f.read_bytes()
        f.close()
        var hx = ctx.enqueue_create_host_buffer[DType.float32](D_MODEL)
        var hids = ctx.enqueue_create_host_buffer[DType.int32](K * (ROUTES + 1))
        var hw = ctx.enqueue_create_host_buffer[DType.float32](K)
        var hy = ctx.enqueue_create_host_buffer[DType.float32](D_MODEL)
        ctx.synchronize()
        var cb = cs.unsafe_ptr()
        for i in range(D_MODEL):
            hx.unsafe_ptr()[i] = (cb + 4 * i).bitcast[Float32]()[0]
        for e in range(K):
            hids.unsafe_ptr()[e] = (cb + D_MODEL * 4 + 4 * e).bitcast[Int32]()[0]
            hw.unsafe_ptr()[e] = (cb + D_MODEL * 4 + K * 4 + 4 * e).bitcast[Float32]()[0]
        # random sorted routings for timing
        seed(1)
        for rt in range(ROUTES):
            var picked = List[Int]()
            while len(picked) < K:
                var c = Int(random_ui64(0, N_EXPERTS - 1))
                if c not in picked:
                    picked.append(c)
            sort(picked)
            for e in range(K):
                hids.unsafe_ptr()[K * (rt + 1) + e] = Int32(picked[e])

        var x = ctx.enqueue_create_buffer[DType.float32](D_MODEL)
        var xp = ctx.enqueue_create_buffer[DType.float32](D_MODEL)
        var ids = ctx.enqueue_create_buffer[DType.int32](K * (ROUTES + 1))
        var w = ctx.enqueue_create_buffer[DType.float32](K)
        var h = ctx.enqueue_create_buffer[DType.float32](K * D_FF)
        var y = ctx.enqueue_create_buffer[DType.float32](D_MODEL)
        ctx.enqueue_copy(dst_buf=x, src_buf=hx)
        ctx.enqueue_copy(dst_buf=ids, src_buf=hids)
        ctx.enqueue_copy(dst_buf=w, src_buf=hw)

        var only = String(a[12]) if len(a) > 12 else String("all")

        @parameter
        def run(route: Int) raises:
            var ip = ids.unsafe_ptr() + K * route
            if only == "perm":
                ctx.enqueue_function[permute_x_kernel](x, xp, grid_dim=ceildiv(D_MODEL, TPB), block_dim=TPB)
            if only == "gu1":
              ctx.enqueue_function[gate_up_kernel](gate, up, ip, Int64(eb_gu), Int64(K), x, h,
                grid_dim=ceildiv(K * D_FF, WARPS_PER_BLOCK), block_dim=TPB)
            if only == "gu3":
              ctx.enqueue_function[gate_up_kernel_split](gate, up, ip, Int64(eb_gu), Int64(K), x, h,
                grid_dim=ceildiv(K * D_FF * 2, WARPS_PER_BLOCK), block_dim=TPB)
            if only == "all" or only == "gu":
              ctx.enqueue_function[gate_up_kernel_reg](gate, up, ip, Int64(eb_gu), Int64(K), x, h,
                grid_dim=ceildiv(K * D_FF // GU_ROWS, WARPS_PER_BLOCK), block_dim=TPB)
            if only == "down1":
              ctx.enqueue_function[down_kernel](down, Int64(q80), ip, w, Int64(eb_d), Int64(K), h, y,
                grid_dim=ceildiv(D_MODEL, WARPS_PER_BLOCK), block_dim=TPB)
            if only == "all" or only == "down":
              ctx.enqueue_function[down_kernel_par](down, Int64(q80), ip, w, Int64(eb_d), Int64(K), h, y,
                grid_dim=D_MODEL // ROWS_PER_BLOCK, block_dim=K * 32)

        run(0)
        ctx.enqueue_copy(dst_buf=hy, src_buf=y)
        ctx.synchronize()
        var refp = (cb + D_MODEL * 4 + K * 8).bitcast[Float64]()
        var maxref: Float64 = 0
        var maxerr: Float64 = 0
        for i in range(D_MODEL):
            maxref = max(maxref, abs(refp[i]))
            maxerr = max(maxerr, abs(Float64(hy.unsafe_ptr()[i]) - refp[i]))
        var fo = open(String(a[11]), "w")
        fo.write_bytes(Span(unsafe_ptr=hy.unsafe_ptr().bitcast[UInt8](), length=D_MODEL * 4))
        fo.close()

        # timing: ROUTES random routings back to back, best of 5 passes
        for rt in range(1, ROUTES + 1):
            run(rt)
        ctx.synchronize()
        var best = 1 << 62
        for _ in range(5):
            var t0 = perf_counter_ns()
            for rt in range(1, ROUTES + 1):
                run(rt)
            ctx.synchronize()
            best = min(best, perf_counter_ns() - t0)
        var per = Float64(best) / Float64(ROUTES)
        var nbytes = K * (2 * eb_gu + eb_d)
        print("{\"device\": \"gpu\", \"rel_err\": ", maxerr / maxref, ", \"us_per_token_layer\": ", per / 1000.0,
              ", \"gbs\": ", Float64(nbytes) / per, "}")
