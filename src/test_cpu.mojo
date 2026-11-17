# M1/M2 helper: run one reference case of one layer on the CPU and report the error against the float64 reference.
# args: gate_file gate_off up_file up_off down_file down_off down_is_q80 eb_gu eb_d case_file out_file
from std.sys import argv
from std.time import perf_counter_ns
from std.memory import alloc
from cpu_moe import moe_cpu, D_MODEL, D_FF

comptime K = 10


def read_at(path: String, off: Int, n: Int) raises -> List[UInt8]:
    var f = open(path, "r")
    _ = f.seek(off)
    var b = f.read_bytes(n)
    f.close()
    return b^


def main() raises:
    var a = argv()
    var eb_gu = Int(a[8])
    var eb_d = Int(a[9])
    var cs = read_at(String(a[10]), 0, D_MODEL * 4 + K * 4 + K * 4 + D_MODEL * 8)
    var cp = alloc[UInt8](len(cs))
    for i in range(len(cs)):
        cp[i] = cs[i]
    var x = cp.bitcast[Float32]()
    var ids = (cp + D_MODEL * 4).bitcast[Int32]()
    var w = (cp + D_MODEL * 4 + K * 4).bitcast[Float32]()
    var ref_ = (cp + D_MODEL * 4 + K * 8).bitcast[Float64]()

    # the 10 selected experts, packed one after another
    var gate = alloc[UInt8](K * eb_gu)
    var up = alloc[UInt8](K * eb_gu)
    var down = alloc[UInt8](K * eb_d)
    var off_gu = alloc[Int](K)
    var off_d = alloc[Int](K)
    for e in range(K):
        var id = Int(ids[e])
        var g = read_at(String(a[1]), Int(a[2]) + id * eb_gu, eb_gu)
        var u = read_at(String(a[3]), Int(a[4]) + id * eb_gu, eb_gu)
        var d = read_at(String(a[5]), Int(a[6]) + id * eb_d, eb_d)
        for i in range(eb_gu):
            gate[e * eb_gu + i] = g[i]
            up[e * eb_gu + i] = u[i]
        for i in range(eb_d):
            down[e * eb_d + i] = d[i]
        off_gu[e] = e * eb_gu
        off_d[e] = e * eb_d

    var xp = alloc[Float32](D_MODEL)
    var h = alloc[Float32](K * D_FF)
    var s = alloc[Float32](K * D_MODEL)
    var y = alloc[Float32](D_MODEL)
    var q80 = a[7] == "1"
    moe_cpu(gate, up, down, q80, off_gu, off_d, K, w, x, xp, h, s, y)
    var best = 1 << 62
    for _ in range(20):
        var t0 = perf_counter_ns()
        moe_cpu(gate, up, down, q80, off_gu, off_d, K, w, x, xp, h, s, y)
        best = min(best, perf_counter_ns() - t0)

    var maxref: Float64 = 0
    var maxerr: Float64 = 0
    for i in range(D_MODEL):
        maxref = max(maxref, abs(ref_[i]))
        maxerr = max(maxerr, abs(Float64(y[i]) - ref_[i]))
    var f = open(String(a[11]), "w")
    f.write_bytes(Span(unsafe_ptr=y.bitcast[UInt8](), length=D_MODEL * 4))
    f.close()
    var nbytes = K * (2 * eb_gu + eb_d)
    print("{\"device\": \"cpu\", \"rel_err\": ", maxerr / maxref, ", \"best_us\": ", Float64(best) / 1000.0,
          ", \"gbs\": ", Float64(nbytes) / Float64(best), "}")
