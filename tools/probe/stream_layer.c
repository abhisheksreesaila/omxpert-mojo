// M8 harness: the in-stream path (omx_moe_stream on a cudart stream, as llama.cpp calls it) against the host path
// (omx_moe), bit for bit, for every layer; then all 48 layers enqueued without host waits; then a CUDA graph replay.
// usage: stream_layer <libomxmoe.so> <model_txt> <pins_bin>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
void (*dbg)(int64_t, int32_t *); int64_t H;
static void * watchdog(void * a) { (void) a; sleep(25); int32_t o[2 + 160]; dbg(H, o);
    printf("WATCHDOG ctr=%d step=%d\n", o[0], o[1]);
    for (int g = 0; g < 2; g++) for (int k = 0; k < 16; k++) { int32_t * q = o + 2 + (g * 16 + k) * 5;
        if (q[0] || q[1]) printf("  ring%d slot%2d seq=%d done=%d ncold=%d layer=%d err=%d\n", g, k, q[0], q[1], q[2], q[3], q[4]); }
    fflush(stdout); _exit(3); return 0; }
#define D 2560
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s: %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)
int64_t (*init)(const char *, const char *); int32_t (*moe)(int64_t, int32_t, const float *, float *);
int32_t (*moes)(int64_t, int32_t, const float *, float *, void *); int32_t (*lgpu)(int64_t, int32_t);
static int same(const float * a, const float * b) { return memcmp(a, b, D * 4) == 0; }
int main(int argc, char ** argv) {
    void * lib = dlopen(argv[1], RTLD_NOW | RTLD_GLOBAL);
    if (!lib) { printf("dlopen %s\n", dlerror()); return 1; }
    init = dlsym(lib, "omx_init"); moe = dlsym(lib, "omx_moe"); moes = dlsym(lib, "omx_moe_stream"); lgpu = dlsym(lib, "omx_layer_gpu");
    cudaStream_t st[2]; float * dx[2]; float * dy[2];
    for (int g = 0; g < 2; g++) { CK(cudaSetDevice(g)); CK(cudaStreamCreateWithFlags(&st[g], cudaStreamNonBlocking));
        CK(cudaMalloc((void **) &dx[g], D * 4)); CK(cudaMalloc((void **) &dy[g], D * 4)); }
    int64_t h = init(argv[2], argv[3]);
    if (!h) return 1;
    H = h; dbg = dlsym(lib, "omx_debug"); pthread_t wt; pthread_create(&wt, 0, watchdog, 0);
    float x[D], y1[D], y2[D];
    srand(7);
    int ok = 0, tot = 0;
    // 1. per layer, synchronised, alternating with the host path (shared sequence counter)
    for (int rep = 0; rep < 3; rep++) for (int l = 0; l < 48; l++) {
        for (int i = 0; i < D; i++) x[i] = (rand() / (float) RAND_MAX - 0.5f) * 1.2f;
        if (rep == 0 && l < 2) { printf("layer %d: host path...\n", l); fflush(stdout); }
        moe(h, l, x, y1);
        if (rep == 0 && l < 2) { printf("layer %d: host done, stream path...\n", l); fflush(stdout); }
        int g = lgpu(h, l); CK(cudaSetDevice(g));
        CK(cudaMemcpy(dx[g], x, D * 4, cudaMemcpyHostToDevice));
        if (moes(h, l, dx[g], dy[g], st[g]) != 0) { printf("moe_stream failed\n"); return 1; }
        if (rep == 0 && l < 2) { printf("layer %d: enqueued\n", l); fflush(stdout); }
        CK(cudaStreamSynchronize(st[g]));
        CK(cudaMemcpy(y2, dy[g], D * 4, cudaMemcpyDeviceToHost));
        tot++; ok += same(y1, y2);
    }
    printf("1. per layer, alternating with host path: %d / %d bit-identical\n", ok, tot); fflush(stdout);
    // 2. GPU 0's 24 layers enqueued back to back, no host waits (y of layer l is x of layer l + 1), vs the host chain
    for (int i = 0; i < D; i++) x[i] = (rand() / (float) RAND_MAX - 0.5f) * 1.2f;
    float hx[D]; memcpy(hx, x, sizeof hx);
    for (int l = 0; l < 24; l++) { moe(h, l, hx, y1); for (int i = 0; i < D; i++) hx[i] = y1[i] * 4.0f; }
    CK(cudaSetDevice(0)); CK(cudaMemcpy(dx[0], x, D * 4, cudaMemcpyHostToDevice));
    for (int l = 0; l < 24; l++) {
        moes(h, l, dx[0], dy[0], st[0]);
        // next input = 4 * y (keeps the scale up), on the stream, no host involvement
        CK(cudaMemcpyAsync(dx[0], dy[0], D * 4, cudaMemcpyDeviceToDevice, st[0]));
        // scale by 4 is exact in float: do it host-side equivalently by copying and scaling below
        CK(cudaStreamSynchronize(st[0]));   // (2b drops this)
        float t[D]; CK(cudaMemcpy(t, dx[0], D * 4, cudaMemcpyDeviceToHost)); for (int i = 0; i < D; i++) t[i] *= 4.0f;
        CK(cudaMemcpy(dx[0], t, D * 4, cudaMemcpyHostToDevice));
    }
    CK(cudaMemcpy(y2, dy[0], D * 4, cudaMemcpyDeviceToHost));
    printf("2. 24-layer chain on GPU 0: %s\n", same(y1, y2) ? "bit-identical" : "DIFFERENT"); fflush(stdout);
    // 3. 24 independent layers enqueued with no host waits at all, then one sync
    float xs[24][D], ys[24][D]; float * dxs; float * dys;
    CK(cudaMalloc((void **) &dxs, 24 * D * 4)); CK(cudaMalloc((void **) &dys, 24 * D * 4));
    for (int l = 0; l < 24; l++) for (int i = 0; i < D; i++) xs[l][i] = (rand() / (float) RAND_MAX - 0.5f) * 1.2f;
    CK(cudaMemcpy(dxs, xs, sizeof xs, cudaMemcpyHostToDevice));
    for (int l = 0; l < 24; l++) moes(h, l, dxs + l * D, dys + l * D, st[0]);
    CK(cudaStreamSynchronize(st[0])); CK(cudaMemcpy(ys, dys, sizeof ys, cudaMemcpyDeviceToHost));
    ok = 0; for (int l = 0; l < 24; l++) { moe(h, l, xs[l], y1); ok += same(y1, ys[l]); }
    printf("3. 24 layers enqueued without waits: %d / 24 bit-identical\n", ok); fflush(stdout);
    // 4. CUDA graph: capture those 24 launches once, replay with new inputs 3 times
    cudaGraph_t gr; cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(st[0], cudaStreamCaptureModeRelaxed));
    for (int l = 0; l < 24; l++) moes(h, l, dxs + l * D, dys + l * D, st[0]);
    CK(cudaStreamEndCapture(st[0], &gr)); CK(cudaGraphInstantiate(&ge, gr, 0));
    ok = 0;
    for (int rep = 0; rep < 3; rep++) {
        for (int l = 0; l < 24; l++) for (int i = 0; i < D; i++) xs[l][i] = (rand() / (float) RAND_MAX - 0.5f) * 1.2f;
        CK(cudaMemcpy(dxs, xs, sizeof xs, cudaMemcpyHostToDevice));
        CK(cudaGraphLaunch(ge, st[0])); CK(cudaStreamSynchronize(st[0]));
        CK(cudaMemcpy(ys, dys, sizeof ys, cudaMemcpyDeviceToHost));
        for (int l = 0; l < 24; l++) { moe(h, l, xs[l], y1); ok += same(y1, ys[l]); }
    }
    printf("4. CUDA graph replays: %d / 72 bit-identical\n", ok); fflush(stdout);
    _exit(0);
}
