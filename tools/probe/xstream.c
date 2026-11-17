// M8 probe driver: cudart stream (like ggml-cuda) + Mojo launches on it, then the same under CUDA graph capture.
#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdint.h>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s: %s\n", #x, cudaGetErrorString(e)); return 1; } } while (0)
int main(int argc, char ** argv) {
    void * lib = dlopen(argv[1], RTLD_NOW | RTLD_GLOBAL);
    if (!lib) { printf("dlopen: %s\n", dlerror()); return 1; }
    int64_t (*pctx)(int32_t) = dlsym(lib, "probe_ctx");
    int32_t (*plaunch)(int32_t, int64_t, int64_t, int32_t) = dlsym(lib, "probe_launch");
    for (int dev = 0; dev < 2; dev++) {
        CK(cudaSetDevice(dev)); CK(cudaFree(0));
        CUcontext cur; cuCtxGetCurrent(&cur);
        int64_t mctx = pctx(dev);
        printf("dev %d: cudart ctx %p, Mojo ctx %p -> %s\n", dev, (void *) cur, (void *) mctx, (int64_t) cur == mctx ? "SAME" : "DIFFERENT");
        cudaStream_t st; CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
        int32_t * d; CK(cudaMalloc((void **) &d, 4)); CK(cudaMemset(d, 0, 4));
        if (plaunch(dev, (int64_t) st, (int64_t) d, 5) != 0) return 1;
        CK(cudaStreamSynchronize(st));
        int32_t h = 0; CK(cudaMemcpy(&h, d, 4, cudaMemcpyDeviceToHost));
        printf("dev %d: direct launch on the external stream: %d (want 5)\n", dev, h);
        cudaGraph_t g; cudaGraphExec_t ge;
        CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeRelaxed));
        int r = plaunch(dev, (int64_t) st, (int64_t) d, 7);
        cudaError_t ce = cudaStreamEndCapture(st, &g);
        printf("dev %d: launch during capture returned %d, end capture: %s\n", dev, r, cudaGetErrorString(ce));
        if (ce != cudaSuccess) return 1;
        size_t n = 0; cudaGraphGetNodes(g, NULL, &n);
        CK(cudaGraphInstantiate(&ge, g, 0));
        CK(cudaGraphLaunch(ge, st)); CK(cudaGraphLaunch(ge, st)); CK(cudaStreamSynchronize(st));
        CK(cudaMemcpy(&h, d, 4, cudaMemcpyDeviceToHost));
        printf("dev %d: graph nodes %zu, after 2 replays: %d (want 19)\n", dev, n, h);
    }
    return 0;
}
