// Standalone coverage for the LDS-DMA path in cooperative_groups::memcpy_async.
//
// Build and run:
//   hipcc -std=c++17 -O3 --offload-arch=$(offload-arch) memcpy_async_lds_dma_repro.hip -o repro
//   ./repro
//
// This is a convenience reproducer for validating the change on hardware. The coverage belongs in
// projects/hip-tests/catch/unit/deviceLib/memcpy_async.cc before this is proposed upstream.
//
// The accelerated path is only taken for element types aligned to at least a dword, so int-typed
// copies exercise it while char-typed copies exercise the traditional fallback. Block sizes include
// partial waves and non-multiples of the wave size, because the LDS destination base is derived per
// wave and the hardware adds lane_id * 4 to it.
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <hip/cooperative_groups/memcpy_async.h>
#include <cstdio>
#include <vector>

namespace cg = cooperative_groups;

#define CAP 8192

__global__ void reportPaths(int *out)
{
    out[0] = __builtin_amdgcn_is_invocable(__builtin_amdgcn_load_to_lds) ? 1 : 0;
#if __has_builtin(__builtin_amdgcn_global_load_async_to_lds_b128)
    out[1] = __builtin_amdgcn_is_invocable(__builtin_amdgcn_global_load_async_to_lds_b128) ? 1 : 0;
#else
    out[1] = 0;
#endif
}

template <typename T>
__global__ void globalToLds(const T *in, T *out, int bytes)
{
    __shared__ __align__(16) unsigned char raw[CAP];
    cg::thread_block block = cg::this_thread_block();

    for (int i = block.thread_rank(); i < CAP; i += block.num_threads()) raw[i] = 0;
    block.sync();

    cg::memcpy_async(block, reinterpret_cast<T *>(raw), in, (size_t)bytes);
    block.sync();

    unsigned char *o = reinterpret_cast<unsigned char *>(out);
    for (int i = block.thread_rank(); i < bytes; i += block.num_threads()) o[i] = raw[i];
}

template <typename T>
__global__ void ldsToGlobal(const T *in, T *out, int bytes)
{
    __shared__ __align__(16) unsigned char raw[CAP];
    cg::thread_block block = cg::this_thread_block();

    const unsigned char *i8 = reinterpret_cast<const unsigned char *>(in);
    for (int i = block.thread_rank(); i < bytes; i += block.num_threads()) raw[i] = i8[i];
    block.sync();

    cg::memcpy_async(block, out, reinterpret_cast<T *>(raw), (size_t)bytes);
    block.sync();
}

static int checks = 0, failures = 0;

template <typename T>
static void run(const char *tag, int threads, int bytes, const std::vector<unsigned char> &ref,
                unsigned char *d_in, unsigned char *d_out, bool ldsToGlobalDir)
{
    std::vector<unsigned char> got(bytes, 0xAB);
    (void)hipMemset(d_out, 0xAB, CAP);

    if (ldsToGlobalDir)
        ldsToGlobal<T><<<1, threads>>>((const T *)d_in, (T *)d_out, bytes);
    else
        globalToLds<T><<<1, threads>>>((const T *)d_in, (T *)d_out, bytes);

    checks++;
    hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        printf("  FAIL %-9s threads=%-5d bytes=%-5d launch: %s\n", tag, threads, bytes,
               hipGetErrorString(err));
        failures++;
        return;
    }

    (void)hipMemcpy(got.data(), d_out, bytes, hipMemcpyDeviceToHost);
    for (int i = 0; i < bytes; i++) {
        if (got[i] != ref[i]) {
            printf("  FAIL %-9s threads=%-5d bytes=%-5d byte %d: got %u want %u\n", tag, threads,
                   bytes, i, got[i], ref[i]);
            failures++;
            return;
        }
    }
}

int main()
{
    hipDeviceProp_t prop;
    if (hipGetDeviceProperties(&prop, 0) != hipSuccess) {
        printf("no device\n");
        return 1;
    }

    int *d_paths, h_paths[2] = {0, 0};
    (void)hipMalloc(&d_paths, 2 * sizeof(int));
    reportPaths<<<1, 1>>>(d_paths);
    (void)hipDeviceSynchronize();
    (void)hipMemcpy(h_paths, d_paths, sizeof(h_paths), hipMemcpyDeviceToHost);

    printf("device      : %s (wavefront %d)\n", prop.gcnArchName, prop.warpSize);
    printf("load_to_lds : %s\n", h_paths[0] ? "invocable" : "not invocable");
    printf("async_to_lds: %s\n", h_paths[1] ? "invocable" : "not invocable");
    printf("expected memcpy_async path for dword-aligned global->LDS: %s\n\n",
           h_paths[1] ? "async builtins" : (h_paths[0] ? "LDS DMA" : "traditional copy"));

    unsigned char *d_in, *d_out;
    (void)hipMalloc(&d_in, CAP);
    (void)hipMalloc(&d_out, CAP);
    std::vector<unsigned char> ref(CAP);
    for (size_t i = 0; i < ref.size(); i++) ref[i] = (unsigned char)(i * 7 + 3);
    (void)hipMemcpy(d_in, ref.data(), CAP, hipMemcpyHostToDevice);

    // whole waves, partial waves, and non-multiples of either wavefront size
    const int threadsList[] = {32, 64, 65, 100, 128, 129, 192, 200, 256, 320, 512, 1024};
    // dword multiples, ragged tails, counts below the group size, and large copies
    const int byteList[] = {4,    8,    12,   64,   100,  255,  256,  257, 258,
                            259,  1024, 1025, 1027, 2048, 4095, 4096, 8188};

    printf("global -> LDS, int elements (accelerated path where available)\n");
    for (int t : threadsList)
        for (int b : byteList) run<int>("int g2l", t, b, ref, d_in, d_out, false);

    printf("global -> LDS, char elements (traditional path)\n");
    for (int t : threadsList)
        for (int b : byteList) run<unsigned char>("char g2l", t, b, ref, d_in, d_out, false);

    printf("LDS -> global, int elements (traditional path)\n");
    for (int t : threadsList)
        for (int b : byteList) run<int>("int l2g", t, b, ref, d_in, d_out, true);

    printf("\n%d checks, %d failures -> %s\n", checks, failures,
           failures == 0 ? "ALL PASS" : "FAILURES");
    return failures == 0 ? 0 : 1;
}
