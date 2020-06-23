#ifndef CUDA_ERROR_CHECK
__device__ __host__ inline void CUDA_ERROR_CHECK_OUTPUT(cudaError_t code, const char *file, int line, bool abort=false) {
    if (code != cudaSuccess) {
        printf("%s(%d): CUDA Function Error: %s \n", file, line, cudaGetErrorString(code));
        if (abort) assert(0);
    }
}
#define CUDA_ERROR_CHECK(ans) { CUDA_ERROR_CHECK_OUTPUT((ans), __FILE__, __LINE__); }
#endif

#define CUDA_CHECK_AFTER_CALL() {CUDA_ERROR_CHECK(cudaGetLastError());}
#define VcudaDeviceSynchronize() {CUDA_ERROR_CHECK(cudaDeviceSynchronize())}
