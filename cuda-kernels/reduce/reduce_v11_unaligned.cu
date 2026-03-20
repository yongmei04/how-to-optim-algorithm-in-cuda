#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>
#include <algorithm>
#include <iostream>
#include <cuda_fp16.h>
#include <assert.h>
#include <cub/cub.cuh>
using namespace std;

constexpr int kN = 1024 * 1024 * 1024 + 17311;
constexpr int kPackSize = 4;
constexpr int kBlockSize = 1024;
constexpr int kNumWaves = 1;
constexpr int kMaxVal = 456;
constexpr int kNumStages = 2;
constexpr int kSmemStageSize = kBlockSize * kPackSize;
constexpr int kSmemSize = kSmemStageSize * kNumStages * sizeof(float);

int64_t GetNumBlocks(int64_t n) {
  int dev;
  {
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) { return err; }
  }
  int sm_count;
  {
    cudaError_t err = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
    if (err != cudaSuccess) { return err; }
  }
  int tpm;
  {
    cudaError_t err = cudaDeviceGetAttribute(&tpm, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
    if (err != cudaSuccess) { return err; }
  }
  int64_t num_blocks = std::max<int64_t>(1, std::min<int64_t>((n + kBlockSize - 1) / kBlockSize,
                                                   sm_count * (tpm / kBlockSize) * kNumWaves));
  return num_blocks;
}

__device__ __forceinline__ void preload_4(float *smem_ptr, const float *gmem_ptr)
{
    uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, 16;\n" ::"r"(smem_int_ptr), "l"(gmem_ptr));
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void preload_2(float *smem_ptr, const float *gmem_ptr)
{
    uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, 8;\n" ::"r"(smem_int_ptr), "l"(gmem_ptr));
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void preload_1(float *smem_ptr, const float *gmem_ptr)
{
    uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, 4;\n" ::"r"(smem_int_ptr), "l"(gmem_ptr));
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int num>
__device__ __forceinline__ void wait()
{
    asm volatile("cp.async.wait_group %0;\n" ::"n"(num));
}


__device__ __forceinline__ int inc(int pos)
{
    int curr = (pos + 1) % kNumStages;
    return curr;
}

__global__ void reduce_v11(float *g_idata,float *g_odata, unsigned int n, int *error_cnt)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int block_size = blockDim.x;
    const int grid_size = block_size * gridDim.x;
    const int total_packs = n / kPackSize;
    extern __shared__ float smem[];

    unsigned int gmem_indx = bid * block_size + tid;
    unsigned int gmem_pack_indx = gmem_indx * kPackSize;
    unsigned int smem_indx = tid * kPackSize;

    int curr = 0;

    float* gmem_addr = reinterpret_cast<float *>(&g_idata[gmem_pack_indx]);
    float* smem_addr = reinterpret_cast<float *>(&smem[curr * kSmemStageSize + smem_indx]);
    
    float sum = 0.0f;
    int mismatch = 0;

    if (gmem_indx < total_packs)
        preload_4(smem_addr, gmem_addr);

    for(; gmem_indx < total_packs; gmem_indx += grid_size) {
        wait<0>();
        //__syncthreads();
        #pragma unroll 4
        for (int j=0; j < kPackSize; j++) {
            if (smem[curr * kSmemStageSize + smem_indx + j] != gmem_addr[j])
                mismatch++;
            sum += smem[curr * kSmemStageSize + smem_indx + j];
       }

        if (gmem_indx + grid_size < total_packs) {
            curr = inc(curr);
            gmem_pack_indx = (gmem_indx + grid_size) * kPackSize;
            gmem_addr = reinterpret_cast<float*>(&g_idata[gmem_pack_indx]);
            smem_addr = reinterpret_cast<float*>(&smem[curr * kSmemStageSize + smem_indx]);
            preload_4(smem_addr, gmem_addr);
        }
    }

#if 1
    gmem_pack_indx = gmem_indx * kPackSize;
    if (gmem_pack_indx < n) {
        unsigned int residual = n - gmem_pack_indx;
        unsigned k = (residual + kPackSize - 1) / kPackSize * kPackSize;
        k >>= 1;
        #pragma unroll
        for (; k > 0; k >>=1) {
            curr = inc(curr);
            gmem_addr = reinterpret_cast<float*>(&g_idata[gmem_pack_indx]);
            smem_addr = reinterpret_cast<float*>(&smem[curr * kSmemStageSize + smem_indx]);
            if (k >= 2)
                preload_2(smem_addr, gmem_addr);
            else
                preload_1(smem_addr, gmem_addr);
            wait<0>();
            for (int j=0; j < k; j++) {
                if (smem[curr * kSmemStageSize + smem_indx + j] != gmem_addr[j])
                    mismatch++;
                sum += smem[curr * kSmemStageSize + smem_indx + j];
            }
            gmem_pack_indx = gmem_indx * kPackSize + k;
        }
    }
#else
    for (unsigned k = gmem_indx * kPackSize; k < n; k += grid_size)
        sum += g_idata[k];
#endif

    if (mismatch > 0)
         atomicAdd(error_cnt, mismatch);

    typedef cub::BlockReduce<float, kBlockSize> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    float block_sum = BlockReduce(temp_storage).Sum(sum);
    if (threadIdx.x == 0) g_odata[blockIdx.x] = block_sum;
}

bool check_d(double *out,double *res,int n){
    const double diff_thres = 0.0000001;
    for (int i=0; i<n; i++) {
        if (out[i] == res[i])
            continue;
        double diff = out[i] < res[i] ? res[i] - out[i] : out[i] - res[i];
        if (diff / res[i] >= diff_thres)
            return false;
    }
    return true;
}

int main(int argc, char **argv)
{
    float *a=(float *)malloc(kN*sizeof(float));
    float *d_a;
    cudaMalloc((void **)&d_a,kN*sizeof(float));

    const int64_t block_num = GetNumBlocks(kN);

    float *out=(float *)malloc(sizeof(float));
    float *g_odata;
    cudaMalloc((void **)&g_odata,block_num*sizeof(float));
    float *g_final_data;
    cudaMalloc((void **)&g_final_data,1*sizeof(float));

    for(int i=0;i<kN;i++){
        a[i]=(float)(i % kMaxVal) / kMaxVal;
    }
    double *res=(double *)malloc(sizeof(double));
    res[0] = 0;
    for(int i=0;i<kN;i++)
        res[0] += (double)a[i] * kMaxVal;

    cudaMemcpy(d_a,a,kN*sizeof(float),cudaMemcpyHostToDevice);;

    int error_cnt = 0;
    reduce_v11<<<block_num, kBlockSize, kSmemSize>>>(d_a, g_odata, kN, &error_cnt);
    assert(!cudaGetLastError());
    assert(error_cnt == 0);
    reduce_v11<<<1, kBlockSize, kSmemSize>>>(g_odata, g_final_data, block_num, &error_cnt);
    assert(!cudaGetLastError());
    assert(error_cnt == 0);

    cudaMemcpy(out,g_final_data,1*sizeof(float),cudaMemcpyDeviceToHost);
    double out_d = (double)out[0] * kMaxVal;
    if (check_d(&out_d, res, 1)) printf("the answer is right\n");
    else{
        printf("the answer is wrong\n");
        printf("out:    %lf\n",out_d);
        printf("res[0]: %lf\n", res[0]);
    }

    cudaFree(d_a);
    cudaFree(g_odata);
    free(a);
    free(out);
    return 0;
}
