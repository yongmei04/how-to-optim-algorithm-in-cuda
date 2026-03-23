#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>
#include <algorithm>
#include <iostream>
#include <cuda_fp16.h>
#include <assert.h>
#include <cub/cub.cuh>

using namespace std;

constexpr int kN = 1024 * 1024 * 1024;
constexpr int kPackSize = 4;
constexpr int kBlockSize = 256;
constexpr int kNumWaves = 1;
constexpr int kMaxVal = 456;
constexpr int kNumStages = 3;
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

__device__ __forceinline__ void commit_group()
{
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int num>
__device__ __forceinline__ void wait()
{
    asm volatile("cp.async.wait_group %0;\n" ::"n"(num));
}


__device__ __forceinline__ int stage_inc(int pos)
{
    int curr = (pos + 1) % kNumStages;
    return curr;
}

__device__ __forceinline__ float* get_gmem_addr(float *gmem, unsigned int gmem_indx)
{
    unsigned int gmem_pack_indx = gmem_indx * kPackSize;
    float* gmem_addr = reinterpret_cast<float *>(&gmem[gmem_pack_indx]);
    return gmem_addr;
}

__device__ __forceinline__ float* get_smem_addr(float *smem, int curr, unsigned int tid)
{
    unsigned int smem_pack_indx = tid * kPackSize;
    float* smem_addr = reinterpret_cast<float *>(&smem[curr * kSmemStageSize + smem_pack_indx]);
    return smem_addr;
}

__device__ __forceinline__ void preload_one_stage(float *gmem,
                                                  float *smem,
                                                  int curr,
                                                  unsigned int gmem_indx,
                                                  unsigned int tid)
{
    float* gmem_addr = get_gmem_addr(gmem, gmem_indx);
    float* smem_addr = get_smem_addr(smem, curr, tid);

    uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_addr));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, 16;\n" ::"r"(smem_int_ptr), "l"(gmem_addr));
}

//__global__ void reduce_v12(float *g_idata,float *g_odata, int n, int *error_cnt)
__global__ void reduce_v12(float *g_idata,float *g_odata, int n)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int block_size = blockDim.x;
    const int grid_size = block_size * gridDim.x;
    const int total_packs = n / kPackSize;
    extern __shared__ float smem[];

    unsigned int gmem_indx = bid * block_size + tid;

    int preload_curr = -1, read_curr = -1;
    float sum = 0.0f;
    //int mismatch = 0;
    unsigned int preload_gmem_indx = gmem_indx;

    for (int preload_cnt = 0;
         preload_gmem_indx < total_packs && preload_cnt < kNumStages - 1;
         preload_gmem_indx += grid_size, preload_cnt++) {
        preload_curr = stage_inc(preload_curr);
        preload_one_stage(g_idata, smem, preload_curr, preload_gmem_indx, tid);
        commit_group();
    }

    for (; gmem_indx < total_packs && preload_gmem_indx < total_packs; gmem_indx += grid_size) {
        wait<kNumStages - 2>();
        read_curr = stage_inc(read_curr);

        float4* smem_addr_v4 = reinterpret_cast<float4*>(get_smem_addr(smem, read_curr, tid));
        float4 val = *smem_addr_v4;
        sum += val.x;
        sum += val.y;
        sum += val.z;
        sum += val.w;

        /*float *gmem_addr = get_gmem_addr(g_idata, gmem_indx);
        if (val.x != gmem_addr[0] || val.y != gmem_addr[1] || val.z != gmem_addr[2] || val.w != gmem_addr[3])
            mismatch++;*/

        if (preload_gmem_indx < total_packs) {
            preload_curr = stage_inc(preload_curr);
            preload_one_stage(g_idata, smem, preload_curr, preload_gmem_indx, tid);
            commit_group();
            preload_gmem_indx += grid_size;
        }
    }

    if (gmem_indx < total_packs) {
        wait<0>();
        for (; gmem_indx < total_packs; gmem_indx += grid_size) {
            read_curr = stage_inc(read_curr);

            float4* smem_ptr_v4 = reinterpret_cast<float4*>(get_smem_addr(smem, read_curr, tid));
            float4 val = *smem_ptr_v4;
            sum += val.x;
            sum += val.y;
            sum += val.z;
            sum += val.w;

            /*float *gmem_addr = get_gmem_addr(g_idata, gmem_indx);
            if (val.x != gmem_addr[0] || val.y != gmem_addr[1] || val.z != gmem_addr[2] || val.w != gmem_addr[3])
                mismatch++;*/
        }
    }

    /*if (mismatch > 0)
         atomicAdd(error_cnt, mismatch);*/

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

    /*int error_cnt = 0;
    int *error_cnt_d;
    cudaMalloc((void **)&error_cnt_d, sizeof(int));
    cudaMemcpy(error_cnt_d, &error_cnt, sizeof(int), cudaMemcpyHostToDevice);
    reduce_v12<<<block_num, kBlockSize, kSmemSize>>>(d_a, g_odata, kN, error_cnt_d); */
    reduce_v12<<<block_num, kBlockSize, kSmemSize>>>(d_a, g_odata, kN);
    assert(!cudaGetLastError());
    /*cudaMemcpy(&error_cnt, error_cnt_d, sizeof(int), cudaMemcpyDeviceToHost);
    assert(error_cnt == 0);
    reduce_v12<<<1, kBlockSize, kSmemSize>>>(g_odata, g_final_data, block_num, error_cnt_d);*/
    reduce_v12<<<1, kBlockSize, kSmemSize>>>(g_odata, g_final_data, block_num);
    assert(!cudaGetLastError());
    /*cudaMemcpy(&error_cnt, error_cnt_d, sizeof(int), cudaMemcpyDeviceToHost);
    assert(error_cnt == 0);*/

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
    //cudaFree(error_cnt_d);
    return 0;
}
