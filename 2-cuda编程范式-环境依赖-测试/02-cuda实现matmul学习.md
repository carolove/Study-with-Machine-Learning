# Matmul优化过程的学习与理解
- 主要参考文档为 [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM)
- [NVIDIA_SGEMM_PRACTICE](https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE/tree/master?tab=readme-ov-file)
## matmul代码优化讲解和个人理解
### 最基础代码结构
```
__global__ void sgemm_naive(int Height, int Width, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
  // compute position in C that this thread is responsible for
  // 这个x、y是对应到矩阵中的位置，也就是说一个thread最后只计算目标c的一个element, 因为这个thread
  // 一个sm调度中落在同一个sm中的thread的id可能是并不是一个连续串
  // 这里需要着重去看 nvidia cuda c++ guide 的share memory内容，https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=matrix%20multiply#shared-memory
  // 这个文档说的非常清晰，特别是用row代替y，用col 代替x，用with代替 N 等等，比较清晰的显示出来代码的逻辑
  
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < Height && col < Width) {
        float tmp = 0.0;
        for (uint i=0; i< K;i++) {
            tmp+=A[row*K+i]*B[Width*i+col];
        }
    // 这里最终当前thread计算c中的一个element数据
        C[row*Width+col] = alpha*tmp+beta*C[row*Width+col];
    }
}

// create as many blocks as necessary to map all of C
dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);
// 32 * 32 = 1024 thread per block
// 这里的含义就是说block是一个二维的，因此threadid.x threadid.y 都是有值的
dim3 blockDim(32, 32, 1);
// launch the asynchronous execution of the kernel on the device
// The function call returns immediately on the host
sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);

```
### 全局内存加速-连续访问
```
// 这里的含义 blocksize其实和dim 是一个维度
// 这里y=row，如果不分block tilling的话，因为为 y=row= blockid.y * dimy + threadid.y(二维 thread)， 如果是一维的话，则直接为 blockid.y * dimy
// 精髓就在这里，因为threadDim定义是一个一维数组，因此threadid++的时候，也就是threadid.x ++的时候，会线性映射到同一个block中
    const uint row = blockIdx.y * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    const uint col = blockIdx.x * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

    if (row < Height && col < Width) {
        float tmp = 0.0;
        for (uint i =0;i< K;i++) {
            tmp += A[row*K+i] * B[i*Width+col];
        }
        C[row*Width + col] = alpha * tmp + beta * C[row*Width+col];
    }

// gridDim stays the same
dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
// make blockDim 1-dimensional, but don't change number of threads
// 将block作为一维，这样的话是一个线性block，可以tilling
dim3 blockDim(32 * 32);
sgemm_coalescing<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
```
### 全局内存连续访问+共享内存
```
/*
dim3 blockDim(1024); 只要是存在共享内存加速的，thread都应该是一维的，这样才便于访问数据做block分段划区
dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
mysgemm_v2<32><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
*/

template<const int BLOCK_SIZE>
__global__ void mysgemm_v2(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    // bx by用于As Bs subblock指针偏移
    int bx = blockIdx.x;
    int by = blockIdx.y;

    const int BM = BLOCK_SIZE;
    const int BN = BLOCK_SIZE;
    const int BK = BLOCK_SIZE;

    // tx ty用于数据实际指针偏移，也就是内存数据加载
    int tx = threadIdx.x % BN;
    int ty = threadIdx.x / BN;

    // 申请共享内存空间
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // 移动到当前block
    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    float tmp = 0.;
    for (int k = 0; k < K; k += BK) {
        // 缓存A_tile和B_tile
        // 单个thread只从A、B全局内存中缓存一个数据到As Bs中
        // 缓存结束后，对应的缓存数据，是有可能被其他并行thread再次访问的
        As[ty * BK + tx] = A[ty * K + tx];
        Bs[ty * BN + tx] = B[ty * N + tx];
        // 同步所有线程缓存完成
        // 只有多线程访问共享内存才有这个sync需要，属于单线程编码、多线程执行、多线程共享内存才需要
        __syncthreads();
        A += BK;
        B += BK * N;
        for (int i = 0; i < BK; i++) {
            tmp += As[ty * BK + i] * Bs[i * BN + tx];
        }
        // FMA计算需要读取缓存数据，在新一轮写入缓存前进行同步，确保所有线程计算完成
        __syncthreads();
    }
    C[ty * N + tx] = alpha * tmp + beta * C[ty * N + tx];
}
```
