# Matmul优化过程的学习与理解
- 主要参考文档为 [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM)
- [NVIDIA_SGEMM_PRACTICE](https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE/tree/master?tab=readme-ov-file)
## 学习的目标
- 学习理解单指令多线程编程范式SIMT
- 学习理解SIMT范式中的线程组织结构
- 学习理解SIMT范式中的内存组织结构
- 学习理解SIMT范式中提高运算强度的优化逻辑和过程
### 学习理解SIMT范式
- SIMT范式在编码上看起来是一个单线程模式的代码
```c++
  // `if` condition is necessary for when M or N aren't multiples of 32.
  if (row < M && col < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[row * K + i] * B[i * N + col];
    }
    // C = α*(A@B)+β*C
    C[row * N + col] = alpha * tmp + beta * C[row * N + col];
  }
```
- SIMT在编译器中被优化为了多线程执行模式，上述的单线程代码块类似于一个template 模板，在模板代码执行前会计算实际运行线程的索引（x,y），以此多线程执行时各自加载对应索引的数据来互不干涉的运行
```c++
  // compute position in C that this thread is responsible for
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  // `if` condition is necessary for when M or N aren't multiples of 32.
  if (x < M && y < N) {
  // 单线程模板代码
  }

```
- SIMT中的索引运算用到的blockIdx、blockDim、threadIdx通过全局获取以及范式传入

### SIMT的线程结构
### SIMT的内存缓存结构
- 全局内存
- 共享缓存
- 寄存器
### SIMT提高运算强度的优化分析
- SIMT运算强度上不来的主要原因，1）计算访存比低：每次迭代需要进行一次FMA（乘累加）和两次全局内存读取，计算访存比1/2；2）访存量大：访问全局内存，C矩阵每个元素计算需要访问2K个单精度浮点数，完成全部计算需要 2*K*M*N；
- 计算/内存/缓存设施的cycle，全局内存访问延迟高（几百cycle）、共享缓存访问延迟中（几十cycle）、寄存器访问延迟低（几个cycle），计算累加器一般为1～3个cycle完成
### SIMT提高运算强度的优化方法
- 共享缓存-避免相同位置元素被重复读取，降低访存量，直接的效果就是降低了访问延时
- 一维thread tile优化，1）降低全局内存访存量，访存量降至1/64；2） 计算访存比：引入thread tile，利用单个线程负责多个元素计算，增加计算访存比；当TM=8时，每执行共享内存As的8个次访存指令和共享内存Bs的1个访存指令，可执行8次计算指令，相比初始版本的计算访存比1:2，提高至8:9，有效隐藏访存延迟；
- 利用二维thread tile优化，1）全局访存量降低为1/128； 2）访存比也得到了相应的下降，运算强度提高1倍
- 寄存器缓存共享内存
- 向量内存指令FLOAT4优化，结合寄存器缓存优化
- 数据预取，双缓存通道，降低同步次数，将读和写分开，计算数据读取一块存储空间同时，可以同时向另一块内存写入下一轮依赖的数据，因此，只需要保证计算前待读取共享内存完成写入，即一次同步即可
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
const int x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
// 这里y=row，如果不分block tilling的话，因为为 y=row= blockid.y * dimy + threadid.y(二维 thread)， 如果是一维的话，则直接为 blockid.y * dimy
const int y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

if (x < M && y < N) {
  float tmp = 0.0;
  for (int i = 0; i < K; ++i) {
    tmp += A[x * K + i] * B[i * N + y];
  }
  C[x * N + y] = alpha * tmp + beta * C[x * N + y];
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
